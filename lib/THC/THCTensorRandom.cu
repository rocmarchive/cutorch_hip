#include "hip/hip_runtime.h"
#include "THCTensorRandom.h"
#include "THCDeviceUtils.cuh"
#include "THCGeneral.h"
#include "THCTensorCopy.h"
#include "THCTensorMath.h"
#include "THCReduceApplyUtils.cuh"
#ifdef CURAND_PATH
#include <curand.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
#else
#include "hip/hcc.h"
#include "MTGP/hiprand_mtgp32.h"
#endif

#ifdef THRUST_PATH
    #include <thrust/functional.h>
#else
    #include <bolt/amp/functional.h>
#endif

#define MAX_NUM_BLOCKS 64
#define BLOCK_SIZE 256

/* Sets up generator. Allocates but does not create the generator states. */
void initializeGenerator(THCState *state, Generator* gen)
{
#ifdef CURAND_PATH
  THCudaCheck(THCudaMalloc(state, (void**)&gen->gen_states, MAX_NUM_BLOCKS * sizeof(curandStateMtgp32)));
  THCudaCheck(THCudaMalloc(state, (void**)&gen->kernel_params, sizeof(mtgp32_kernel_params)));
#else
  assert(gen);
  gen->h_gen_states = new HipRandStateMtgp32;
  assert(gen->h_gen_states);
  hipStream_t currentStream = THCState_getCurrentStream(state);
  hc::accelerator_view* current_accl_view;
  hipHccGetAcceleratorView(currentStream, &current_accl_view);
  HipRandStateMtgp32_init(*current_accl_view, gen->h_gen_states);
#endif 
}

/* Frees memory allocated during setup. */
void destroyGenerator(THCState *state, Generator* gen)
{
#ifdef CURAND_PATH
  if (gen->gen_states)
  {
    THCudaCheck(THCudaFree(state, gen->gen_states));
    gen->gen_states = NULL;
  }
  if (gen->kernel_params)
  {
    THCudaCheck(THCudaFree(state, gen->kernel_params));
    gen->kernel_params = NULL;
  }
#else
  if (gen->h_gen_states) {
    HipRandStateMtgp32_release(gen->h_gen_states);
    delete gen->h_gen_states;
  }
#endif
}

/* Creates a new generator state given the seed. */
void createGeneratorState(THCState* state, Generator* gen, unsigned long seed)
{
#ifdef CURAND_PATH
  if (curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, gen->kernel_params) != CURAND_STATUS_SUCCESS)
  {
    THError("Creating MTGP constants failed.");
  }
  if (curandMakeMTGP32KernelState(gen->gen_states, mtgp32dc_params_fast_11213,
                                  gen->kernel_params, MAX_NUM_BLOCKS, seed) != CURAND_STATUS_SUCCESS)
  {
    THError("Creating MTGP kernel state failed.");
  }
#else
  hipStream_t currentStream = THCState_getCurrentStream(state);
  hc::accelerator_view* current_accl_view;
  hipHccGetAcceleratorView(currentStream, &current_accl_view);

  if (mtgp32_init_params_kernel(*current_accl_view, mtgp32_params_fast_11213, gen->h_gen_states)) {
    THError("Creating MTGP constants failed.");
  }

  // Using device API
  if (mtgp32_init_seed_kernel(*current_accl_view, gen->h_gen_states, seed)) {
    THError("Creating MTGP kernel state failed.");
  }

#endif
}

/* Initialize generator array (must be called before any other function) */
void THCRandom_init(THCState* state, int devices, int current_device)
{
  THCRNGState* rng_state = THCState_getRngState(state);
  rng_state->num_devices = devices;
  rng_state->gen = (Generator*)malloc(rng_state->num_devices * sizeof(Generator));
  for (int i = 0; i < rng_state->num_devices; ++i)
  {
    rng_state->gen[i].initf = 0;
    rng_state->gen[i].initial_seed = 0;
#ifdef CURAND_PATH
    rng_state->gen[i].gen_states = NULL;
    rng_state->gen[i].kernel_params = NULL;
#else
    rng_state->gen[i].h_gen_states = NULL;
#endif
  }
}

/* Destroy generators and free memory */
void THCRandom_shutdown(THCState* state)
{
  THCRNGState* rng_state = THCState_getRngState(state);
  if (rng_state->gen == NULL) return;
  for (int i = 0; i < rng_state->num_devices; ++i)
  {
    destroyGenerator(state, &rng_state->gen[i]);
  }
  free(rng_state->gen);
  rng_state->gen = NULL;
}

/* Manually set the generator seed */
static void THCRandom_manualSeedGen(THCState* state, Generator* gen, unsigned long seed)
{
  gen->initial_seed = seed;
  createGeneratorState(state, gen, seed);
  gen->initf = 1;
}

/* Get the generator for the current device */
Generator* THCRandom_getGenerator(THCState* state)
{
  THCRNGState* rng_state = THCState_getRngState(state);

  int device;
  THCudaCheck(hipGetDevice(&device));
  if (device >= rng_state->num_devices) THError("Invalid device index.");

  Generator* gen = &rng_state->gen[device];
  if (gen->initf == 0)
  {
    initializeGenerator(state, gen);
    THCRandom_manualSeedGen(state, gen, (unsigned long)time(0));
  }
  return gen;
}

#ifdef CURAND_PATH
struct curandStateMtgp32* THCRandom_generatorStates(struct THCState* state)
{
  return THCRandom_getGenerator(state)->gen_states;
}
#endif
/* Random seed */
unsigned long THCRandom_seed(THCState* state)
{
  unsigned long s = (unsigned long)time(0);
  THCRandom_manualSeed(state, s);
  return s;
}

unsigned long THCRandom_seedAll(THCState* state)
{
  unsigned long s = (unsigned long)time(0);
  THCRandom_manualSeedAll(state, s);
  return s;
}

/* Manually set the seed */
void THCRandom_manualSeed(THCState* state, unsigned long seed)
{
  Generator* gen = THCRandom_getGenerator(state);
  THCRandom_manualSeedGen(state, gen, seed);
}

void THCRandom_manualSeedAll(THCState* state, unsigned long seed)
{
  THCRNGState* rng_state = THCState_getRngState(state);
  int currentDevice;
  THCudaCheck(hipGetDevice(&currentDevice));
  for (int i = 0; i < rng_state->num_devices; ++i) {
    THCudaCheck(hipSetDevice(i));
    THCRandom_manualSeed(state, seed);
  }
  THCudaCheck(hipSetDevice(currentDevice));
}

/* Get the initial seed */
unsigned long THCRandom_initialSeed(THCState* state)
{
  return THCRandom_getGenerator(state)->initial_seed;
}

void THCRandom_getRNGState(THCState* state, THByteTensor *rng_state)
{
  Generator* gen = THCRandom_getGenerator(state);

  // The RNG state comprises the MTPG32 states and the seed.
#ifdef CURAND_PATH
  static const size_t states_size = MAX_NUM_BLOCKS * sizeof(curandStateMtgp32);
#else
  static const size_t states_size = MAX_NUM_BLOCKS * sizeof(HipRandStateMtgp32);
#endif
  static const size_t seed_size = sizeof(unsigned long);
  static const size_t total_size = states_size + seed_size;
  THByteTensor_resize1d(rng_state, total_size);
  THArgCheck(THByteTensor_nElement(rng_state) == total_size, 1, "RNG state is wrong size");
  THArgCheck(THByteTensor_isContiguous(rng_state), 1, "RNG state must be contiguous");
#ifdef CURAND_PATH
  THCudaCheck(hipMemcpy(THByteTensor_data(rng_state), gen->gen_states,
                         states_size, hipMemcpyDeviceToHost));
#else
  THCudaCheck(hipMemcpy(THByteTensor_data(rng_state), gen->h_gen_states,
                         states_size, hipMemcpyDeviceToHost));
#endif
  memcpy(THByteTensor_data(rng_state) + states_size, &gen->initial_seed, seed_size);
}

#ifdef CURAND_PATH
__global__ void set_rngstate_kernel(hipLaunchParm lp, curandStateMtgp32 *state, mtgp32_kernel_params *kernel)
{
  state[hipThreadIdx_x].k = kernel;
}
#else

#endif

void THCRandom_setRNGState(THCState* state, THByteTensor *rng_state)
{
  Generator* gen = THCRandom_getGenerator(state);

#ifdef CURAND_PATH
  static const size_t states_size = MAX_NUM_BLOCKS * sizeof(curandStateMtgp32);
#else
  static const size_t states_size = MAX_NUM_BLOCKS * sizeof(HipRandStateMtgp32);
#endif
  static const size_t seed_size = sizeof(unsigned long);
  static const size_t total_size = states_size + seed_size;
  THArgCheck(THByteTensor_nElement(rng_state) == total_size, 1, "RNG state is wrong size");
  THArgCheck(THByteTensor_isContiguous(rng_state), 1, "RNG state must be contiguous");

#ifdef CURAND_PATH
  THCudaCheck(hipMemcpy(gen->gen_states, THByteTensor_data(rng_state),
                         states_size, hipMemcpyHostToDevice));
  hipLaunchKernel(HIP_KERNEL_NAME(set_rngstate_kernel), dim3(1), dim3(MAX_NUM_BLOCKS), 0, THCState_getCurrentStream(state), 
      gen->gen_states, gen->kernel_params);
#else
  THCudaCheck(hipMemcpy(gen->h_gen_states, THByteTensor_data(rng_state),
                         states_size, hipMemcpyHostToDevice));
#endif
  memcpy(&gen->initial_seed, THByteTensor_data(rng_state) + states_size, seed_size);
}

#ifdef CURAND_PATH
#define GENERATE_KERNEL1(NAME, ARG1, CURAND_FUNC, TRANSFORM)                   \
__global__ void NAME(hipLaunchParm  lp, curandStateMtgp32 *state, int size, float *result, ARG1)  \
{                                                                              \
  int idx = hipBlockIdx_x * BLOCK_SIZE + hipThreadIdx_x;                             \
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;                     \
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {      \
    float x = CURAND_FUNC(&state[hipBlockIdx_x]);                                 \
    if (i < size) {                                                            \
      x = TRANSFORM;                                                           \
      result[i] = x;                                                           \
    }                                                                          \
  }                                                                            \
}

#define GENERATE_KERNEL2(NAME, ARG1, ARG2, CURAND_FUNC, TRANSFORM)                   \
__global__ void NAME(hipLaunchParm lp, curandStateMtgp32 *state, int size, float *result, ARG1, ARG2)  \
{                                                                                    \
  int idx = hipBlockIdx_x * BLOCK_SIZE + hipThreadIdx_x;                                   \
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;                           \
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {            \
    float x = CURAND_FUNC(&state[hipBlockIdx_x]);                                       \
    if (i < size) {                                                                  \
      x = TRANSFORM;                                                                 \
      result[i] = x;                                                                 \
    }                                                                                \
  }                                                                                  \
}
#else

#define GENERATE_KERNEL1(NAME, ARG1, HIPRAND_FUNC, FUNCTOR)                   \
void NAME(THCState* state, HipRandStateMtgp32 *rngstate, int size, float *result, ARG1)  \
{ \
  hipStream_t currentStream = THCState_getCurrentStream(state); \
  hc::accelerator_view* current_accl_view; \
  hipHccGetAcceleratorView(currentStream, &current_accl_view); \
  HIPRAND_FUNC##_kernel(*current_accl_view, rngstate, result, FUNCTOR); \
}

#define GENERATE_KERNEL2(NAME, ARG1, ARG2, HIPRAND_FUNC, FUNCTOR)                   \
void NAME(THCState* state, HipRandStateMtgp32 *rngstate, int size, float *result, ARG1, ARG2)  \
{                                                                                    \
  hipStream_t currentStream = THCState_getCurrentStream(state); \
  hc::accelerator_view* current_accl_view; \
  hipHccGetAcceleratorView(currentStream, &current_accl_view); \
  HIPRAND_FUNC##_kernel(*current_accl_view, rngstate, result, FUNCTOR);                                                 \
}

#endif
#ifdef CURAND_PATH
GENERATE_KERNEL2(generate_uniform, double a, double b, curand_uniform, x * (b-a) + a)
GENERATE_KERNEL1(generate_bernoulli, double p, curand_uniform, (float)x <= p)
GENERATE_KERNEL2(generate_normal, double mean, double stdv, curand_normal, (x * stdv) + mean)
GENERATE_KERNEL1(generate_geometric, double p, curand_uniform, (log(1-x) / log(p)) + 1)
GENERATE_KERNEL1(generate_exponential, double lambda, curand_uniform, (float)(-1. / lambda * log(1-x)))
GENERATE_KERNEL2(generate_cauchy, double median, double sigma, curand_uniform, (float)(median + sigma * tan(M_PI*(x-0.5))))
#else

// Adding All HC based constructors

struct user_uniform_functor {
  double _a;
  double _b;
  user_uniform_functor(double a, double b) __attribute__((hc, cpu)) : _a(a), _b(b) {}
  inline double operator()(const float& x) const __attribute__((hc, cpu)) {
    return x * (_b - _a) + _a;
  }
  // User should provide copy ctor
  user_uniform_functor(const user_uniform_functor&other) __attribute__((hc, cpu)) : _a(other._a), _b(other._b) { }
  // User should provide copy assign ctor
  user_uniform_functor& operator = (const user_uniform_functor&other) __attribute__((hc, cpu)) {
    _a = other._a;
    _b = other._b;
    return *this;
  }
};


struct user_bernoulli_functor {
  double _p;
  user_bernoulli_functor(double p) __attribute__((hc, cpu)) : _p(p) {}
  inline double operator()(const float& x) const __attribute__((hc, cpu)) {
    return (double)x <= _p;
  }
  // User should provide copy ctor
  user_bernoulli_functor(const user_bernoulli_functor&other) __attribute__((hc, cpu)) : _p(other._p) { }
  // User should provide copy assign ctor
  user_bernoulli_functor& operator = (const user_bernoulli_functor&other) __attribute__((hc, cpu)) {
    _p = other._p;
    return *this;
  }
};


struct user_normal_functor {
  double _stdv;
  double _mean;
  user_normal_functor(double stdv, double mean) __attribute__((hc, cpu)) : _stdv(stdv), _mean(mean) {}
  inline double operator()(const float& x) const __attribute__((hc, cpu)) {
    return (x * _stdv) + _mean;
  }
  // User should provide copy ctor
  user_normal_functor(const user_normal_functor&other) __attribute__((hc, cpu))
    : _stdv(other._stdv), _mean(other._mean) { }
  // User should provide copy assign ctor
  user_normal_functor& operator = (const user_normal_functor&other) __attribute__((hc, cpu)) {
    _stdv = other._stdv;
    _mean = other._mean;
    return *this;
  }
};

struct user_geometric_functor {
  double _p;
  user_geometric_functor(double p) __attribute__((hc, cpu)) : _p(p) {}
  inline double operator()(const float& x) const __attribute__((hc, cpu)) {
    return (hc::precise_math::log((double)(1 - x)) / hc::precise_math::log(_p)) + 1;
  }
  // User should provide copy ctor
  user_geometric_functor(const user_geometric_functor&other) __attribute__((hc, cpu)) : _p(other._p) { }
  // User should provide copy assign ctor
  user_geometric_functor& operator = (const user_geometric_functor&other) __attribute__((hc, cpu)) {
    _p = other._p;
    return *this;
  }
};

struct user_exponential_functor {
  double _lambda;
  user_exponential_functor(double lambda) __attribute__((hc, cpu)) : _lambda(lambda) {}
  inline double operator()(const float& x) const __attribute__((hc, cpu)) {
    return (double)(-1. / _lambda * hc::precise_math::log((double)(1 - x)));
  }
  // User should provide copy ctor
  user_exponential_functor(const user_exponential_functor&other) __attribute__((hc, cpu)) : _lambda(other._lambda) { }
  // User should provide copy assign ctor
  user_exponential_functor& operator = (const user_exponential_functor&other) __attribute__((hc, cpu)) {
    _lambda = other._lambda;
    return *this;
  }
};

struct user_cauchy_functor {
  double _median;
  double _sigma;
  user_cauchy_functor(double median, double sigma) __attribute__((hc, cpu)) : _median(median), _sigma(sigma) {}
  inline double operator()(const float& x) const __attribute__((hc, cpu)) {
    return (double)(_median + _sigma * hc::precise_math::tan((double)M_PI * (x - 0.5)));
  }
  // User should provide copy ctor
  user_cauchy_functor(const user_cauchy_functor&other) __attribute__((hc, cpu))
    : _median(other._median), _sigma(other._sigma) { }
  // User should provide copy assign ctor
  user_cauchy_functor& operator = (const user_cauchy_functor&other) __attribute__((hc, cpu)) {
    _median = other._median;
    _sigma = other._sigma;
    return *this;
  }
};


GENERATE_KERNEL2(generate_uniform, double a, double b, user_uniform, user_uniform_functor(a, b))
GENERATE_KERNEL1(generate_bernoulli, double p, user_uniform, user_bernoulli_functor(p))
GENERATE_KERNEL2(generate_normal, double mean, double stdv, user_normal, user_normal_functor(stdv,  mean))
GENERATE_KERNEL1(generate_geometric,  double p, user_uniform, user_geometric_functor(p))
GENERATE_KERNEL1(generate_exponential, double lambda, user_uniform, user_exponential_functor(lambda))
GENERATE_KERNEL2(generate_cauchy, double median, double sigma, user_uniform, user_cauchy_functor(median, sigma))
#endif

#undef GENERATE_KERNEL1
#undef GENERATE_KERNEL2

#ifdef CURAND_PATH
/* Separate kernel because curand_log_normal gets extra parameters. */
__global__ void generate_log_normal(hipLaunchParm lp, curandStateMtgp32 *state, int size, float *result, float mean, float stddev)
{
  int idx = hipBlockIdx_x * BLOCK_SIZE + hipThreadIdx_x;
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {
    float x = curand_log_normal(&state[hipBlockIdx_x], mean, stddev);
    if (i < size) {
      result[i] = x;
    }
  }
}
#else
/* Separate kernel because curand_log_normal gets extra parameters. */
void generate_log_normal(THCState* state, HipRandStateMtgp32 *rngstate, int size, float* result, float mean, float stddev) {
  hipStream_t currentStream = THCState_getCurrentStream(state); 
  hc::accelerator_view* current_accl_view; 
  hipHccGetAcceleratorView(currentStream, &current_accl_view); 
  user_log_normal_kernel(*current_accl_view, rngstate, result, mean, stddev);
}
#endif

#define NUM_BLOCKS min((int)THCCeilDiv(size, (ptrdiff_t) BLOCK_SIZE), MAX_NUM_BLOCKS)
THC_API void THCudaTensor_uniform(THCState* state, THCudaTensor *self_, double a, double b)
{
  THAssert(THCudaTensor_checkGPU(state, 1, self_));
  Generator* gen = THCRandom_getGenerator(state);
  THCudaTensor *self = THCudaTensor_newContiguous(state, self_);
  ptrdiff_t size = THCudaTensor_nElement(state, self);
  float *data = THCudaTensor_data(state, self);
  #ifdef CURAND_PATH
  hipLaunchKernel(HIP_KERNEL_NAME(generate_uniform), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state), 
      gen->gen_states, size, data, a, b);
  #else
    generate_uniform(state, gen->h_gen_states, size, data, a, b);
  #endif

  THCudaTensor_freeCopyTo(state, self, self_);
};

THC_API void THCudaTensor_bernoulli(THCState* state, THCudaTensor *self_, double p)
{
  THAssert(THCudaTensor_checkGPU(state, 1, self_));
  Generator* gen = THCRandom_getGenerator(state);
  THCudaTensor *self = THCudaTensor_newContiguous(state, self_);
  ptrdiff_t size = THCudaTensor_nElement(state, self);
  float *data = THCudaTensor_data(state, self);
  #ifdef CURAND_PATH
  hipLaunchKernel(HIP_KERNEL_NAME(generate_bernoulli), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state), 
      gen->gen_states, size, data, p);
  #else
  generate_bernoulli(state, gen->h_gen_states, size, data, p);
  #endif
  THCudaTensor_freeCopyTo(state, self, self_);
};

THC_API void THCudaTensor_normal(THCState* state, THCudaTensor *self_, double mean, double stdv)
{
  THAssert(THCudaTensor_checkGPU(state, 1, self_));
  Generator* gen = THCRandom_getGenerator(state);
  THCudaTensor *self = THCudaTensor_newContiguous(state, self_);
  ptrdiff_t size = THCudaTensor_nElement(state, self);
  float *data = THCudaTensor_data(state, self);
  #ifdef CURAND_PATH
  hipLaunchKernel(HIP_KERNEL_NAME(generate_normal), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state), 
      gen->gen_states, size, data, mean, stdv);
  #else
  generate_normal(state, gen->h_gen_states, size, data, mean, stdv);
  #endif

  THCudaTensor_freeCopyTo(state, self, self_);
};

THC_API void THCudaTensor_logNormal(THCState* state, THCudaTensor *self_, double mean, double stdv)
{
  THAssert(THCudaTensor_checkGPU(state, 1, self_));
  Generator* gen = THCRandom_getGenerator(state);

  THCudaTensor *self = THCudaTensor_newContiguous(state, self_);
  ptrdiff_t size = THCudaTensor_nElement(state, self);
  float *data = THCudaTensor_data(state, self);
  #ifdef CURAND_PATH
  hipLaunchKernel(HIP_KERNEL_NAME(generate_log_normal), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state), 
      gen->gen_states, size, data, mean, stdv);
  #else
    generate_log_normal(state, gen->h_gen_states, size, data, mean, stdv); 
  #endif
  THCudaTensor_freeCopyTo(state, self, self_);
};

THC_API void THCudaTensor_geometric(THCState* state, THCudaTensor *self_, double p)
{
  THAssert(THCudaTensor_checkGPU(state, 1, self_));
  Generator* gen = THCRandom_getGenerator(state);

  THCudaTensor *self = THCudaTensor_newContiguous(state, self_);
  ptrdiff_t size = THCudaTensor_nElement(state, self);
  float *data = THCudaTensor_data(state, self);
  #ifdef CURAND_PATH
  hipLaunchKernel(HIP_KERNEL_NAME(generate_geometric), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state), 
      gen->gen_states, size, data, p);
  #else
  generate_geometric(state, gen->h_gen_states, size, data, p);
  #endif

  THCudaTensor_freeCopyTo(state, self, self_);
};

THC_API void THCudaTensor_exponential(THCState* state, THCudaTensor *self_, double lambda)
{
  THAssert(THCudaTensor_checkGPU(state, 1, self_));
  Generator* gen = THCRandom_getGenerator(state);

  THCudaTensor *self = THCudaTensor_newContiguous(state, self_);
  ptrdiff_t size = THCudaTensor_nElement(state, self);
  float *data = THCudaTensor_data(state, self);
  #ifdef CURAND_PATH
  hipLaunchKernel(HIP_KERNEL_NAME(generate_exponential), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state), 
      gen->gen_states, size, data, lambda);
  #else
  generate_exponential(state, gen->h_gen_states, size, data, lambda);
  #endif

  THCudaTensor_freeCopyTo(state, self, self_);
};

THC_API void THCudaTensor_cauchy(THCState* state, THCudaTensor *self_, double median, double sigma)
{
  THAssert(THCudaTensor_checkGPU(state, 1, self_));
  Generator* gen = THCRandom_getGenerator(state);

  THCudaTensor *self = THCudaTensor_newContiguous(state, self_);
  ptrdiff_t size = THCudaTensor_nElement(state, self);
  float *data = THCudaTensor_data(state, self);
  #ifdef CURAND_PATH
  hipLaunchKernel(HIP_KERNEL_NAME(generate_cauchy), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state), 
      gen->gen_states, size, data, median, sigma);
  #else
  generate_cauchy(state, gen->h_gen_states, size, data, median, sigma);
  #endif

  THCudaTensor_freeCopyTo(state, self, self_);
};

__device__ int binarySearchForMultinomial(float* dist,
                                          int size,
                                          float val) {
  int start = 0;
  int end = size;

  while (end - start > 0) {
    int mid = start + (end - start) / 2;

    float midVal = dist[mid];
    if (midVal < val) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }

  if (start == size) {
    // No probability mass or precision problems; just return the
    // first element
    start = 0;
  }

  return start;
}

// Normalizes the L1 norm of every row to 1; used by multinomial
__global__ void renormRowsL1(hipLaunchParm lp, float* dist, long rows, long cols) {
  HIP_DYNAMIC_SHARED( float, smem)

  for (long row = hipBlockIdx_x; row < rows; row += hipGridDim_x) {
    float sum = 0.0f;
    for (long col = hipThreadIdx_x; col < cols; col += hipBlockDim_x) {
      sum += dist[row * cols + col];
    }

   sum = reduceBlock(reinterpret_cast<float*>(smem),
                      hipBlockDim_x,
                      sum,
    #if defined(THRUST_PATH)
                      thrust::plus<float>(),
    #else
                      bolt::amp::plus<float>(),
    #endif
                      0.0f);
    if (hipThreadIdx_x == 0) {
      smem[0] = sum;
    }
    __syncthreads();

    sum = smem[0];
    if (sum > 0.0f) {
      for (long col = hipThreadIdx_x; col < cols; col += hipBlockDim_x) {
        dist[row * cols + col] /= sum;
      }
    }
  }
}

void THCudaTensor_renormRows(struct THCState* state,
                             THCudaTensor* t) {
  THAssert(THCudaTensor_nDimension(state, t) == 2);
  long rows = THCudaTensor_size(state, t, 0);
  long cols = THCudaTensor_size(state, t, 1);

  hipDeviceProp_t* props = THCState_getCurrentDeviceProperties(state);
  THAssert(props != NULL);

  int numSM = props->multiProcessorCount;
  int maxThreads = props->maxThreadsPerBlock;

  dim3 grid(rows < numSM * 4 ? rows : numSM * 4);
  dim3 block(cols < maxThreads ? cols : maxThreads);

  hipLaunchKernel(HIP_KERNEL_NAME(renormRowsL1), dim3(grid), dim3(block), block.x * sizeof(float), THCState_getCurrentStream(state), THCudaTensor_data(state, t),
                                        rows, cols);
}

__global__ void
sampleMultinomialOnce(hipLaunchParm lp, float* dest,
                      long distributions,
                      int categories,
                      float* dist) {
  HIP_DYNAMIC_SHARED( float, smem)

  for (long curDist = hipBlockIdx_x;
       curDist < distributions; curDist += hipGridDim_x) {
    // Each block handles one distribution
    // First pass, find the total sum of the distribution
    float sum = 0.0f;
    for (int cat = hipThreadIdx_x; cat < categories; cat += hipBlockDim_x) {
      sum += dist[curDist * categories + cat];
    }

    // hipThreadIdx_x == 0 has the sum value from this
    sum = reduceBlock(reinterpret_cast<float*>(smem),
                      hipBlockDim_x,
                      sum,
    #if defined(THRUST_PATH)
                      thrust::plus<float>(),
    #else
                      bolt::amp::plus<float>(),
    #endif
                      0.0f);

    // Broadcast sum and sample value
    if (hipThreadIdx_x == 0) {
      smem[0] = sum;
      smem[1] = dest[curDist];
    }
    __syncthreads();

    sum = smem[0];
    float sample = smem[1];
    __syncthreads();

    if (sum == 0.0f || sample == 0.0f) {
      // Choose the first element
      if (hipThreadIdx_x == 0) {
        dest[curDist] = 1;
      }

      continue;
    }

    int chunks = THCCeilDiv(categories, (int) hipBlockDim_x);
    float prevHighProb = 0.0f;

    for (int chunk = 0; chunk < chunks; ++chunk) {
      // All threads in bounds load a value
      int cat = chunk * hipBlockDim_x + hipThreadIdx_x;

      float val =
        cat < categories ? dist[curDist * categories + cat] / sum : 0.0f;
      smem[hipThreadIdx_x] = val;
      __syncthreads();

      // Perform an inclusive prefix sum of the shared memory contents
      for (int offset = 1; offset < hipBlockDim_x; offset *= 2) {
        float val = 0.0f;

        if (hipThreadIdx_x >= offset) {
          val = smem[hipThreadIdx_x - offset] + smem[hipThreadIdx_x];
        }

        __syncthreads();
        if (hipThreadIdx_x >= offset) {
          smem[hipThreadIdx_x] = val;
        }
        __syncthreads();
      }

      // Each thread will check to see if the sample falls in its
      // bucket
      float curBucket =
        smem[hipThreadIdx_x] + prevHighProb;
      float prevBucket =
        hipThreadIdx_x == 0 ? prevHighProb : smem[hipThreadIdx_x - 1] + prevHighProb;
      bool inBucket =
        (cat < categories) && (sample <= curBucket) && (sample > prevBucket);

      if (inBucket) {
        // We're done; we have the sample
        // Torch indices are 1-based
        // FIXME: broadcast exit flag?
        dest[curDist] = cat + TH_INDEX_BASE;
      }

      // Store the previous scan's high value for future use
      prevHighProb += smem[hipBlockDim_x - 1];

      __syncthreads();
    }
  }
}

#ifdef CURAND_PATH
__global__ void
sampleMultinomialWithReplacement(hipLaunchParm lp, curandStateMtgp32* state,
                                 int totalSamples,
                                 float* dest,
                                 long distributions,
                                 int categories,
                                 float* normDistPrefixSum) {
#else
__global__ void
sampleMultinomialWithReplacement(hipLaunchParm lp, HipRandStateMtgp32* state,
                                 int totalSamples,
                                 float* dest,
                                 long distributions,
                                 int categories,
                                 float* normDistPrefixSum) {

#endif
  // At the moment, each warp computes one sample value in the binary
  // search due to divergence. It seems possible to compute multiple
  // values and limit divergence though later on. However, no matter
  // what, all block threads must participate in the curand_uniform
  // call to update the generator state.

  // The block determines the distribution for which we generate a point
  for (long curDist = hipBlockIdx_x;
       curDist < distributions;
       curDist += hipGridDim_x) {
    for (int sampleBase = 0;
         sampleBase < totalSamples; sampleBase += hipBlockDim_y) {
      // The warp determines the sample
      int sample = sampleBase + hipThreadIdx_y;

      // All threads participate in this
      float r = 0.0f;
      #ifdef CURAND_PATH
      r = curand_uniform(&state[hipBlockIdx_x]);
      #endif 

      if (hipThreadIdx_x == 0 && sample < totalSamples) {
        // Find the bucket that a uniform sample lies in
        int choice = binarySearchForMultinomial(
          normDistPrefixSum + curDist * categories,
          categories,
          r);

        // Torch indices are 1-based
        dest[curDist * totalSamples + sample] = (float) choice + (float)TH_INDEX_BASE;
      }
    }
  }
}

#ifdef CURAND_PATH
__global__ void
sampleMultinomialWithoutReplacement(hipLaunchParm lp, curandStateMtgp32* state,
                                    int totalSamples,
                                    int sample,
                                    float* dest,
                                    long distributions,
                                    int categories,
                                    float* origDist,
                                    float* normDistPrefixSum) {
#else
__global__ void
sampleMultinomialWithoutReplacement(hipLaunchParm lp, HipRandStateMtgp32* state,
                                    int totalSamples,
                                    int sample,
                                    float* dest,
                                    long distributions,
                                    int categories,
                                    float* origDist,
                                    float* normDistPrefixSum) {
#endif
  // At the moment, each warp computes one sample value in the binary
  // search due to divergence. It seems possible to compute multiple
  // values and limit divergence though later on. However, no matter
  // what, all block threads must participate in the curand_uniform
  // call to update the generator state.

  // The block and warp determines the distribution for which we
  // generate a point
  for (long curDistBase = hipBlockIdx_x * hipBlockDim_y;
       curDistBase < distributions;
       curDistBase += hipGridDim_x * hipBlockDim_y) {
    // The warp determines the distribution
    long curDist = curDistBase + hipThreadIdx_y;

    // All threads must participate in this
    float r = 0.0f;
    #ifdef CURAND_PATH
    r = curand_uniform(&state[hipBlockIdx_x]);
    #endif

    if (hipThreadIdx_x == 0 && curDist < distributions) {
      // Find the bucket that a uniform sample lies in
      int choice = binarySearchForMultinomial(
        normDistPrefixSum + curDist * categories,
        categories,
        r);

      // Torch indices are 1-based
      dest[curDist * totalSamples + sample] = (float) choice + (float)TH_INDEX_BASE;

      // Without replacement, so update the original probability so it
      // is not considered a second time
      origDist[curDist * categories + choice] = 0.0f;
    }
  }
}

THC_API void THCudaTensor_multinomial(struct THCState *state,
                                      THCudaTensor *self,
                                      THCudaTensor *prob_dist,
                                      int n_sample,
                                      int with_replacement)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self, prob_dist));
  Generator* gen = THCRandom_getGenerator(state);

  int inputSize = THCudaTensor_nDimension(state, prob_dist);
  THArgCheck(inputSize > 0 && inputSize <= 2, 2,
             "prob_dist must be 1 or 2 dim");

  // Categories are in the innermost dimension
  long numDist =
    inputSize == 1 ? 1 : THCudaTensor_size(state, prob_dist, 0);
  long numCategoriesLong =
    inputSize == 1 ? THCudaTensor_size(state, prob_dist, 0) :
    THCudaTensor_size(state, prob_dist, 1);

  // Since the index tensor is float, numCategories cannot exceed max
  // float integer precision
  THArgCheck(numCategoriesLong <= FLOAT32_MAX_CONSECUTIVE_INT, 2,
             "number of categories cannot exceed 2^24");
  int numCategories = (int) numCategoriesLong;

  THArgCheck(n_sample > 0, 3, "cannot sample <= 0 samples");

  if (!with_replacement) {
    THArgCheck(n_sample <= numCategories, 2,
               "cannot sample n_sample > prob_dist:size(1) samples without "
               "replacement");
  }

  // It is possible that prob_dist is non-contiguous
  THCudaTensor* probDistContig =
    THCudaTensor_newContiguous(state, prob_dist);

  // Restructure data for 2d
  if (inputSize == 1) {
    THCudaTensor_resize2d(state, probDistContig, 1, numCategories);
  }

  THCudaTensor_resize2d(state, self, numDist, n_sample);

  if (n_sample == 1) {
    // Optimized allocation-free implementation

    // To exploit greater parallelism for the sampling, generate the
    // Uniform random samples in a separate kernel launch, into the
    // result memory. The device RNG is thread-limited
    THCudaTensor_uniform(state, self, 0.0, 1.0);

    hipDeviceProp_t* props = THCState_getCurrentDeviceProperties(state);
    THAssert(props != NULL);

    int numSM = props->multiProcessorCount;
    int maxThreads = props->maxThreadsPerBlock;

    dim3 block(numCategories < maxThreads ? numCategories : maxThreads);
    dim3 grid(numDist < numSM * 4 ? numDist : numSM * 4);

    hipLaunchKernel(HIP_KERNEL_NAME(sampleMultinomialOnce), dim3(grid), dim3(block), block.x * sizeof(float), THCState_getCurrentStream(state), 
      THCudaTensor_data(state, self),
      numDist,
      numCategories,
      THCudaTensor_data(state, probDistContig));
  } else {
    // Generic, slow implementation with memory allocations

    // For sampling without replacement, we modify the distribution
    // for subsequent samples in this space
    THCudaTensor* origDist = THCudaTensor_new(state);
    THCudaTensor_resizeAs(state, origDist, probDistContig);
    THCudaTensor_copy(state, origDist, probDistContig);

    THCudaTensor* normDist = THCudaTensor_new(state);
    THCudaTensor_resizeAs(state, normDist, probDistContig);

    THCudaTensor* prefixSum = THCudaTensor_new(state);

    // Renorm along rows
    THCudaTensor_copy(state, normDist, origDist);
    THCudaTensor_renormRows(state, normDist);

    // Prefix sum along rows
    THCudaTensor_cumsum(state, prefixSum, normDist, 1);

    if (with_replacement) {
      // Sample with replacement

      // Binary search is warp divergent (so effectively we're running
      // with just a single thread), but for better utilization,
      // we need each block to have at least 4 warps.
      dim3 block(32, 4);

      // Each warp in a block will generate a sample from one
      // distribution concurrently.
      dim3 grid(numDist < MAX_NUM_BLOCKS ? numDist : MAX_NUM_BLOCKS);

#ifdef CURAND_PATH
      hipLaunchKernel(HIP_KERNEL_NAME(sampleMultinomialWithReplacement), dim3(grid), dim3(block), 0, THCState_getCurrentStream(state), 
          gen->gen_states,
          n_sample,
          THCudaTensor_data(state, self),
          numDist, numCategories,
          THCudaTensor_data(state, prefixSum));
#else
      hipLaunchKernel(HIP_KERNEL_NAME(sampleMultinomialWithReplacement), dim3(grid), dim3(block), 0, THCState_getCurrentStream(state), 
          gen->h_gen_states,
          n_sample,
          THCudaTensor_data(state, self),
          numDist, numCategories,
          THCudaTensor_data(state, prefixSum));
#endif
    } else {
      // Sample without replacement

      // Binary search is warp divergent (so effectively we're running
      // with just a single thread), but for better utilization,
      // we need each block to have at least 4 warps.
      dim3 block(32, 4);

      // Each warp in a block will generate a sample from a different
      // distribution concurrently.
      ptrdiff_t numBlocks = THCCeilDiv(numDist, 4L);
      dim3 grid(numBlocks < MAX_NUM_BLOCKS ? numBlocks : MAX_NUM_BLOCKS);

      for (int sample = 0; sample < n_sample; ++sample) {
        if (sample > 0) {
          // Update probabilities
          // Renorm along rows
          THCudaTensor_copy(state, normDist, origDist);
          THCudaTensor_renormRows(state, normDist);

          // Prefix sum along rows
          THCudaTensor_cumsum(state, prefixSum, normDist, 1);
        }

        // The kernel can only draw one sample before we have to
        // recalculate our distribution
#ifdef CURAND_PATH
        hipLaunchKernel(HIP_KERNEL_NAME(sampleMultinomialWithoutReplacement), dim3(grid), dim3(block), 0, THCState_getCurrentStream(state), 
            gen->gen_states,
            n_sample,
            sample,
            THCudaTensor_data(state, self),
            numDist, numCategories,
            THCudaTensor_data(state, origDist),
            THCudaTensor_data(state, prefixSum));
#else
        hipLaunchKernel(HIP_KERNEL_NAME(sampleMultinomialWithoutReplacement), dim3(grid), dim3(block), 0, THCState_getCurrentStream(state), 
            gen->h_gen_states,
            n_sample,
            sample,
            THCudaTensor_data(state, self),
            numDist, numCategories,
            THCudaTensor_data(state, origDist),
            THCudaTensor_data(state, prefixSum));
#endif
      }
    }

    THCudaTensor_free(state, prefixSum);
    THCudaTensor_free(state, normDist);
    THCudaTensor_free(state, origDist);
  }

  // Revert data restructuring based on input sizes
  if (inputSize == 1) {
    THCudaTensor_resize1d(state, self, n_sample);

    // Unfortunately, if prob_dist is contiguous already,
    // newContiguous is not a private copy, so we have to restructure
    // this too, so as to not affect prob_dist
    THCudaTensor_resize1d(state, probDistContig, numCategories);
  }

  THCudaTensor_free(state, probDistContig);
}

#undef NUM_BLOCKS
