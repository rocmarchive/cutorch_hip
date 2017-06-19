#include "THCTensorRandom.h"
#include "THCDeviceUtils.cuh"
#include "THCGeneral.h"
#include "THCTensorCopy.h"
#include "THCTensorMath.h"
#include "THCReduceApplyUtils.cuh"
#include "THCTensorRandom.cuh"

#ifdef CURAND_PATH
  #include <curand.h>
  #include <curand_kernel.h>
  #include <curand_mtgp32_host.h>
  #include <curand_mtgp32dc_p_11213.h>
#else
  #include <hip/hip_hcc.h>
  #include "MTGP/hiprand_mtgp32.h"
#endif

#ifdef THRUST_PATH
    #include <thrust/functional.h>
#else
    #include <bolt/amp/functional.h>
#endif


#define MAX_NUM_BLOCKS 64
#define BLOCK_SIZE 256


Generator* THCRandom_getGenerator(THCState* state);

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

/* Creates a new generator state given the seed. */
void createGeneratorState(THCState* state, Generator* gen, unsigned long long seed)
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
__global__ void set_rngstate_kernel(curandStateMtgp32 *state, mtgp32_kernel_params *kernel)
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
  hipLaunchKernelGGL(
    set_rngstate_kernel,
    dim3(1),
    dim3(MAX_NUM_BLOCKS),
    0,
    THCState_getCurrentStream(state),
    gen->gen_states,
    gen->kernel_params);
#else
  THCudaCheck(hipMemcpy(gen->h_gen_states, THByteTensor_data(rng_state),
                         states_size, hipMemcpyHostToDevice));
#endif
  memcpy(&gen->initial_seed, THByteTensor_data(rng_state) + states_size, seed_size);

}

// CURAND_PATH

#ifdef CURAND_PATH

#define GENERATE_KERNEL1(NAME, T, ARG1, CURAND_T, CURAND_FUNC, TRANSFORM)      \
__global__ void NAME(curandStateMtgp32 *state, int size, T *result, ARG1)      \
{                                                                              \
  int idx = hipBlockIdx_x * BLOCK_SIZE + hipThreadIdx_x;                             \
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;                \
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {      \
    CURAND_T x = CURAND_FUNC(&state[hipBlockIdx_x]);                              \
    if (i < size) {                                                            \
      T y = TRANSFORM;                                                         \
      result[i] = y;                                                           \
    }                                                                          \
  }                                                                            \
}

#define GENERATE_KERNEL2(NAME, T, ARG1, ARG2, CURAND_T, CURAND_FUNC, TRANSFORM)      \
__global__ void NAME(curandStateMtgp32 *state, int size, T *result, ARG1, ARG2)      \
{                                                                                    \
  int idx = hipBlockIdx_x * BLOCK_SIZE + hipThreadIdx_x;                                   \
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;                      \
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {            \
    CURAND_T x = CURAND_FUNC(&state[hipBlockIdx_x]);                                    \
    if (i < size) {                                                                  \
      T y = TRANSFORM;                                                               \
      result[i] = y;                                                                 \
    }                                                                                \
  }                                                                                  \
}
#else

#define GENERATE_KERNEL1(NAME, T, ARG1, HIPRAND_T, HIPRAND_FUNC, TRANSFORM)      \
void NAME(THCState* state, HipRandStateMtgp32 *rngstate, int size, T *result, ARG1)  \
{ \
  hipStream_t currentStream = THCState_getCurrentStream(state); \
  hc::accelerator_view* current_accl_view; \
  hipHccGetAcceleratorView(currentStream, &current_accl_view); \
  HIPRAND_FUNC##_kernel(*current_accl_view, rngstate, result, size, TRANSFORM); \
}

#define GENERATE_KERNEL2(NAME, T, ARG1, ARG2, HIPRAND_T, HIPRAND_FUNC, TRANSFORM)      \
void NAME(THCState* state, HipRandStateMtgp32 *rngstate, int size, T *result, ARG1, ARG2)  \
{                                                                                    \
  hipStream_t currentStream = THCState_getCurrentStream(state); \
  hc::accelerator_view* current_accl_view; \
  hipHccGetAcceleratorView(currentStream, &current_accl_view); \
  HIPRAND_FUNC##_kernel(*current_accl_view, rngstate, result, size, TRANSFORM);                                                 \
}


#endif

template<typename T, typename U>
struct is_same { static const bool value = false; };

template<typename T>
struct is_same<T, T> { static const bool value = true; };

#ifdef CURAND_PATH
template<typename real, typename prob_type>
__global__ void generate_bernoulli_tensor(curandStateMtgp32 *state, int size,
        real *result, prob_type *probs)
{
  int idx = hipBlockIdx_x * BLOCK_SIZE + hipThreadIdx_x;
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {
    if (is_same<prob_type, double>::value) {
      double x = curand_uniform_double(&state[hipBlockIdx_x]);
      if (i < size)
        result[i] = ScalarConvert<bool, real>::to(x <= probs[i]);
    } else {
      float x = curand_uniform(&state[hipBlockIdx_x]);
      if (i < size)
        result[i] = ScalarConvert<bool, real>::to(x <= probs[i]);
    }
  }
}

GENERATE_KERNEL2(generate_uniform, float, double a, double b, float, curand_uniform, x * (b-a) + a)
GENERATE_KERNEL2(generate_uniform, double, double a, double b, double, curand_uniform_double, x * (b-a) + a)

GENERATE_KERNEL2(generate_normal, float, double mean, double stdv, float, curand_normal, (x * stdv) + mean)
GENERATE_KERNEL2(generate_normal, double, double mean, double stdv, double, curand_normal_double, (x * stdv) + mean)

GENERATE_KERNEL1(generate_exponential, float, double lambda, float, curand_uniform, (float)(-1. / lambda * log(1-x)))
GENERATE_KERNEL1(generate_exponential, double, double lambda, double, curand_uniform_double, (double)(-1. / lambda * log(1-x)))

GENERATE_KERNEL2(generate_cauchy, float, double median, double sigma, float, curand_uniform, (float)(median + sigma * tan(M_PI*(x-0.5))))
GENERATE_KERNEL2(generate_cauchy, double, double median, double sigma, double, curand_uniform_double, (double)(median + sigma * tan(M_PI*(x-0.5))))

#ifdef CUDA_HALF_TENSOR
GENERATE_KERNEL2(generate_uniform, half, double a, double b, float, curand_uniform, (ScalarConvert<float, half>::to(x * (b-a) + a)))
GENERATE_KERNEL2(generate_normal, half, double mean, double stdv, float, curand_normal, (ScalarConvert<float, half>::to((x * stdv) + mean)))
GENERATE_KERNEL1(generate_exponential, half, double lambda, float, curand_uniform, (ScalarConvert<float, half>::to((float)(-1. / lambda * log(1-x)))))
GENERATE_KERNEL2(generate_cauchy, half, double median, double sigma, float, curand_uniform, (ScalarConvert<float, half>::to((float)(median + sigma * tan(M_PI*(x-0.5))))))
#endif // CUDA_HALF_TENSOR

#else

template<typename real, typename prob_type>
void generate_bernoulli_tensor(hc::accelerator_view accl_view, HipRandStateMtgp32 *s, int size,
        real *&result, prob_type *probs)
{
  hc::accelerator accl = accl_view.get_accelerator();
  int rounded_size = DIVUP(size, BLOCK_SIZE) * BLOCK_SIZE;
  int blocks = std::min((int)DIVUP(size, BLOCK_SIZE), MAX_NUM_BLOCKS);
  hc::extent<1> ext(blocks*BLOCK_SIZE);
  hc::tiled_extent<1> t_ext = ext.tile(BLOCK_SIZE);
  const uint32_t* av_param_tbl = (s->param_tbl);
  const uint32_t* av_temper_tbl = (s->temper_tbl);
  const uint32_t* av_sh1_tbl = (s->sh1_tbl);
  const uint32_t* av_sh2_tbl = (s->sh2_tbl);
  const uint32_t* av_offset = (s->offset);
  const uint32_t* av_index = (s->index);
  const uint32_t* av_pos_tbl = (s->pos_tbl);
  const uint32_t* av_mask = (s->mask);
  const uint32_t* av_d_status = (s->d_status);
  hc::parallel_for_each(
      accl_view, t_ext, [=] (const hc::tiled_index<1>& tidx) [[hc]] {
    int threadId = tidx.global[0];
    int groupId = tidx.tile[0];
    if (groupId >= USER_GROUP_NUM)
      return;
    for (int i = threadId; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {
      float x = hiprand_uniform(
          av_param_tbl,
          av_temper_tbl,
          av_sh1_tbl,
          av_sh2_tbl,
          av_offset,
          av_index,
          av_pos_tbl,
          av_mask,
          av_d_status,
          tidx);
      if (i < size) {
        result[i] = ScalarConvert<bool, real>::to(x <= probs[i]);
      }
    }
  }).wait();
}



// Adding All HC based constructors

class user_uniform_functor {
  double _a;
  double _b;
public:
  __host__ __device__
  user_uniform_functor(double a, double b) : _a(a), _b(b) {}

  __host__ __device__
  double operator()(float x) const { return x * (_b - _a) + _a; }
};


class user_bernoulli_functor {
  double _p;
public:
  __host__ __device__
  explicit
  user_bernoulli_functor(double p) : _p(p) {}

  __host__ __device__
  double operator()(float x) const { return static_cast<double>(x) <= _p; }
};


class user_normal_functor {
  double _stdv;
  double _mean;
public:
  __host__ __device__
  user_normal_functor(double stdv, double mean) : _stdv(stdv), _mean(mean) {}

  __host__ __device__
  double operator()(float x) const { return (x * _stdv) + _mean; }
};

class user_geometric_functor {
  double _p;
public:
  __host__ __device__
  explicit
  user_geometric_functor(double p) : _p(p) {}

  __device__
  double operator()(float x) const
  {
      return ceilf(logf(x) / log(1-_p));
  }
};

class user_exponential_functor {
  double _lambda;
public:
  __host__ __device__
  explicit
  user_exponential_functor(double lambda) : _lambda(lambda) {}

  __device__
  double operator()(float x) const
  {
    return (double)(-1. / _lambda * log((double)(1 - x)));
  }
};

class user_cauchy_functor {
  double _median;
  double _sigma;
public:
  __host__ __device__
  user_cauchy_functor(double median, double sigma)
      : _median(median), _sigma(sigma)
  {}

  __device__
  double operator()(float x) const
  {
    return (double)(_median + _sigma * tan((double)M_PI * (x - 0.5)));
  }
};


GENERATE_KERNEL2(generate_uniform, float, double a, double b, float, user_uniform, (user_uniform_functor(a, b)))
GENERATE_KERNEL2(generate_uniform, double, double a, double b, double, user_uniform, (user_uniform_functor(a, b)))
GENERATE_KERNEL2(generate_uniform, half, double a, double b, float, user_uniform, (user_uniform_functor(a, b)))

GENERATE_KERNEL2(generate_normal, float, double mean, double stdv, float, user_normal, user_normal_functor(stdv, mean))
GENERATE_KERNEL2(generate_normal, double, double mean, double stdv, double, user_normal, user_normal_functor(stdv, mean))
GENERATE_KERNEL2(generate_normal, half, double mean, double stdv, float, user_normal, user_normal_functor(stdv, mean))

GENERATE_KERNEL1(generate_exponential, float, double lambda, float, user_uniform, user_exponential_functor(lambda))
GENERATE_KERNEL1(generate_exponential, double, double lambda, double, user_uniform, user_exponential_functor(lambda))
GENERATE_KERNEL1(generate_exponential, half, double lambda, float, user_uniform, user_exponential_functor(lambda))

GENERATE_KERNEL2(generate_cauchy, float, double median, double sigma, float, user_uniform, user_cauchy_functor(median, sigma))
GENERATE_KERNEL2(generate_cauchy, double, double median, double sigma, double, user_uniform, user_cauchy_functor(median, sigma))
GENERATE_KERNEL2(generate_cauchy, half, double median, double sigma, float, user_uniform, user_cauchy_functor(median, sigma))



#endif



#include "generic/THCTensorRandom.cu"
#include "THCGenerateAllTypes.h"

#undef GENERATE_KERNEL1
#undef GENERATE_KERNEL2
