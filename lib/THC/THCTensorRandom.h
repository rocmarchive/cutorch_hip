#ifndef TH_CUDA_TENSOR_RANDOM_INC
#define TH_CUDA_TENSOR_RANDOM_INC

#include "THCTensor.h"

#include "generic/THCTensorRandom.h"
#include "THCGenerateAllTypes.h"

typedef struct _Generator {
#ifdef CURAND_PATH
  struct curandStateMtgp32* gen_states;
  struct mtgp32_kernel_params *kernel_params;
#else
  struct HipRandStateMtgp32* h_gen_states;
#endif
  int initf;
  unsigned long initial_seed;
} Generator;

typedef struct THCRNGState {
  /* One generator per GPU */
  Generator* gen;
  int num_devices;
} THCRNGState;

struct THCState;

THC_API void THCRandom_init(struct THCState *state, int num_devices, int current_device);
THC_API void THCRandom_shutdown(struct THCState *state);
THC_API unsigned long long THCRandom_seed(struct THCState *state);
THC_API unsigned long long THCRandom_seedAll(struct THCState *state);
THC_API void THCRandom_manualSeed(struct THCState *state, unsigned long long the_seed_);
THC_API void THCRandom_manualSeedAll(struct THCState *state, unsigned long long the_seed_);
THC_API unsigned long long THCRandom_initialSeed(struct THCState *state);
THC_API void THCRandom_getRNGState(struct THCState *state, THByteTensor *rng_state);
THC_API void THCRandom_setRNGState(struct THCState *state, THByteTensor *rng_state);

#ifdef CURAND_PATH
THC_API struct curandStateMtgp32* THCRandom_generatorStates(struct THCState* state);
#else
THC_API struct HipRandStateMtgp32* THCRandom_generatorStates(struct THCState* state);
#endif
#endif
