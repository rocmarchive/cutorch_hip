#ifndef THC_HALF_CONVERSION_INC
#define THC_HALF_CONVERSION_INC

#include "THCGeneral.h"

#include <hip/hip_runtime_api.h>
#include <hip/hip_fp16.h>

#include <GGL/grid_launch.hpp>

#define CUDA_HAS_FP16 1
/* We compile with CudaHalfTensor support if we have this: */
#if CUDA_VERSION >= 7050 || defined(CUDA_HAS_FP16)
    #define CUDA_HALF_TENSOR 1
    using half = __half;
#endif

#ifdef CUDA_HALF_TENSOR

#include <stdint.h>

THC_EXTERNC
void THCFloat2Half(THCState *state, half *out, float *in, ptrdiff_t len);
THC_EXTERNC
void THCHalf2Float(THCState *state, float *out, half *in, ptrdiff_t len);
THC_API
half THC_float2half(float a);
THC_API
float THC_half2float(half a);

/* Check for native fp16 support on the current device (CC 5.3+) */
THC_API
int THC_nativeHalfInstructions(THCState *state);

/* Check for performant native fp16 support on the current device */
THC_API
int THC_fastHalfInstructions(THCState *state);

#endif /* CUDA_HALF_TENSOR */

#endif
