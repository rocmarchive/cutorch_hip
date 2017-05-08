#include "THCHalf.h"
#include "THCThrustAllocator.cuh"
#ifdef THRUST_PATH
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#else
  #include <bolt/amp/iterator/ubiquitous_iterator.h>
  #include <bolt/amp/transform.h>
#endif

struct __half2floatOp {
  __device__ float operator()(half v) const { return __half2float(v); }
};

struct __float2halfOp {
  __device__ half operator()(float v) const { return __float2half(v); }
};

void THCFloat2Half(THCState *state, half *out, float *in, ptrdiff_t len) {
  
#ifdef THRUST_PATH
  THCThrustAllocator thrustAlloc(state);
  thrust::transform(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#else
    thrust::device,
#endif
    in, in + len, out, __float2halfOp());
#else
    bolt::amp::transform(
        bolt::amp::make_ubiquitous_iterator(in),
        bolt::amp::make_ubiquitous_iterator(in + len),
        bolt::amp::make_ubiquitous_iterator(out),
        __float2halfOp{});
#endif
}

void THCHalf2Float(THCState *state, float *out, half *in, ptrdiff_t len) {
#ifdef THRUST_PATH
  THCThrustAllocator thrustAlloc(state);
  thrust::transform(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#else
    thrust::device,
#endif
    in, in + len, out, __half2floatOp());
#else
    bolt::amp::transform(
        bolt::amp::make_ubiquitous_iterator(in),
        bolt::amp::make_ubiquitous_iterator(in + len),
        bolt::amp::make_ubiquitous_iterator(out),
        __half2floatOp{});

#endif
}


THC_EXTERNC int THC_nativeHalfInstructions(THCState *state) {
  hipDeviceProp_t* prop =
    THCState_getCurrentDeviceProperties(state);

  // CC 5.3+
  return (prop->major > 5 ||
          (prop->major == 5 && prop->minor == 3));
}

THC_EXTERNC int THC_fastHalfInstructions(THCState *state) {
  hipDeviceProp_t* prop =
    THCState_getCurrentDeviceProperties(state);

  // Check for CC 6.0 only (corresponds to P100)
  return (prop->major == 6 && prop->minor == 0);
}
