#ifndef THC_TENSOR_MASKED_CUH
#define THC_TENSOR_MASKED_CUH
#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCReduce.cuh"

#ifdef THRUST_PATH
  #include <thrust/device_ptr.h>
  #include <thrust/scan.h>
#if CUDA_VERSION >= 7000
  #include <thrust/system/cuda/execution_policy.h>
#endif
#else
  #include "bolt/amp/scan.h"
#endif

template <typename T, typename MaskT>
struct TensorMaskedFillOp {
  __host__ __device__ TensorMaskedFillOp(T v) : value(v) {}
  __host__ __device__ TensorMaskedFillOp(const TensorMaskedFillOp &maskedFillOp) : value(maskedFillOp.value) {};
  __device__ inline void operator()(T* t, MaskT* mask) {
    if (*mask) {
      *t = value;
    }
  }
  __host__ __device__ ~TensorMaskedFillOp() {}
  T value;
};

template <typename T, typename MaskT, typename MaskPrefixSumT>
struct TensorMaskedCopyOp {
  __host__ __device__ TensorMaskedCopyOp(T* s) : in(s) {}

  __host__ __device__ inline void operator()(T* out,
                                    MaskT* mask,
                                    MaskPrefixSumT* maskPrefixSum) {
    if (*mask) {
      out[0] = in[*maskPrefixSum];
    }
  }
  __host__ __device__ ~TensorMaskedCopyOp() {}
  // Where we are copying from
  T* in;
};

template <typename T, typename MaskT, typename MaskPrefixSumT>
struct TensorMaskedSelectOp {
  __host__ __device__ TensorMaskedSelectOp(T* t) : out(t) {}
  __host__ __device__ inline void operator()(MaskT* mask,
                                    MaskPrefixSumT* maskPrefixSum,
                                    T* in) {
    if (*mask) {
      //out[*maskPrefixSum] = *in;
    }
  }
  __host__ __device__ ~TensorMaskedSelectOp() {}
  T* out;
};

#endif // THC_TENSOR_MASKED_CUH
