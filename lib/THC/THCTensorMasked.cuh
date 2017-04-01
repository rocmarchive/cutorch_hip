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
    #include <bolt/amp/scan.h>
#endif

template <typename T, typename MaskT>
struct TensorMaskedFillOp {
  __host__ __device__
  explicit
  TensorMaskedFillOp(T v) : value{v} {}

  __device__
  void operator()(T* t, const MaskT* mask) const
  {
    if (*mask) {
      *t = value;
    }
  }

  T value;
};

template <typename T, typename MaskT, typename MaskPrefixSumT>
struct TensorMaskedCopyOp {
  __host__ __device__
  explicit
  TensorMaskedCopyOp(T* s) : in{s} {}

  __device__
  inline
  void operator()(
      T* out, const MaskT* mask, const MaskPrefixSumT* maskPrefixSum) const
  {
    if (*mask) {
      *out = in[*maskPrefixSum];
    }
  }

  // Where we are copying from
  T* in;
};

template <typename T, typename MaskT, typename MaskPrefixSumT>
struct TensorMaskedSelectOp {
  __host__ __device__
  explicit
  TensorMaskedSelectOp(T* t) : out{t} {}

  __device__
  inline
  void operator()(
      const MaskT* mask, const MaskPrefixSumT* maskPrefixSum, const T* in) const
  {
    if (*mask) {
      out[*maskPrefixSum] = *in;
    }
  }

  T* out;
};

#endif // THC_TENSOR_MASKED_CUH
