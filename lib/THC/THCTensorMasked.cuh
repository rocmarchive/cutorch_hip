#ifndef THC_TENSOR_MASKED_CUH
#define THC_TENSOR_MASKED_CUH
#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCReduce.cuh"

#include <bolt/amp/scan.h>

template <typename T, typename MaskT>
struct TensorMaskedFillOp {
  __host__ __device__
  explicit
  TensorMaskedFillOp(T v) : value{v} {}
  __device__ inline void operator()(T* t, MaskT* mask) {
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

  __device__ inline void operator()(T* out,
                                    MaskT* mask,
                                    MaskPrefixSumT* maskPrefixSum) {
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
  __device__ inline void operator()(MaskT* mask,
                                    MaskPrefixSumT* maskPrefixSum,
                                    T* in) {
    if (*mask) {
      out[*maskPrefixSum] = *in;
    }
  }

  T* out;
};

#endif // THC_TENSOR_MASKED_CUH
