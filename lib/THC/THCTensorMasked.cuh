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
  TensorMaskedFillOp() = default;
  TensorMaskedFillOp(const TensorMaskedFillOp&) = default;
  TensorMaskedFillOp(TensorMaskedFillOp&&) = default;

  __host__ __device__
  explicit
  TensorMaskedFillOp(T v) : value{v} {}

  __device__ inline void operator()(T* t, MaskT* mask) const {
    if (*mask) {
      *t = value;
    }
  }

  __host__ __device__
  ~TensorMaskedFillOp() {}

  T value;
};

template <typename T, typename MaskT, typename MaskPrefixSumT>
struct TensorMaskedCopyOp {
  TensorMaskedCopyOp() = default;
  TensorMaskedCopyOp(const TensorMaskedCopyOp&) = default;
  TensorMaskedCopyOp(TensorMaskedCopyOp&&) = default;

  __host__ __device__
  explicit
  TensorMaskedCopyOp(T* s) : in{s} {}

  __device__
  inline
  void operator()(T* out, MaskT* mask, MaskPrefixSumT* maskPrefixSum) const {
    if (*mask) {
      *out = in[*maskPrefixSum];
    }
  }

  __host__ __device__
  ~TensorMaskedCopyOp() {}
  // Where we are copying from
  T* in;
};

template <typename T, typename MaskT, typename MaskPrefixSumT>
struct TensorMaskedSelectOp {
  TensorMaskedSelectOp() = default;
  TensorMaskedSelectOp(const TensorMaskedSelectOp&) = default;
  TensorMaskedSelectOp(TensorMaskedSelectOp&&) = default;

  __host__ __device__
  explicit
  TensorMaskedSelectOp(T* t) : out{t} {}

  __device__
  inline
  void operator()(MaskT* mask, MaskPrefixSumT* maskPrefixSum, T* in) const {
    if (*mask) {
      out[*maskPrefixSum] = *in;
    }
  }

  __host__ __device__
  ~TensorMaskedSelectOp() {}

  T* out;
};

#endif // THC_TENSOR_MASKED_CUH
