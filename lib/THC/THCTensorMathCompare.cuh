#ifndef THC_TENSORMATH_COMPARE_CUH
#define THC_TENSORMATH_COMPARE_CUH

#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCNumerics.cuh"

template <typename T, typename TOut>
struct TensorLTValueOp {
  __host__ __device__
  explicit
  TensorLTValueOp(T v) : value(v) {}

  __device__ __forceinline__
  void operator()(TOut* out, T* in) const {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::lt(*in, value));
  }

  __host__ __device__
  ~TensorLTValueOp() = default;

  T value;
};

template <typename T, typename TOut>
struct TensorGTValueOp {
  __host__ __device__
  explicit
  TensorGTValueOp(T v) : value(v) {}

  __device__ __forceinline__
  void operator()(TOut* out, T* in) const {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::gt(*in, value));
  }

  __host__ __device__
  ~TensorGTValueOp() = default;

  T value;
};


template <typename T, typename TOut>
struct TensorLEValueOp {
  __host__ __device__
  explicit
  TensorLEValueOp(T v) : value(v) {}

  __device__ __forceinline__
  void operator()(TOut* out, T* in) const {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::le(*in, value));
  }

  __host__ __device__
  ~TensorLEValueOp() = default;

  T value;
};

template <typename T, typename TOut>
struct TensorGEValueOp {
  __host__ __device__
  explicit
  TensorGEValueOp(T v) : value(v) {}

  __device__ __forceinline__
  void operator()(TOut* out, T* in) const {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::ge(*in, value));
  }

  __host__ __device__
  ~TensorGEValueOp() = default;

  T value;
};

template <typename T, typename TOut>
struct TensorEQValueOp {
  __host__ __device__
  explicit
  TensorEQValueOp(T v) : value(v) {}

  __device__ __forceinline__
  void operator()(TOut* out, T* in) const {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::eq(*in, value));
  }

  __host__ __device__
  ~TensorEQValueOp() = default;

  T value;
};

template <typename T, typename TOut>
struct TensorNEValueOp {
  __host__ __device__
  explicit
  TensorNEValueOp(T v) : value(v) {}

  __device__ __forceinline__
  void operator()(TOut* out, T* in) const {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::ne(*in, value));
  }

  __host__ __device__
  ~TensorNEValueOp() = default;

  T value;
};

template<typename TensorType, typename TensorTypeOut, class Op>
void THC_logicalValue(
    THCState *state,
    TensorTypeOut *self_,
    TensorType *src,
    Op op)
{
  THLongStorage* st = TensorUtils<TensorType>::newSizeOf(state, src);
  TensorUtils<TensorTypeOut>::resize(state, self_, st, NULL);
  THLongStorage_free(st);

  if (!THC_pointwiseApply2(state, self_, src, op)) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(hipGetLastError());
}

#endif // THC_TENSORMATH_COMPARE_CUH
