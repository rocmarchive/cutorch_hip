#ifndef THC_TENSORMATH_COMPARE_CUH
#define THC_TENSORMATH_COMPARE_CUH

#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCNumerics.cuh"

template <typename T, typename TOut>
struct TensorLTValueOp {
  TensorLTValueOp() = default;
  TensorLTValueOp(const TensorLTValueOp&) = default;
  TensorLTValueOp(TensorLTValueOp&&) = default;

  __host__ __device__
  explicit
  TensorLTValueOp(T v) : value(v) {}

  __device__ __forceinline__
  void operator()(TOut* out, T* in) const {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::lt(*in, value));
  }

  __host__ __device__
  ~TensorLTValueOp() {}

  T value;
};

template <typename T, typename TOut>
struct TensorGTValueOp {
  TensorGTValueOp() = default;
  TensorGTValueOp(const TensorGTValueOp&) = default;
  TensorGTValueOp(TensorGTValueOp&&) = default;

  __host__ __device__
  explicit
  TensorGTValueOp(T v) : value(v) {}

  __device__ __forceinline__
  void operator()(TOut* out, T* in) const {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::gt(*in, value));
  }

  __host__ __device__
  ~TensorGTValueOp() {}

  T value;
};


template <typename T, typename TOut>
struct TensorLEValueOp {
  TensorLEValueOp() = default;
  TensorLEValueOp(const TensorLEValueOp&) = default;
  TensorLEValueOp(TensorLEValueOp&&) = default;

  __host__ __device__
  explicit
  TensorLEValueOp(T v) : value(v) {}

  __device__ __forceinline__
  void operator()(TOut* out, T* in) const {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::le(*in, value));
  }

  __host__ __device__
  ~TensorLEValueOp() {}

  T value;
};

template <typename T, typename TOut>
struct TensorGEValueOp {
  TensorGEValueOp() = default;
  TensorGEValueOp(const TensorGEValueOp&) = default;
  TensorGEValueOp(TensorGEValueOp&&) = default;

  __host__ __device__
  explicit
  TensorGEValueOp(T v) : value(v) {}

  __device__ __forceinline__
  void operator()(TOut* out, T* in) const {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::ge(*in, value));
  }

  __host__ __device__
  ~TensorGEValueOp() {}

  T value;
};

template <typename T, typename TOut>
struct TensorEQValueOp {
  TensorEQValueOp() = default;
  TensorEQValueOp(const TensorEQValueOp&) = default;
  TensorEQValueOp(TensorEQValueOp&&) = default;

  __host__ __device__
  explicit
  TensorEQValueOp(T v) : value(v) {}

  __device__ __forceinline__
  void operator()(TOut* out, T* in) const {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::eq(*in, value));
  }

  __host__ __device__
  ~TensorEQValueOp() {}

  T value;
};

template <typename T, typename TOut>
struct TensorNEValueOp {
  TensorNEValueOp() = default;
  TensorNEValueOp(const TensorNEValueOp&) = default;
  TensorNEValueOp(TensorNEValueOp&&) = default;

  __host__ __device__
  explicit
  TensorNEValueOp(T v) : value(v) {}

  __device__ __forceinline__
  void operator()(TOut* out, T* in) const {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::ne(*in, value));
  }

  __host__ __device__
  ~TensorNEValueOp() {}

  T value;
};

template<typename TensorType, typename TensorTypeOut, class Op>
void THC_logicalValue(THCState *state,
                      TensorTypeOut *self_,
                      TensorType *src,
                      Op op) {
  THLongStorage* st = TensorUtils<TensorType>::newSizeOf(state, src);
  TensorUtils<TensorTypeOut>::resize(state, self_, st, NULL);
  THLongStorage_free(st);

  if (!THC_pointwiseApply2(state, self_, src, op)) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(hipGetLastError());
}

#endif // THC_TENSORMATH_COMPARE_CUH
