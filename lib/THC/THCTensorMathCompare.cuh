#ifndef THC_TENSORMATH_COMPARE_CUH
#define THC_TENSORMATH_COMPARE_CUH

#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCNumerics.cuh"

template <typename T, typename TOut>
struct TensorLTValueOp {
  __host__ __device__ TensorLTValueOp(T v) : value(v) {}
  __device__ __forceinline__ void operator()(TOut* out, T* in) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::lt(*in, value));
  }
  __host__ __device__ ~TensorLTValueOp() {}
  const T value;
};

template <typename T, typename TOut>
struct TensorGTValueOp {
  __host__ __device__ TensorGTValueOp(T v) : value(v) {}
  __device__ __forceinline__ void operator()(TOut* out, T* in) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::gt(*in, value));
  }

  __host__ __device__ ~TensorGTValueOp() {}
  const T value;
};


template <typename T, typename TOut>
struct TensorLEValueOp {
  __host__ __device__ TensorLEValueOp(T v) : value(v) {}
  __device__ __forceinline__ void operator()(TOut* out, T* in) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::le(*in, value));
  }

  __host__ __device__ ~TensorLEValueOp() {}
  const T value;
};

template <typename T, typename TOut>
struct TensorGEValueOp {
  __host__ __device__ TensorGEValueOp(T v) : value(v) {}
  __device__ __forceinline__ void operator()(TOut* out, T* in) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::ge(*in, value));
  }

  __host__ __device__ ~TensorGEValueOp() {}
  const T value;
};

template <typename T, typename TOut>
struct TensorEQValueOp {
  __host__ __device__ TensorEQValueOp(T v) : value(v) {}
  __device__ __forceinline__ void operator()(TOut* out, T* in) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::eq(*in, value));
  }

  __host__ __device__ ~TensorEQValueOp() {}
  const T value;
};

template <typename T, typename TOut>
struct TensorNEValueOp {
  __host__ __device__ TensorNEValueOp(T v) : value(v) {}
  __device__ __forceinline__ void operator()(TOut* out, T* in) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::ne(*in, value));
  }

  __host__ __device__ ~TensorNEValueOp() {}
  const T value;
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
