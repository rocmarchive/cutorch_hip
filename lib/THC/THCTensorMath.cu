#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCNumerics.cuh"

#include <cfloat>

template <typename T>
struct TensorFillOp {
  __host__ __device__
  explicit
  TensorFillOp(T v) : val(v) {}

  __device__ __forceinline__
  void operator()(T* v) const { *v = val; }

  __host__ __device__
  ~TensorFillOp() {}

  T val;
};

#include "generic/THCTensorMath.cu"
#include "THCGenerateAllTypes.h"
