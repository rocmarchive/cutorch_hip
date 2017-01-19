#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCBlas.h"
#include "THCTensorCopy.h"
#include "THCTensorRandom.h"
#include "THCApply.cuh"
#include "THCReduce.cuh"
#include "THCTensorMathReduce.cuh"
#include "THCTensorMathPointwise.cuh"

#include <bolt/amp/functional.h>
#include <bolt/amp/inner_product.h>


struct TensorTPowOp {
  __host__ __device__
  explicit
  TensorTPowOp(float v) : val{v} {}

  __device__ __forceinline__ void operator()(float* out, float* in) const {
    *out = powf(val, *in);
  }

  __device__ __forceinline__ void operator()(float* v) const {
    *v = powf(val, *v);
  }

  __host__ __device__
  ~TensorTPowOp() {}

  float val;
};

inline
void THCudaTensor_tpow(THCState *state, THCudaTensor *self_, float value, THCudaTensor *src)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src));
  if (self_ == src) {
    if (!THC_pointwiseApply1(state, self_, TensorTPowOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src);

    if (!THC_pointwiseApply2(state, self_, src, TensorTPowOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(hipGetLastError());
}

struct TensorATan2Op {
  __device__ __forceinline__
  void operator()(float* out, float* a, float* b) const {
    *out = atan2f(*a, *b);
  }
};

void THCudaTensor_atan2(THCState *state, THCudaTensor *self_, THCudaTensor *tx, THCudaTensor *ty)
{
  THAssert(THCudaTensor_checkGPU(state, 3, self_, tx, ty));
  THArgCheck(THCudaTensor_nElement(state, tx) ==
             THCudaTensor_nElement(state, ty), 3, "sizes do not match");
  THCudaTensor_resizeAs(state, self_, tx);

  if (!THC_pointwiseApply3(state, self_, tx, ty, TensorATan2Op())) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(hipGetLastError());
}

float THCudaTensor_dist(THCState *state, THCudaTensor *self, THCudaTensor *src, float value)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self, src));
  self = THCudaTensor_newContiguous(state, self);
  ptrdiff_t size = THCudaTensor_nElement(state, self);
  src = THCudaTensor_newContiguous(state, src);

  auto self_data = THCudaTensor_data(state, self);
  auto src_data = THCudaTensor_data(state, src);

  float result = 0.0; bolt::amp::inner_product( // TODO: add localised version.
#if CUDA_VERSION >= 7000
//    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    self_data, self_data+size, src_data, (float) 0,
    bolt::amp::plus<float>(), TensorDistOp<float>(value));

  THCudaTensor_free(state, src);
  THCudaTensor_free(state, self);

  return pow(result, (float)1.0/value);
}

void THCudaTensor_rand(THCState *state, THCudaTensor *r_, THLongStorage *size)
{
  THAssert(THCudaTensor_checkGPU(state, 1, r_));
  THCudaTensor_resize(state, r_, size, NULL);
  THCudaTensor_uniform(state, r_, 0, 1);
}

void THCudaTensor_randn(THCState *state, THCudaTensor *r_, THLongStorage *size)
{
  THAssert(THCudaTensor_checkGPU(state, 1, r_));
  THCudaTensor_resize(state, r_, size, NULL);
  THCudaTensor_normal(state, r_, 0, 1);
}
