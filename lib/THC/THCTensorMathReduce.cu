#include "THCTensorMathReduce.cuh"

struct Identity_fn { // TODO: this is temporary and should be removed.
    template<typename T>
    __host__ __device__
    constexpr
    T operator()(const T& x) const { return x; }
};

THC_API int
THCudaByteTensor_logicalall(THCState *state, THCudaByteTensor *self) {
  THAssert(THCudaByteTensor_checkGPU(state, 1, self));
  unsigned char result;
  if (!THC_reduceAll(state, self,
                     Identity_fn{},
                     LogicalAll(),
                     LogicalAll(),
                     (unsigned char) 1, &result, 0)) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  return (int) result;
}

THC_API int
THCudaByteTensor_logicalany(THCState *state, THCudaByteTensor *self) {
  THAssert(THCudaByteTensor_checkGPU(state, 1, self));
  unsigned char result;
  if (!THC_reduceAll(state, self,
                     Identity_fn{},
                     LogicalAny(),
                     LogicalAny(),
                     (unsigned char) 0, &result, 0)) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  return (int) result;
}
