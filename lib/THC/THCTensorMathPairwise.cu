#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCHalf.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCNumerics.cuh"

#include <hip/hip_runtime.h>

template <typename T>
struct TensorAddConstantOp {
  TensorAddConstantOp() = default;
  TensorAddConstantOp(const TensorAddConstantOp&) = default;
  TensorAddConstantOp(TensorAddConstantOp&&) = default;

  __host__ __device__
  explicit
  TensorAddConstantOp(T v) : val(v) {}

  __device__ __forceinline__
  void operator()(T* out, T* in) const { *out = *in + val; }

  __device__ __forceinline__
  void operator()(T* v) const { *v += val; }

  __host__ __device__
  ~TensorAddConstantOp() {}

  T val;
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorAddConstantOp<half> {
  TensorAddConstantOp() = default;
  TensorAddConstantOp(const TensorAddConstantOp&) = default;
  TensorAddConstantOp(TensorAddConstantOp&&) = default;

#ifdef CUDA_HALF_INSTRUCTIONS
  __host__ __device__
  explicit
  TensorAddConstantOp(half v) : val(v) {}
#else
  TensorAddConstantOp(half v) : fval(THC_half2float(v)) {}
#endif

  __device__ __forceinline__
  void operator()(half* out, half* in) const {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hadd(*in, val);
#else
    float fin = __half2float(*in);
    float fout = fin + fval;
    *out = __float2half(fout);
#endif
  }

  __device__ __forceinline__
  void operator()(half* v) const {
#ifdef CUDA_HALF_INSTRUCTIONS
    *v = __hadd(*v, val);
#else
    float fv = __half2float(*v);
    fv += fval;
    *v = __float2half(fv);
#endif
  }

  __host__ __device__
  ~TensorAddConstantOp() {}
#ifdef CUDA_HALF_INSTRUCTIONS
  half val;
#else
  float fval;
#endif
};
#endif // CUDA_HALF_TENSOR


template <typename T>
struct TensorSubConstantOp {
  TensorSubConstantOp() = default;
  TensorSubConstantOp(const TensorSubConstantOp&) = default;
  TensorSubConstantOp(TensorSubConstantOp&&) = default;

  __host__ __device__
  explicit
  TensorSubConstantOp(T v) : val(v) {}

  __device__ __forceinline__
  void operator()(T* out, T* in) const { *out = *in - val; }

  __device__ __forceinline__
  void operator()(T* v) const { *v -= val; }

  __host__ __device__
  ~TensorSubConstantOp() {}

  T val;
};


#ifdef CUDA_HALF_TENSOR
template <>
struct TensorSubConstantOp<half> {
  TensorSubConstantOp() = default;
  TensorSubConstantOp(const TensorSubConstantOp&) = default;
  TensorSubConstantOp(TensorSubConstantOp&&) = default;

  explicit
#ifdef CUDA_HALF_INSTRUCTIONS
  TensorSubConstantOp(half v): val(__float2half(-(__half2float(v)))) {}
#else
  TensorSubConstantOp(half v): fval(-(THC_half2float(v))) {}
#endif

  __device__ __forceinline__
  void operator()(half* out, half* in) const {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hadd(*in, val);
#else
    float fin = __half2float(*in);
    float fout = fin + fval;
    *out = __float2half(fout);
#endif
  }

  __device__ __forceinline__
  void operator()(half* v) const {
#ifdef CUDA_HALF_INSTRUCTIONS
    *v = __hadd(*v, val);
#else
    float fv = __half2float(*v);
    fv += fval;
    *v = __float2half(fv);
#endif
  }

  __host__ __device__
  ~TensorSubConstantOp() {}

#ifdef CUDA_HALF_INSTRUCTIONS
  half val;
#else
  float fval;
#endif
};
#endif // CUDA_HALF_TENSOR


template <typename T>
struct TensorMulConstantOp {
  TensorMulConstantOp() = default;
  TensorMulConstantOp(const TensorMulConstantOp&) = default;
  TensorMulConstantOp(TensorMulConstantOp&&) = default;

  __host__ __device__
  explicit
  TensorMulConstantOp(T v) : val(v) {}

  __device__ __forceinline__
  void operator()(T* out, T* in) const { *out = *in * val; }

  __device__ __forceinline__
  void operator()(T* v) const { *v *= val; }

  __host__ __device__
  ~TensorMulConstantOp() {}

  T val;
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorMulConstantOp<half> {
  TensorMulConstantOp() = default;
  TensorMulConstantOp(const TensorMulConstantOp&) = default;
  TensorMulConstantOp(TensorMulConstantOp&&) = default;

  explicit
#ifdef CUDA_HALF_INSTRUCTIONS
  TensorMulConstantOp(half v) : val(v) {}
#else
  TensorMulConstantOp(half v) : fval(THC_half2float(v)) {}
#endif

  __device__ __forceinline__
  void operator()(half* out, half* in) const {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hmul(*in, val);
#else
    float fin = __half2float(*in);
    float fout = fin * fval;
    *out = __float2half(fout);
#endif
  }

  __device__ __forceinline__
  void operator()(half* v) const {
#ifdef CUDA_HALF_INSTRUCTIONS
    *v = __hmul(*v, val);
#else
    float fv = __half2float(*v);
    fv *= fval;
    *v = __float2half(fv);
#endif
  }

  __host__ __device__
  ~TensorMulConstantOp() {}

#ifdef CUDA_HALF_INSTRUCTIONS
  half val;
#else
  float fval;
#endif
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorDivConstantOp {
  TensorDivConstantOp() = default;
  TensorDivConstantOp(const TensorDivConstantOp&) = default;
  TensorDivConstantOp(TensorDivConstantOp&&) = default;

  __host__ __device__
  explicit
  TensorDivConstantOp(T v) : val(v) {}

  __device__ __forceinline__
  void operator()(T* out, T* in) const { *out = *in / val; }

  __device__ __forceinline__
  void operator()(T* v) const { *v /= val; }

  __host__ __device__
  ~TensorDivConstantOp() {}

  T val;
};

template <>
struct TensorDivConstantOp<float> {
  TensorDivConstantOp() = default;
  TensorDivConstantOp(const TensorDivConstantOp&) = default;
  TensorDivConstantOp(TensorDivConstantOp&&) = default;

  __host__ __device__
  explicit
  TensorDivConstantOp(float v) : val(1.f / v) {}

  __device__ __forceinline__
  void operator()(float* out, float* in) const { *out = *in * val; }

  __device__ __forceinline__
  void operator()(float* v) const { *v *= val; }

  __host__ __device__
  ~TensorDivConstantOp() {}

  float val;
};

template <>
struct TensorDivConstantOp<double> {
  TensorDivConstantOp() = default;
  TensorDivConstantOp(const TensorDivConstantOp&) = default;
  TensorDivConstantOp(TensorDivConstantOp&&) = default;

  __host__ __device__
  explicit
  TensorDivConstantOp(double v) : val(1. / v) {}

  __device__
  __forceinline__
  void operator()(double* out, double* in) const {
    *out = *in * val;
  }

  __device__
  __forceinline__
  void operator()(double* v) const {
    *v *= val;
  }

  __host__ __device__
  ~TensorDivConstantOp() {}

  double val;
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorDivConstantOp<half> {
  TensorDivConstantOp() = default;
  TensorDivConstantOp(const TensorDivConstantOp&) = default;
  TensorDivConstantOp(TensorDivConstantOp&&) = default;

  explicit
#ifdef CUDA_HALF_INSTRUCTIONS
  TensorDivConstantOp(half v) : val(ScalarInv<half>::to(v)) {}
#else
  TensorDivConstantOp(half v) : fval(1.f / THC_half2float(v)) {}
#endif
  __device__ __forceinline__
  void operator()(half* out, half* in) const {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hmul(*in, val);
#else
    float fin = __half2float(*in);
    float fout = fin * fval;
    *out = __float2half(fout);
#endif
  }

  __device__ __forceinline__
  void operator()(half* v) const {
#ifdef CUDA_HALF_INSTRUCTIONS
    *v = __hmul(*v, val);
#else
    float fv = __half2float(*v);
    fv *= fval;
    *v = __float2half(fv);
#endif
  }

  __host__ __device__
  ~TensorDivConstantOp() {}

#ifdef CUDA_HALF_INSTRUCTIONS
  half val;
#else
  float fval;
#endif
};
#endif // CUDA_HALF_TENSOR

template <int Upper>
struct TensorTriOp {
  TensorTriOp() = default;
  __host__ __device__
  TensorTriOp(const TensorTriOp& x)
      : start(x.start), stride0(x.stride0), stride1(x.stride1), k(x.k)
  {}
  TensorTriOp(TensorTriOp&&) = default;

  __host__ __device__
  TensorTriOp(float *start_, long stride0_, long stride1_, long k_)
    : start(start_), stride0(stride0_), stride1(stride1_), k(k_) {}

  __device__ __forceinline__
  int mask(float *in) const {
    ptrdiff_t n = in - start;
    long row, col;
    if (stride0 > stride1)
    {
      row = (long) (n / stride0);
      col = (long) ((n % stride0) / stride1);
    }
    else
    {
      row = (long) ((n % stride1) / stride0);
      col = (long) (n / stride1);
    }

    return Upper ? (col - row >= k) : (col - row <= k);
  }

  __device__ __forceinline__
  void operator()(float* out, float* in) const { *out = mask(in) ? *in : 0; }

  __device__ __forceinline__
  void operator()(float* v) const { if (!mask(v)) *v = 0; }

  __host__ __device__
  ~TensorTriOp() {}

  float *start;
  long stride0;
  long stride1;
  long k;
};

void THCudaTensor_tril(THCState *state, THCudaTensor *self_, THCudaTensor *src_, long k)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src_));
  THArgCheck(src_->nDimension == 2, 1, "expected a matrix");

  THCudaTensor *src = src_;
  if (self_ == src_)
    src = THCudaTensor_newContiguous(state, src_);

  long stride0 = src->stride[0];
  long stride1 = src->stride[1];
  float *start = THCudaTensor_data(state, src) + src->storageOffset;

  TensorTriOp<0> op(start, stride0, stride1, k);

  if (self_ == src_) {
    if (!THC_pointwiseApply1(state, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src);

    if (!THC_pointwiseApply2(state, self_, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  if (self_ == src_)
    THCudaTensor_freeCopyTo(state, src, src_);

  THCudaCheck(hipGetLastError());
}

void THCudaTensor_triu(THCState *state, THCudaTensor *self_, THCudaTensor *src_, long k)
{
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src_));
  THArgCheck(src_->nDimension == 2, 1, "expected a matrix");

  THCudaTensor *src = src_;
  if (self_ == src_)
    src = THCudaTensor_newContiguous(state, src_);

  long stride0 = src->stride[0];
  long stride1 = src->stride[1];
  float *start = THCudaTensor_data(state, src) + src->storageOffset;

  TensorTriOp<1> op(start, stride0, stride1, k);

  if (self_ == src_) {
    if (!THC_pointwiseApply1(state, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src);

    if (!THC_pointwiseApply2(state, self_, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  if (self_ == src_)
    THCudaTensor_freeCopyTo(state, src, src_);

  THCudaCheck(hipGetLastError());
}

#include "generic/THCTensorMathPairwise.cu"
#include "THCGenerateAllTypes.h"

// Copy the kth diagonal of a matrix B to a vector A.
__global__
void THCudaTensor_copyFromDiagonal(hipLaunchParm lp, float* a, float* b, ptrdiff_t start, ptrdiff_t size, ptrdiff_t strideSum, ptrdiff_t strideA) {
  for (ptrdiff_t linearIndex = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
       linearIndex < size;
       linearIndex += hipGridDim_x * hipBlockDim_x) {
    const ptrdiff_t bOffset = start + strideSum * linearIndex;
    a[strideA * linearIndex] = b[bOffset];
  }
}

// Copy vector B to the kth diagonal of a matrix A
__global__
void THCudaTensor_copyToDiagonal(hipLaunchParm lp, float* a, float* b, ptrdiff_t start, ptrdiff_t size, ptrdiff_t strideSum, ptrdiff_t strideB) {
  for (ptrdiff_t linearIndex = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
       linearIndex < size;
       linearIndex += hipGridDim_x * hipBlockDim_x) {
    const ptrdiff_t aOffset = start + strideSum * linearIndex;
    a[aOffset] = b[strideB * linearIndex];
  }
}

void THCudaTensor_diag(THCState *state, THCudaTensor *self_, THCudaTensor *src_, long k){
  THAssert(THCudaTensor_checkGPU(state, 2, self_, src_));
  int nDimension = THCudaTensor_nDimension(state, src_);
  THArgCheck((nDimension == 2) || (nDimension == 1), 1, "expected a matrix or a vector");
  if (nDimension == 2) {
    long stride0 = THCudaTensor_stride(state, src_, 0);
    long stride1 = THCudaTensor_stride(state, src_, 1);
    long size0 = THCudaTensor_size(state, src_, 0);
    long size1 = THCudaTensor_size(state, src_, 1);
    long size = (k > 0) ? min((long long)size0, (long long)size1 - k) : min((long long)size0 + k, (long long)size1);
    THCudaTensor_resize1d(state, self_, size);
    long strideSelf = THCudaTensor_stride(state, self_, 0);
    const dim3 threads(min((long long)THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock, (long long)size));
    dim3 grid(min((long long)1024, (long long)THCCeilDiv(size, (long)threads.x)));
    long start = (k >= 0 ? k * stride1 : -k * stride0);
    hipLaunchKernel(HIP_KERNEL_NAME(THCudaTensor_copyFromDiagonal),
                    dim3(grid),
                    dim3(threads),
                    0,
                    THCState_getCurrentStream(state),
                    THCudaTensor_data(state, self_),
                    THCudaTensor_data(state, src_),
                    start,
                    size,
                    stride0 + stride1,
                    strideSelf);
  } else {
    ptrdiff_t totalElements = THCudaTensor_nElement(state, src_);
    ptrdiff_t size = (k > 0) ? totalElements + k : totalElements - k;
    long strideSrc = THCudaTensor_stride(state, src_, 0);
    THCudaTensor_resize2d(state, self_, size, size);
    THCudaTensor_zero(state, self_);
    long stride0 = THCudaTensor_stride(state, self_, 0);
    long stride1 = THCudaTensor_stride(state, self_, 1);
    const dim3 threads(min((long long)THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock, (long long)size));
    dim3 grid(min((long long)1024, (long long)THCCeilDiv(size, (ptrdiff_t)threads.x)));
    ptrdiff_t start = (k >= 0 ? k * stride1 : -k * stride0);
    hipLaunchKernel(HIP_KERNEL_NAME(THCudaTensor_copyToDiagonal),
                    dim3(grid),
                    dim3(threads),
                    0,
                    THCState_getCurrentStream(state),
                    THCudaTensor_data(state, self_),
                    THCudaTensor_data(state, src_),
                    start,
                    totalElements,
                    stride0 + stride1,
                    strideSrc);
  }
  THCudaCheck(hipGetLastError());
}

float THCudaTensor_trace(THCState *state, THCudaTensor *src_) {
  THAssert(THCudaTensor_checkGPU(state, 1, src_));
  THArgCheck((src_->nDimension == 2), 1, "expected a matrix");
  THCudaTensor *diag = THCudaTensor_new(state);
  THCudaTensor_diag(state, diag, src_, 0);
  float trace = THCudaTensor_sumall(state, diag);
  THCudaTensor_free(state, diag);
  return trace;
}
