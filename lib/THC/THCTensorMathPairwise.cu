#include "hip/hip_runtime.h"
#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCHalf.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCNumerics.cuh"
#include "THCTensorMathCompareT.cuh"

template <typename T>
struct TensorAddConstantOp {
  __host__ __device__
  explicit
  TensorAddConstantOp(T v) : val(v) {}

  __device__ __forceinline__
  void operator()(T* out, T* in) { *out = *in + val; }

  __device__ __forceinline__
  void operator()(T* v) { *v += val; }

  __host__ __device__
  ~TensorAddConstantOp() {}

  T val;
};

#ifdef CUDA_HALF_TENSOR
    template <>
    struct TensorAddConstantOp<half> {
    #if defined(CUDA_HALF_INSTRUCTIONS) || defined(__HIP_PLATFORM_HCC__)
      __host__ __device__
      explicit
      TensorAddConstantOp(half v) : val(v) {}
    #else
      __host__
      explicit
      TensorAddConstantOp(half v) : fval(THC_half2float(v)) {}
    #endif

      __device__ __forceinline__
      void operator()(half* out, half* in)
      {
        #if defined(__HIP_PLATFORM_HCC__)
          *out = *in + val;
        #elif defined(CUDA_HALF_INSTRUCTIONS)
            *out = __hadd(*in, val);
        #else
          float fin = __half2float(*in);
          float fout = fin + fval;
          *out = __float2half(fout);
        #endif
      }

      __device__ __forceinline__
      void operator()(half* v)
      {
        #if defined(__HIP_PLATFORM_HCC__)
           *v += val;
        #elif defined(CUDA_HALF_INSTRUCTIONS)
          *v = __hadd(*v, val);
        #else
          float fv = __half2float(*v);
          fv += fval;
          *v = __float2half(fv);
        #endif
      }

    #if defined(CUDA_HALF_INSTRUCTIONS) || defined(__HIP_PLATFORM_HCC__)
      half val;
    #else
      float fval;
    #endif
    };
#endif // CUDA_HALF_TENSOR


template <typename T>
struct TensorSubConstantOp {
  __host__ __device__
  explicit
  TensorSubConstantOp(T v) : val(v) {}

  __device__ __forceinline__
  void operator()(T* out, T* in) { *out = *in - val; }

  __device__ __forceinline__
  void operator()(T* v) { *v -= val; }

  __host__ __device__
  ~TensorSubConstantOp() {}

  T val;
};


#ifdef CUDA_HALF_TENSOR
    template <>
    struct TensorSubConstantOp<half> {
      #if defined(__HIP_PLATFORM_HCC__)
        __host__ __device__
        explicit
        TensorSubConstantOp(half v) : val{v} {}
      #elif defined(CUDA_HALF_INSTRUCTIONS)
        __host__ __device__
        explicit
        TensorSubConstantOp(half v)
          : val(THC_float2half(-(THC_half2float(v)))) {}
      #else
        __host__
        explicit
        TensorSubConstantOp(half v): fval(-(THC_half2float(v))) {}
      #endif

      __device__ __forceinline__
      void operator()(half* out, half* in)
      {
        #if defined(__HIP_PLATFORM_HCC__)
          *out = *in + val;
        #elif defined(CUDA_HALF_INSTRUCTIONS)
          *out = __hadd(*in, val);
        #else
          float fin = __half2float(*in);
          float fout = fin + fval;
          *out = __float2half(fout);
        #endif
      }

      __device__ __forceinline__
      void operator()(half* v)
      {
        #if defined(__HIP_PLATFORM_HCC__)
          *v += val;
        #elif defined(CUDA_HALF_INSTRUCTIONS)
          *v = __hadd(*v, val);
        #else
          float fv = __half2float(*v);
          fv += fval;
          *v = __float2half(fv);
        #endif
      }

    #if defined(CUDA_HALF_INSTRUCTIONS) || defined(__HIP_PLATFORM_HCC__)
      half val;
    #else
      float fval;
    #endif
    };
#endif // CUDA_HALF_TENSOR


template <typename T>
struct TensorMulConstantOp {
  __host__ __device__
  explicit
  TensorMulConstantOp(T v) : val(v) {}

  __device__ __forceinline__
  void operator()(T* out, T* in) { *out = *in * val; }

  __device__ __forceinline__
  void operator()(T* v) { *v *= val; }

  __host__ __device__
  ~TensorMulConstantOp() {}

  T val;
};

#ifdef CUDA_HALF_TENSOR
    template <>
    struct TensorMulConstantOp<half> {
      #if defined(CUDA_HALF_INSTRUCTIONS) || defined(__HIP_PLATFORM_HCC__)
        __host__ __device__
        explicit
        TensorMulConstantOp(half v) : val(v) {}
      #else
        explicit
        TensorMulConstantOp(half v) : fval(THC_half2float(v)) {}
      #endif

        __device__ __forceinline__
        void operator()(half* out, half* in)
        {
          #if defined(__HIP_PLATFORM_HCC__)
            *out = *in * val;
          #elif defined(CUDA_HALF_INSTRUCTIONS)
            *out = __hmul(*in, val);
          #else
            float fin = __half2float(*in);
            float fout = fin * fval;
            *out = __float2half(fout);
          #endif
        }

        __device__ __forceinline__
        void operator()(half* v)
        {
          #if defined(__HIP_PLATFORM_HCC__)
            *v = *v * val;
          #elif defined(CUDA_HALF_INSTRUCTIONS)
            *v = __hmul(*v, val);
          #else
            float fv = __half2float(*v);
            fv *= fval;
            *v = __float2half(fv);
          #endif
        }

        #if defined(CUDA_HALF_INSTRUCTIONS) || defined(__HIP_PLATFORM_HCC__)
          half val;
        #else
          float fval;
        #endif
    };
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorDivConstantOp {
  __host__ __device__
  explicit
  TensorDivConstantOp(T v) : val(v) {}
  __device__ __forceinline__
  void operator()(T* out, T* in) { *out = *in / val; }

  __device__ __forceinline__
  void operator()(T* v) { *v /= val; }

  __host__ __device__
  ~TensorDivConstantOp() {}

  T val;
};

template <>
struct TensorDivConstantOp<float> {
  __host__ __device__
  explicit
  TensorDivConstantOp(float v) : val(1.f / v) {}
  __device__ __forceinline__
  void operator()(float* out, float* in) { *out = *in * val; }

  __device__ __forceinline__
  void operator()(float* v) { *v *= val; }

  __host__ __device__
  ~TensorDivConstantOp() {}

  float val;
};

template <>
struct TensorDivConstantOp<double> {
  __host__ __device__
  explicit
  TensorDivConstantOp(double v) : val(1. / v) {}

  __device__ __forceinline__
  void operator()(double* out, double* in) { *out = *in * val; }

  __device__ __forceinline__
  void operator()(double* v) { *v *= val; }

  __host__ __device__
  ~TensorDivConstantOp() {}

  double val;
};

#ifdef CUDA_HALF_TENSOR
  template <>
  struct TensorDivConstantOp<half> {
    #if defined(CUDA_HALF_INSTRUCTIONS) || defined(__HIP_PLATFORM_HCC__)
      __host__ __device__
      explicit
      TensorDivConstantOp(half v) : val(ScalarInv<half>::to(v)) {}
    #else
      TensorDivConstantOp(half v) : fval(1.f / THC_half2float(v)) {}
    #endif
    __device__ __forceinline__
    void operator()(half* out, half* in)
    {
      #if defined(__HIP_PLATFORM_HCC__)
        *out = *in * val;
      #elif defined(CUDA_HALF_INSTRUCTIONS)
        *out = __hmul(*in, val);
      #else
        float fin = __half2float(*in);
        float fout = fin * fval;
        *out = __float2half(fout);
      #endif
    }

    __device__ __forceinline__
    void operator()(half* v)
    {
      #if defined(__HIP_PLATFORM_HCC__)
        *v *= val;
      #elif defined(CUDA_HALF_INSTRUCTIONS)
        *v = __hmul(*v, val);
      #else
        float fv = __half2float(*v);
        fv *= fval;
        *v = __float2half(fv);
      #endif
    }

      #if defined(CUDA_HALF_INSTRUCTIONS) || defined(__HIP_PLATFORM_HCC__)
        half val;
      #else
        float fval;
      #endif
  };
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorRemainderOp {
  __host__ __device__ TensorRemainderOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in % val;
    if ((*out * val) < 0){
      *out += val;
    }
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v = *v % val;
    if ((*v * val) < 0){
      *v += val;
    }
  }

  __host__ __device__ ~TensorRemainderOp() {};
  const T val;
};

template <>
struct TensorRemainderOp<float> {
  __host__ __device__ TensorRemainderOp(float v) : val(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = *in - val * floorf(*in / val);
  }

  __device__ __forceinline__ void operator()(float* v) {
    *v = *v - val * floorf(*v / val);
  }

  __host__ __device__ ~TensorRemainderOp() {};
  const float val;
};

template <>
struct TensorRemainderOp<double> {
  __host__ __device__ TensorRemainderOp(double v) : val(v) {}
  __device__ __forceinline__ void operator()(double* out, double* in) {
    *out = *in - val * floor(*in / val);
  }

  __device__ __forceinline__ void operator()(double* v) {
    *v = *v - val * floor(*v / val);
  }

  __host__ __device__ ~TensorRemainderOp() {};
  const double val;
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorRemainderOp<half> {
#ifdef CUDA_HALF_INSTRUCTIONS
  __host__ __device__ TensorRemainderOp(half v) : val(v) {}
#else
#ifdef __NVCC__
  __host__ __device__ TensorRemainderOp(half v): fval(THC_half2float(v)) {}
#else
  __host__ __device__ TensorRemainderOp(half v): fval(v) {}
#endif
#endif

  __device__ __forceinline__ void operator()(half* out, half* in) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hsub(*in,  __hmul(val, hfloor(__hdiv(*in,  val))));
#else
    float fin = __half2float(*in);
    float fout = fin - fval * floorf(fin / fval);
    *out = __float2half(fout);
#endif
  }

  __device__ __forceinline__ void operator()(half* v) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *v = __hsub(*v, __hmul(val, hfloor(__hdiv(*v, val))));
#else
    float fv = __half2float(*v);
    fv = fv - fval * floorf(fv / fval);
    *v = __float2half(fv);
#endif
  }

#ifdef CUDA_HALF_INSTRUCTIONS
  const half val;
#else
  const float fval;
#endif
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorFmodOp {
  __host__ __device__ TensorFmodOp(T v) : val((float)v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = (T) fmodf((float) *in, val);
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v = (T) fmodf((float) *v, val);
  }

  const float val;
};

template <>
struct TensorFmodOp<double> {
  __host__ __device__ TensorFmodOp(double v) : val(v) {}
  __device__ __forceinline__ void operator()(double* out, double* in) {
    *out = fmod(*in, val);
  }

  __device__ __forceinline__ void operator()(double* v) {
    *v = fmod(*v, val);
  }

  const double val;
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorFmodOp<half> {
#ifdef CUDA
__host__ __device__  TensorFmodOp(half v): fval(THC_half2float(v)) {}
#else
__host__ __device__  TensorFmodOp(half v): fval((v)) {}
#endif
  __device__ __forceinline__ void operator()(half* out, half* in) {
    *out = __float2half(fmodf(__half2float(*in), fval));
  }

  __device__ __forceinline__ void operator()(half* v) {
    *v = __float2half(fmodf(__half2float(*v), fval));
  }

  const float fval;
};
#endif // CUDA_HALF_TENSOR
template <typename T, int Upper>
struct TensorTriOp {
  TensorTriOp(T *start_, long stride0_, long stride1_, long k_)
    : start(start_), stride0(stride0_), stride1(stride1_), k(k_) {}

  __device__ __forceinline__ int mask(T *in) {
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

  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = mask(in) ? *in : ScalarConvert<int, T>::to(0);
  }

  __device__ __forceinline__ void operator()(T* v) {
    if (!mask(v))
      *v = ScalarConvert<int, T>::to(0);
  }

  const T *start;
  const long stride0, stride1, k;
};


template <typename T>
struct TensorLShiftConstantOp {
  __host__ __device__ TensorLShiftConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in << val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v <<= val;
  }

  __host__ __device__ ~TensorLShiftConstantOp() {};
  const T val;
};

template <typename T>
struct TensorRShiftConstantOp {
  __host__ __device__ TensorRShiftConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in >> val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v >>= val;
  }

  __host__ __device__ ~TensorRShiftConstantOp() {};
  const T val;
};

template <typename T>
struct TensorBitAndConstantOp {
  __host__ __device__ TensorBitAndConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in & val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v &= val;
  }
  
  __host__ __device__ ~TensorBitAndConstantOp() {}
  const T val;
};

template <typename T>
struct TensorBitOrConstantOp {
  __host__ __device__  TensorBitOrConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in | val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v |= val;
  }

  __host__ __device__ ~TensorBitOrConstantOp() {}
  const T val;
};

template <typename T>
struct TensorBitXorConstantOp {
  __host__ __device__ TensorBitXorConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in ^ val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v ^= val;
  }

  __host__ __device__ ~TensorBitXorConstantOp() {}
  const T val;
};

#include "generic/THCTensorMathPairwise.cu"
#include "THCGenerateAllTypes.h"
