#ifndef THC_TENSORMATH_POINTWISE_CUH
#define THC_TENSORMATH_POINTWISE_CUH

#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCHalf.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCNumerics.cuh"
#include "THCReduce.cuh"

template <typename T>
struct TensorSigmoidOp {
  __device__ __forceinline__ void operator()(T* out, T* in) const {
    T one = (T) 1.0;
    *out = one / (one + THCNumerics<T>::exp(- *in));
  }

  __device__ __forceinline__ void operator()(T* v) const {
    T one = (T) 1.0;
    *v = one / (one + THCNumerics<T>::exp(- *v));
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorSigmoidOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) const {
#ifdef CUDA_HALF_INSTRUCTIONS
    half one = ScalarConvert<int, half>::to(1);
    *out = hdiv(one, __hadd(one, hexp(__hneg(*in))));
#else
    float fin = __half2float(*in);
    *out = __float2half(1.0f / (1.0f + expf(- fin)));
#endif
  }

  __device__ __forceinline__ void operator()(half* v) const {
#ifdef CUDA_HALF_INSTRUCTIONS
    half one = ScalarConvert<int, half>::to(1);
    *v = hdiv(one, __hadd(one, hexp(__hneg(*v))));
#else
    float fv = __half2float(*v);
    *v = __float2half(1.0f / (1.0f + expf(- fv)));
#endif
  }
};
#endif

template <typename T>
struct TensorSignOp {
  __device__ __forceinline__ void operator()(T* out, T* in) const {
    T orig = *in;
    *out = (orig > 0) - (orig < 0);
  }

  __device__ __forceinline__ void operator()(T* v) const {
    T orig = *v;
    *v = (orig > 0) - (orig < 0);
  }
};

template <>
struct TensorSignOp<unsigned char> {
  __device__ __forceinline__ void operator()(unsigned char* out,
                                             unsigned char* in) const {
    unsigned char orig = *in;
    *out = (orig == 0) ? 0 : 1;
  }

  __device__ __forceinline__ void operator()(unsigned char* v) const {
    unsigned char orig = *v;
    *v = (orig == 0) ? 0 : 1;
  }

  float foo;
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorSignOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) const {
//#ifdef CUDA_HALF_INSTRUCTIONS
    half zero = ScalarConvert<int, half>::to(0);
    half orig = *in;
    *out = __float2half((float) __hgt(orig, zero) - (float) __hlt(orig, zero));
//#else
//    float orig = __half2float(*in);
//    *out = __float2half((orig > 0) - (orig < 0));
//#endif
  }

  __device__ __forceinline__ void operator()(half* v) const {
//#ifdef CUDA_HALF_INSTRUCTIONS
    half zero = ScalarConvert<int, half>::to(0);
    half orig = *v;
    *v = __float2half((float) __hgt(orig, zero) -  (float) __hlt(orig, zero));
//#else
//    float orig = __half2float(*v);
//    *v = __float2half((orig > 0) - (orig < 0));
//#endif
  }
};
#endif

template <typename T>
struct TensorAddOp {
  __device__ __forceinline__ void operator()(T* out, T* in) const {
    *out += *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) const {
    *out = *in1 + *in2;
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorAddOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) const {
//#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hadd(*out, *in);
//#else
//    float fout = __half2float(*out);
//    float fin = __half2float(*in);
//    fout += fin;
//    *out = __float2half(fout);
//#endif
  }

  __device__ __forceinline__ void operator()(half* out, half* in1, half* in2) const {
//#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hadd(*in1, *in2);
//#else
//    float fin1 = __half2float(*in1);
//    float fin2 = __half2float(*in2);
//    float fout = fin1 + fin2;
//    *out = __float2half(fout);
//#endif
  }
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorCAddOp {
  __host__ __device__
  explicit
  TensorCAddOp(T v) : val(v) {}

  __device__
  __forceinline__
  void operator()(T* out, T* in) const {
    *out += val * *in;
  }

  __device__
  __forceinline__
  void operator()(T* out, T* in1, T* in2) const {
    *out = *in1 + val * *in2;
  }

  __host__ __device__
  ~TensorCAddOp() {}

  T val;
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorCAddOp<half> {
  TensorCAddOp() = default;
  TensorCAddOp(const TensorCAddOp&) = default;
  TensorCAddOp(TensorCAddOp&&) = default;

  __host__ __device__
  explicit
  TensorCAddOp(half v) : val(v) {}

  __device__
  __forceinline__
  void operator()(half* out, half* in) const {
//#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hadd(*out, __hmul(val, *in));
//#else
//    float fout = __half2float(*out);
//    float fval = __half2float(val);
//    float fin = __half2float(*in);

//    fout += fval * fin;
//    *out = __float2half(fout);
//#endif
  }

  __device__
  __forceinline__
  void operator()(half* out, half* in1, half* in2) const {
//#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hadd(*in1, __hmul(val, *in2));
//#else
//    float fin1 = __half2float(*in1);
//    float fin2 = __half2float(*in2);
//    float fval = __half2float(val);

//    float fout = fin1 + fval * fin2;
//    *out = __float2half(fout);
//#endif
  }

  __host__ __device__
  ~TensorCAddOp() {}

  half val;
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorSubOp {
  __device__ __forceinline__ void operator()(T* out, T* in) const {
    *out -= *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) const {
    *out = *in1 - *in2;
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorSubOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) const {
//#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hsub(*out, *in);
//#else
//    float fout = __half2float(*out);
//    float fin = __half2float(*in);
//    fout -= fin;
//    *out = __float2half(fout);
//#endif
  }

  __device__ __forceinline__ void operator()(half* out,
                                             half* in1,
                                             half* in2) const {
//#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hsub(*in1, *in2);
//#else
//    float fin1 = __half2float(*in1);
//    float fin2 = __half2float(*in2);
//    float fout = fin1 - fin2;
//    *out = __float2half(fout);
//#endif
  }
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorMulOp {
  __device__ __forceinline__ void operator()(T* out, T* in) const {
    *out *= *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) const {
    *out = *in1 * *in2;
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorMulOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) const {
//#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hmul(*out, *in);
//#else
//    float fout = __half2float(*out);
//    float fin = __half2float(*in);
//    fout *= fin;
//    *out = __float2half(fout);
//#endif
  }

  __device__ __forceinline__ void operator()(half* out,
                                             half* in1,
                                             half* in2) const {
//#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hmul(*in1, *in2);
//#else
//    float fin1 = __half2float(*in1);
//    float fin2 = __half2float(*in2);
//    float fout = fin1 * fin2;
//    *out = __float2half(fout);
//#endif
  }
};
#endif // CUDA_HALF_TENSOR

template<typename T>
struct TensorPowOp {
  __host__ __device__
  explicit
  TensorPowOp(T v) : val(v) {}

  __device__ __forceinline__ void operator()(T* out, T* in) const {
    *out = powf((float) *in, (float) val);
  }

  __device__ __forceinline__ void operator()(T* v) const {
    *v = powf((float) *v, (float) val);
  }

  __host__ __device__
  ~TensorPowOp() {}

  T val;
};

template <>
struct TensorPowOp<double> {
  __host__ __device__
  explicit
  TensorPowOp(double v) : val(v) {}

  __device__ __forceinline__ void operator()(double* out, double* in) const {
    *out = pow(*in, val);
  }

  __device__ __forceinline__ void operator()(double* v) const {
    *v = pow(*v, val);
  }

  __host__ __device__
  ~TensorPowOp() {}

  double val;
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorPowOp<half> {
  __host__ __device__
  explicit
  TensorPowOp(half v) : val(v) {}

  __device__ __forceinline__ void operator()(half* out, half* in) const {
    // No fp16 pow function yet
    float fin = __half2float(*in);
    float fval = __half2float(val);
    float fout = powf(fin, fval);
    *out = __float2half(fout);
  }

  __device__ __forceinline__ void operator()(half* v) const {
    // No fp16 pow function yet
    float fv = __half2float(*v);
    float fval = __half2float(val);
    float fout = powf(fv, fval);
    *v = __float2half(fout);
  }

  __host__ __device__
  ~TensorPowOp() {}

  half val;
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorCPowOp {
  __device__ __forceinline__ void operator()(T* out, T* in) const {
    *out = powf((float) *out, (float) *in);
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) const {
    *out = powf((float) *in1, (float) *in2);
  }
};

template <>
struct TensorCPowOp<double> {
  __device__ __forceinline__ void operator()(double* out, double* in) const {
    *out = pow(*out, *in);
  }

  __device__ __forceinline__ void operator()(double* out,
                                             double* in1,
                                             double* in2) const {
    *out = pow(*in1, *in2);
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorCPowOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) const {
    // No fp16 pow function yet
    float fout = __half2float(*out);
    float fin = __half2float(*in);
    fout = powf(fout, fin);
    *out = __float2half(fout);
  }

  __device__ __forceinline__ void operator()(half* out,
                                             half* in1,
                                             half* in2) const {
    // No fp16 pow function yet
    float fin1 = __half2float(*in1);
    float fin2 = __half2float(*in2);
    float fout = powf(fin1, fin2);
    *out = __float2half(fout);
  }
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorDivOp {
  __device__ __forceinline__
  void operator()(T* out, T* in) const {
    *out /= *in;
  }

  __device__ __forceinline__
  void operator()(T* out, T* in1, T* in2) const {
    *out = *in1 / *in2;
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorDivOp<half> {
  __device__ __forceinline__
  void operator()(half* out, half* in) const {
    // No fp16 div instruction yet
    float fout = __half2float(*out);
    float fin = __half2float(*in);
    fout /= fin;
    *out = __float2half(fout);
  }

  __device__ __forceinline__
  void operator()(half* out, half* in1, half* in2) const {
    // No fp16 div instruction yet
    float fin1 = __half2float(*in1);
    float fin2 = __half2float(*in2);
    float fout = fin1 / fin2;
    *out = __float2half(fout);
  }
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorClampOp {
  __host__ __device__
  TensorClampOp() : minValue{FLT_MIN}, maxValue{FLT_MAX} {}
  __host__ __device__
  TensorClampOp(const TensorClampOp& x)
    : minValue{x.minValue}, maxValue{x.maxValue} {}
  TensorClampOp(TensorClampOp&&) = default;

  __host__ __device__
  TensorClampOp(T min, T max) : minValue(min), maxValue(max) {}
  __device__ __forceinline__ void operator()(T* out, T* in) const {
    // TODO: the oddly named temporaries are in place as a workaround for HCC
    //       codegen bugs with FP16, and should be removed as soon as possible.
    const auto foo = maxValue;
    const auto bar = minValue;
    T val = THCNumerics<T>::lt(*in, foo) ? *in : foo;
    *out = THCNumerics<T>::gt(bar, val) ? bar : val;
  }

  __device__ __forceinline__ void operator()(T* v) const {
    // TODO: the oddly named temporaries are in place as a workaround for HCC
    //       codegen bugs with FP16, and should be removed as soon as possible.
    const auto foo = minValue;
    const auto bar = maxValue;
    T val = THCNumerics<T>::lt(*v, bar) ? *v : bar;
    *v = THCNumerics<T>::gt(foo, val) ? foo : val;
  }

  __host__ __device__
  ~TensorClampOp() {}

  T minValue;
  T maxValue;
};

template <typename T>
struct TensorLerpOp {
  __host__ __device__
  explicit
  TensorLerpOp(T w) : w(w) {}

  __device__ __forceinline__ void operator()(T *out, T *a, T *b) const {
    *out = THCNumerics<T>::add(
      *a,
      THCNumerics<T>::mul(
          w,
          THCNumerics<T>::sub(*b, *a)
        )
    );
  }

  __host__ __device__
  ~TensorLerpOp() {}

  T w;
};

template <typename T>
struct TensorCrossOp {
  __host__ __device__
  TensorCrossOp(long sx, long sy, long so) : sx(sx), sy(sy), so(so) {}

  __device__ __forceinline__ void operator()(T* out, T* x, T*y) const {
    out[0 * so] = THCNumerics<T>::sub(
        THCNumerics<T>::mul(x[1 * sx], y[2 * sy]),
        THCNumerics<T>::mul(x[2 * sx], y[1 * sy])
    );

    out[1 * so] = THCNumerics<T>::sub(
        THCNumerics<T>::mul(x[2 * sx], y[0 * sy]),
        THCNumerics<T>::mul(x[0 * sx], y[2 * sy])
    );

    out[2 * so] = THCNumerics<T>::sub(
        THCNumerics<T>::mul(x[0 * sx], y[1 * sy]),
        THCNumerics<T>::mul(x[1 * sx], y[0 * sy])
    );
  }

  __host__ __device__
  ~TensorCrossOp() {}

  long sx, sy, so;
};

template <typename T>
struct TensorMaxOp {
  __device__ __forceinline__ void operator()(T* out, T* in) const {
    *out = THCNumerics<T>::gt(*out, *in) ? *out : *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) const {
    *out = THCNumerics<T>::gt(*in1, *in2) ? *in1 : *in2;
  }
};

template <typename T>
struct TensorMinOp {
  __device__ __forceinline__ void operator()(T* out, T* in) const {
    *out = THCNumerics<T>::lt(*out, *in) ? *out : *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) const {
    *out = THCNumerics<T>::lt(*in1, *in2) ? *in1 : *in2;
  }
};

template <typename T>
struct TensorMaxValueOp {
  __host__ __device__
  explicit
  TensorMaxValueOp(T v) : val(v) {}

  __device__ __forceinline__
  void operator()(T* out) const {
    // TODO: the oddly named temporary is in place as a workaround for HCC
    //       codegen bugs with FP16, and should be removed as soon as possible.
    const auto foo = val;
    *out = THCNumerics<T>::gt(*out, foo) ? *out : foo;
  }

  __device__ __forceinline__
  void operator()(T* out, T* in) const {
    // TODO: the oddly named temporary is in place as a workaround for HCC
    //       codegen bugs with FP16, and should be removed as soon as possible.
    const auto foo = val;
    *out = THCNumerics<T>::gt(*in, foo) ? *in : foo;
  }

  __host__ __device__
  ~TensorMaxValueOp() {}

  T val;
};

template <typename T>
struct TensorMinValueOp {
  __host__ __device__
  explicit
  TensorMinValueOp(T v) : val(v) {}

  __device__ __forceinline__ void operator()(T* out) const {
    // TODO: the oddly named temporary is in place as a workaround for HCC
    //       codegen bugs with FP16, and should be removed as soon as possible.
    const auto foo = val;
    *out = THCNumerics<T>::lt(*out, foo) ? *out : foo;
    //out = THCNumerics<T>::lt(out, val) ? out : val;
  }

  __device__ __forceinline__ void operator()(T* out, T* in) const {
    // TODO: the oddly named temporary is in place as a workaround for HCC
    //       codegen bugs with FP16, and should be removed as soon as possible.
    const auto foo = val;
    *out = THCNumerics<T>::lt(*in, foo) ? *in : foo;
  }

  __host__ __device__
  ~TensorMinValueOp() {}

  T val;
};

template <typename T>
struct TensorAddCMulOp {
  __host__ __device__
  explicit
  TensorAddCMulOp(T v) : val(v) {}

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) const {
    *out = THCNumerics<T>::add(
      *out,
      THCNumerics<T>::mul(
        val,
        THCNumerics<T>::mul(*in1, *in2)
      )
    );
  }

  __host__ __device__
  ~TensorAddCMulOp() {}

  T val;
};

template <typename T>
struct TensorAddCDivOp {
  __host__ __device__
  explicit
  TensorAddCDivOp(T v) : val(v) {}

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) const {
    *out = THCNumerics<T>::add(
      *out,
      THCNumerics<T>::mul(
        val,
        THCNumerics<T>::div(*in1, *in2)
      )
    );
  }

  __host__ __device__
  ~TensorAddCDivOp() {}

  T val;
};

#endif // THC_TENSORMATH_POINTWISE_CUH
