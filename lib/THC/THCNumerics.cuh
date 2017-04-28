#ifndef THC_NUMERICS_INC
#define THC_NUMERICS_INC

#ifdef CUDA_PATH
  #include <cuda.h>
#endif

#include "hip/hip_runtime.h"

#include "THCHalf.h"

#include <climits>

/// Class for numeric limits of the particular data type, which
/// includes support for `half`.
/// Unfortunately since `half` does not have a constructor, these have
/// to be expressed as functions (either that or non-const statics).
template <typename T>
struct THCNumerics {
};

template <>
struct THCNumerics<unsigned char> {
  static inline __host__ __device__ unsigned char min() { return 0; }
  static inline __host__ __device__ unsigned char max() { return UCHAR_MAX; }

  static inline __host__ __device__ bool lt(unsigned char a, unsigned char b) { return a < b; }
  static inline __host__ __device__ bool le(unsigned char a, unsigned char b) { return a <= b; }
  static inline __host__ __device__ bool gt(unsigned char a, unsigned char b) { return a > b; }
  static inline __host__ __device__ bool ge(unsigned char a, unsigned char b) { return a >= b; }
  static inline __host__ __device__ bool eq(unsigned char a, unsigned char b) { return a == b; }
  static inline __host__ __device__ bool ne(unsigned char a, unsigned char b) { return a != b; }

  static inline __host__ __device__  unsigned char add(unsigned char a, unsigned char b) { return a + b; }
  static inline __host__ __device__  unsigned char mul(unsigned char a, unsigned char b) { return a * b; }
  static inline __host__ __device__  unsigned char sub(unsigned char a, unsigned char b) { return a - b; }
  static inline __host__ __device__  unsigned char div(unsigned char a, unsigned char b) { return a / b; }
  static inline __host__ __device__  unsigned char abs(unsigned char a) { return a; }
};

template <>
struct THCNumerics<char> {
  static inline __host__ __device__ char min() { return CHAR_MIN; }
  static inline __host__ __device__ char max() { return CHAR_MAX; }

  static inline __host__ __device__ bool lt(char a, char b) { return a < b; }
  static inline __host__ __device__ bool le(char a, char b) { return a <= b; }
  static inline __host__ __device__ bool gt(char a, char b) { return a > b; }
  static inline __host__ __device__ bool ge(char a, char b) { return a >= b; }
  static inline __host__ __device__ bool eq(char a, char b) { return a == b; }
  static inline __host__ __device__ bool ne(char a, char b) { return a != b; }

  static inline __host__ __device__  char add(char a, char b) { return a + b; }
  static inline __host__ __device__  char mul(char a, char b) { return a * b; }
  static inline __host__ __device__  char sub(char a, char b) { return a - b; }
  static inline __host__ __device__  char div(char a, char b) { return a / b; }
  static inline __host__ char abs(char a) { return std::abs(a); }
  static inline __device__ char abs(char a) { return a < 0 ? -a : a; }
};

template <>
struct THCNumerics<short> {
  static inline __host__ __device__ short min() { return SHRT_MIN; }
  static inline __host__ __device__ short max() { return SHRT_MAX; }

  static inline __host__ __device__ bool lt(short a, short b) { return a < b; }
  static inline __host__ __device__ bool le(short a, short b) { return a <= b; }
  static inline __host__ __device__ bool gt(short a, short b) { return a > b; }
  static inline __host__ __device__ bool ge(short a, short b) { return a >= b; }
  static inline __host__ __device__ bool eq(short a, short b) { return a == b; }
  static inline __host__ __device__ bool ne(short a, short b) { return a != b; }

  static inline __host__ __device__  short add(short a, short b) { return a + b; }
  static inline __host__ __device__  short mul(short a, short b) { return a * b; }
  static inline __host__ __device__  short sub(short a, short b) { return a - b; }
  static inline __host__ __device__  short div(short a, short b) { return a / b; }
  static inline __host__ short abs(short a) { return std::abs(a); }
  static inline __device__ short abs(short a) { return a < 0 ? -a : a; }
};

template <>
struct THCNumerics<int> {
  static inline __host__ __device__ int min() { return INT_MIN; }
  static inline __host__ __device__ int max() { return INT_MAX; }

  static inline __host__ __device__ bool lt(int a, int b) { return a < b; }
  static inline __host__ __device__ bool le(int a, int b) { return a <= b; }
  static inline __host__ __device__ bool gt(int a, int b) { return a > b; }
  static inline __host__ __device__ bool ge(int a, int b) { return a >= b; }
  static inline __host__ __device__ bool eq(int a, int b) { return a == b; }
  static inline __host__ __device__ bool ne(int a, int b) { return a != b; }

  static inline __host__ __device__  int add(int a, int b) { return a + b; }
  static inline __host__ __device__  int mul(int a, int b) { return a * b; }
  static inline __host__ __device__  int sub(int a, int b) { return a - b; }
  static inline __host__ __device__  int div(int a, int b) { return a / b; }
  static inline __host__ int abs(int a) { return std::abs(a); }
  static inline __device__ int abs(int a) { return a < 0 ? -a : a; }
};

template <>
struct THCNumerics<long> {
  static inline __host__ __device__ long min() { return LONG_MIN; }
  static inline __host__ __device__ long max() { return LONG_MAX; }

  static inline __host__ __device__ bool lt(long a, long b) { return a < b; }
  static inline __host__ __device__ bool le(long a, long b) { return a <= b; }
  static inline __host__ __device__ bool gt(long a, long b) { return a > b; }
  static inline __host__ __device__ bool ge(long a, long b) { return a >= b; }
  static inline __host__ __device__ bool eq(long a, long b) { return a == b; }
  static inline __host__ __device__ bool ne(long a, long b) { return a != b; }

  static inline __host__ __device__  long add(long a, long b) { return a + b; }
  static inline __host__ __device__  long mul(long a, long b) { return a * b; }
  static inline __host__ __device__  long sub(long a, long b) { return a - b; }
  static inline __host__ __device__  long div(long a, long b) { return a / b; };
  static inline __host__ long abs(long a) { return std::abs(a); }
  static inline __device__ long abs(long a) { return a < 0 ? -a : a; }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct THCNumerics<half> {
    __host__ __device__
    static
    inline
    half min()
    {
        #if defined(__HIP_PLATFORM_HCC__)
            return -65504;
        #else
            half h; h.x = 0xfbff; return h;
        #endif
    }
    __host__ __device__
    static
    inline
    half max()
    {
        #if defined(__HIP_PLATFORM_HCC__)
            return 65504;
        #else
            half h; h.x = 0x7bff; return h;
        #endif
    }

  __device__
  static
  inline
  bool lt(half a, half b)
  {
      #if defined(__HIP_PLATFORM_HCC__)
        return a < b;
      #elif defined(CUDA_HALF_INSTRUCTIONS)
        return __hlt(a, b);
      #else
        float fa = __half2float(a);
        float fb = __half2float(b);
        return fa < fb;
      #endif
  }
  __host__
  static
  inline
  bool lt(half a, half b)
  {
    return THC_half2float(a) < THC_half2float(b);
  }

  __device__
  static
  inline
  bool le(half a, half b)
  {
      #if defined(__HIP_PLATFORM_HCC__)
        return a <= b;
      #elif defined(CUDA_HALF_INSTRUCTIONS)
        return __hle(a, b);
      #else
        float fa = __half2float(a);
        float fb = __half2float(b);
        return fa <= fb;
      #endif
  }
  __host__
  static
  inline
  bool le(half a, half b)
  {
    return THC_half2float(a) <= THC_half2float(b);
  }

  __device__
  static
  inline
  bool gt(half a, half b)
  {
    #if defined(__HIP_PLATFORM_HCC__)
      return a > b;
    #elif defined(CUDA_HALF_INSTRUCTIONS)
      return __hgt(a, b);
    #else
      float fa = __half2float(a);
      float fb = __half2float(b);
      return fa > fb;
    #endif
  }
  __host__
  static
  inline
  bool gt(half a, half b)
  {
    return THC_half2float(a) > THC_half2float(b);
  }

  __device__
  static
  inline
  bool ge(half a, half b)
  {
    #if defined(__HIP_PLATFORM_HCC__)
      return a >= b;
    #elif defined(CUDA_HALF_INSTRUCTIONS)
      return __hge(a, b);
    #else
      float fa = __half2float(a);
      float fb = __half2float(b);
      return fa >= fb;
    #endif
  }
  __host__
  static
  inline
  bool ge(half a, half b)
  {
    return THC_half2float(a) >= THC_half2float(b);
  }

  __device__
  static
  inline
  bool eq(half a, half b)
  {
    #if defined(__HIP_PLATFORM_HCC__)
      return a == b;
    #elif defined(CUDA_HALF_INSTRUCTIONS)
      return __heq(a, b);
    #else
      float fa = __half2float(a);
      float fb = __half2float(b);
      return fa == fb;
    #endif
  }
  __host__
  static
  inline
  bool eq(half a, half b)
  {
    return THC_half2float(a) == THC_half2float(b);
  }

  __device__
  static
  inline
  bool ne(half a, half b)
  {
    #if defined(__HIP_PLATFORM_HCC__)
      return a != b;
    #elif defined(CUDA_HALF_INSTRUCTIONS)
      return __hne(a, b);
    #else
      float fa = __half2float(a);
      float fb = __half2float(b);
      return fa != fb;
    #endif
  }
  __host__
  static
  inline
  bool ne(half a, half b)
  {
    return THC_half2float(a) != THC_half2float(b);
  }

  __device__
  static
  inline
  half exp(half a)
  {
    #if defined(__CUDA_ARCH__) && defined(CUDA_HALF_INSTRUCTIONS)
      return hexp(a);
    #else
      float fa = __half2float(a);
      return __float2half(expf(fa));
    #endif
  }
  __host__
  static
  inline
  half exp(half a)
  {
    return THC_float2half(expf(THC_half2float(a)));
  }

  __device__
  static
  inline
  half log(half a)
  {
    #ifdef CUDA_HALF_INSTRUCTIONS
      return hlog(a);
    #else
      float fa = __half2float(a);
      return __float2half(logf(fa));
    #endif
  }
  __host__
  static
  inline
  half log(half a)
  {
    return THC_float2half(logf(THC_half2float(a)));
  }

  __device__
  static
  inline
  half log1p(half a)
  {
    float fa = __half2float(a);
    return __float2half(log1pf(fa));
  }
  __host__
  static
  inline
  half log1p(half a)
  {
    return THC_float2half(log1pf(THC_half2float(a)));
  }

  __device__
  static
  inline
  half cos(half a)
  {
    #ifdef CUDA_HALF_INSTRUCTIONS
      return hcos(a);
    #else
      float fa = __half2float(a);
      return __float2half(cosf(fa));
    #endif
  }
  __host__
  static
  inline
  half cos(half a)
  {
    return THC_float2half(cosf(THC_half2float(a)));
  }

  __device__
  static
  inline
  half sin(half a)
  {
    #ifdef CUDA_HALF_INSTRUCTIONS
      return hsin(a);
    #else
      float fa = __half2float(a);
      return __float2half(sinf(fa));
    #endif
  }
  __host__
  static
  inline
  half sin(half a)
  {
    return THC_float2half(sinf(THC_half2float(a)));
  }

  __device__
  static
  inline
  half sqrt(half a)
  {
    #ifdef CUDA_HALF_INSTRUCTIONS
      return hsqrt(a);
    #else
      float fa = __half2float(a);
      return __float2half(sqrtf(fa));
    #endif
  }
  __host__
  static
  inline
  half sqrt(half a)
  {
    return THC_float2half(sqrtf(THC_half2float(a)));
  }

  __device__
  static
  inline
  half rsqrt(half a)
  {
    #ifdef CUDA_HALF_INSTRUCTIONS
      return hrsqrt(a);
    #else
      float fa = __half2float(a);
      return __float2half(rsqrtf(fa));
    #endif
  }
//  __host__
//  static
//  inline
//  half rsqrt(half a)
//  {
//    return THC_float2half(std::rsqrt(THC_half2float(a)));
//  }

  __device__
  static
  inline
  half ceil(half a)
  {
    #ifdef CUDA_HALF_INSTRUCTIONS
      return hceil(a);
    #else
      float fa = __half2float(a);
      return __float2half(ceilf(fa));
    #endif
  }
  __host__
  static
  inline
  half ceil(half a)
  {
    return THC_float2half(ceilf(THC_half2float(a)));
  }

  __device__
  static
  inline
  half floor(half a)
  {
    #ifdef CUDA_HALF_INSTRUCTIONS
      return hfloor(a);
    #else
      float fa = __half2float(a);
      return __float2half(floorf(fa));
    #endif
  }
  __host__
  static
  inline
  half floor(half a)
  {
    return THC_float2half(floorf(THC_half2float(a)));
  }

  __device__
  static
  inline
  half trunc(half a)
  {
    #ifdef CUDA_HALF_INSTRUCTIONS
      return htrunc(a);
    #else
      float fa = __half2float(a);
      return __float2half(truncf(fa));
    #endif
  }
  __host__
  static
  inline
  half trunc(half a)
  {
    return THC_float2half(truncf(THC_half2float(a)));
  }

  __device__
  static
  inline
  half neg(half a)
  {
    #if defined(__HIP_PLATFORM_HCC__)
      return -a;
    #elif defined(CUDA_HALF_INSTRUCTIONS)
      return __hneg(a);
    #else
      float fa = __half2float(a);
      return __float2half(-fa);
    #endif
  }
  __host__
  static
  inline
  half neg(half a)
  {
    return THC_float2half(-(THC_half2float(a)));
  }

  __device__
  static
  inline
  half acos(half a)
  {
    float fa = __half2float(a);
    return __float2half(acosf(fa));
  }
  __host__
  static
  inline
  half acos(half a)
  {
    return THC_float2half(acosf(THC_half2float(a)));
  }

  __device__
  static
  inline
  half cosh(half a)
  {
    float fa = __half2float(a);
    return __float2half(coshf(fa));
  }
  __host__
  static
  inline
  half cosh(half a)
  {
    return THC_float2half(coshf(THC_half2float(a)));
  }

  __device__
  static
  inline
  half asin(half a)
  {
    float fa = __half2float(a);
    return __float2half(asinf(fa));
  }
  __host__
  static
  inline
  half asin(half a)
  {
    return THC_float2half(asinf(THC_half2float(a)));
  }

  __device__
  static
  inline
  half sinh(half a)
  {
    float fa = __half2float(a);
    return __float2half(sinhf(fa));
  }
  __host__
  static
  inline
  half sinh(half a)
  {
    return THC_float2half(sinhf(THC_half2float(a)));
  }

  __device__
  static
  inline
  half tan(half a)
  {
    float fa = __half2float(a);
    return __float2half(tanf(fa));
  }
  __host__
  static
  inline
  half tan(half a)
  {
    return THC_float2half(tanf(THC_half2float(a)));
  }

  __device__
  static
  inline
  half atan(half a)
  {
    float fa = __half2float(a);
    return __float2half(atanf(fa));
  }
  __host__
  static
  inline
  half atan(half a)
  {
    return THC_float2half(atanf(THC_half2float(a)));
  }

  __device__
  static
  inline
  half tanh(half a)
  {
    float fa = __half2float(a);
    return __float2half(tanhf(fa));
  }
  __host__
  static
  inline
  half tanh(half a)
  {
    return THC_float2half(tanhf(THC_half2float(a)));
  }

  __device__
  static
  inline
  half abs(half a)
  {
    float fa = __half2float(a);
    return __float2half(fabs(fa));
  }
  __host__
  static
  inline
  half abs(half a)
  {
    return THC_float2half(fabs(THC_half2float(a)));
  }

  __device__
  static
  inline
  half round(half a)
  {
    float fa = __half2float(a);
    return __float2half(roundf(fa));
  }
  __host__
  static
  inline
  half round(half a)
  {
    return THC_float2half(roundf(THC_half2float(a)));
  }

  __device__
  static
  inline
  half frac(half a)
  {
    float fa = __half2float(a);
    return __float2half(fa - truncf(fa));
  }
  __host__
  static
  inline
  half frac(half a)
  {
    float fa = THC_half2float(a);
    return THC_float2half(fa - floorf(fa));
  }

  __device__
  static
  inline
  half cinv(half a)
  {
    float fa = __half2float(a);
    return __float2half(1.0f / fa);
  }
  __host__
  static
  inline
  half cinv(half a)
  {
    return THC_float2half(1.0f / THC_half2float(a));
  }

  __device__
  static
  inline
  half add(half a, half b)
  {
    #if defined(__HIP_PLATFORM_HCC__)
      return a + b;
    #elif defined(CUDA_HALF_INSTRUCTIONS)
      return __hadd(a, b);
    #else
      float fa = __half2float(a);
      float fb = __half2float(b);
      return __float2half( fa + fb );
    #endif
  }
  __host__
  static
  inline
  half add(half a, half b)
  {
    #if defined(__HIP_PLATFORM_HCC__)
      return a + b;
    #else
      return THC_float2half(THC_half2float(a) + THC_half2float(b));
    #endif
  }

  __device__
  static
  inline
  half div(half a, half b)
  {
    #if defined(__HIP_PLATFORM_HCC__)
      return a / b;
    #else
      float fa = __half2float(a);
      float fb = __half2float(b);
      return __float2half( fa / fb );
    #endif
  }
  __host__
  static
  inline
  half dif(half a, half b)
  {
    return THC_float2half(THC_half2float(a) / THC_half2float(b));
  }

  __device__
  static
  inline
  half mul(half a, half b)
  {
    #if defined(__HIP_PLATFORM_HCC__)
      return a * b;
    #elif defined(CUDA_HALF_INSTRUCTIONS)
      return __hmul(a, b);
    #else
        float fa = __half2float(a);
        float fb = __half2float(b);
        return __float2half( fa * fb );
    #endif
  }
  __host__
  static
  inline
  half mul(half a, half b)
  {
    return THC_float2half(THC_half2float(a) * THC_half2float(b));
  }

  __device__
  static
  inline
  half sub(half a, half b)
  {
    #if defined(__HIP_PLATFORM_HCC__)
      return a - b;
    #elif defined(CUDA_HALF_INSTRUCTIONS)
      return __hsub(a, b);
    #else
        float fa = __half2float(a);
        float fb = __half2float(b);
        return __float2half( fa - fb );
    #endif
  }
  __host__
  static
  inline
  half sub(half a, half b)
  {
    return THC_float2half(THC_half2float(a) - THC_half2float(b));
  }

  __device__
  static
  inline
  half pow(half a, half b)
  {
    float fa = __half2float(a);
    float fb = __half2float(b);
    return __float2half(powf(fa, fb));
  }
  __host__
  static
  inline
  half pow(half a, half b)
  {
    return THC_float2half(powf(THC_half2float(a), THC_half2float(b)));
  }
};
#endif

template <>
struct THCNumerics<float> {
  static inline __host__ __device__ float min() { return -FLT_MAX; }
  static inline __host__ __device__ float max() { return FLT_MAX; }

  static inline __host__ __device__ bool lt(float a, float b) { return a < b; }
  static inline __host__ __device__ bool le(float a, float b) { return a <= b; }
  static inline __host__ __device__ bool gt(float a, float b) { return a > b; }
  static inline __host__ __device__ bool ge(float a, float b) { return a >= b; }
  static inline __host__ __device__ bool eq(float a, float b) { return a == b; }
  static inline __host__ __device__ bool ne(float a, float b) { return a != b; }

  static inline __host__ __device__  float exp  (float a) { return   expf(a); }
  static inline __host__ __device__  float log  (float a) { return   logf(a); }
  static inline __host__ __device__  float log1p(float a) { return log1pf(a); }
  static inline __host__ __device__  float cos  (float a) { return   cosf(a); }
  static inline __host__ __device__  float sin  (float a) { return   sinf(a); }
  static inline __host__ __device__  float sqrt (float a) { return  sqrtf(a); }
  static inline __host__ __device__  float rsqrt(float a) { return rsqrtf(a); }
  static inline __host__ __device__  float ceil (float a) { return  ceilf(a); }
  static inline __host__ __device__  float floor(float a) { return floorf(a); }
  static inline __host__ __device__  float trunc(float a) { return truncf(a); }
  static inline __host__ __device__  float neg  (float a) { return        -a; }
  static inline __host__ __device__  float acos (float a) { return  acosf(a); }
  static inline __host__ __device__  float cosh (float a) { return  coshf(a); }
  static inline __host__ __device__  float acosh(float a) { return acoshf(a); }
  static inline __host__ __device__  float asin (float a) { return  asinf(a); }
  static inline __host__ __device__  float sinh (float a) { return  sinhf(a); }
  static inline __host__ __device__  float asinh(float a) { return asinhf(a); }
  static inline __host__ __device__  float tan  (float a) { return   tanf(a); }
  static inline __host__ __device__  float atan (float a) { return  atanf(a); }
  static inline __host__ __device__  float tanh (float a) { return  tanhf(a); }
  static inline __host__ __device__  float abs  (float a) { return   fabs(a); }
  static inline __host__ __device__  float round(float a) { return roundf(a); }
  static inline __host__ __device__  float frac (float a) { return a - truncf(a); }
  static inline __host__ __device__  float cinv (float a) { return 1.0f / a; }
  static inline __host__ __device__  float add  (float a, float b) { return a + b; }
  static inline __host__ __device__  float div  (float a, float b) { return a / b; }
  static inline __host__ __device__  float mul  (float a, float b) { return a * b; }
  static inline __host__ __device__  float sub  (float a, float b) { return a - b; }
  static inline __host__ __device__  float pow  (float a, float b) { return powf(a, b); }
};

template <>
struct THCNumerics<double> {
  static inline __host__ __device__ double min() { return -DBL_MAX; }
  static inline __host__ __device__ double max() { return DBL_MAX; }

  static inline __host__ __device__ bool lt(double a, double b) { return a < b; }
  static inline __host__ __device__ bool le(double a, double b) { return a <= b; }
  static inline __host__ __device__ bool gt(double a, double b) { return a > b; }
  static inline __host__ __device__ bool ge(double a, double b) { return a >= b; }
  static inline __host__ __device__ bool eq(double a, double b) { return a == b; }
  static inline __host__ __device__ bool ne(double a, double b) { return a != b; }

  static inline __host__ __device__  double exp  (double a) { return   ::exp(a); }
  static inline __host__ __device__  double log  (double a) { return   ::log(a); }
  static inline __host__ __device__  double log1p(double a) { return ::log1p(a); }
  static inline __host__ __device__  double cos  (double a) { return   ::cos(a); }
  static inline __host__ __device__  double sin  (double a) { return   ::sin(a); }
  static inline __host__ __device__  double sqrt (double a) { return  ::sqrt(a); }
  static inline __host__ __device__  double rsqrt(double a) { return ::rsqrt(a); }
  static inline __host__ __device__  double ceil (double a) { return  ::ceil(a); }
  static inline __host__ __device__  double floor(double a) { return ::floor(a); }
  static inline __host__ __device__  double trunc(double a) { return ::trunc(a); }
  static inline __host__ __device__  double neg  (double a) { return         -a; }
  static inline __host__ __device__  double acos (double a) { return  ::acos(a); }
  static inline __host__ __device__  double cosh (double a) { return  ::cosh(a); }
  static inline __host__ __device__  double acosh(double a) { return ::acosh(a); }
  static inline __host__ __device__  double asin (double a) { return  ::asin(a); }
  static inline __host__ __device__  double sinh (double a) { return  ::sinh(a); }
  static inline __host__ __device__  double asinh(double a) { return ::asinh(a); }
  static inline __host__ __device__  double tan  (double a) { return   ::tan(a); }
  static inline __host__ __device__  double atan (double a) { return  ::atan(a); }
  static inline __host__ __device__  double tanh (double a) { return  ::tanh(a); }
  static inline __host__ __device__  double abs  (double a) { return  ::fabs(a); }
  static inline __host__ __device__  double round(double a) { return ::round(a); }
  static inline __host__ __device__  double frac (double a) { return a - ::trunc(a); }
  static inline __host__ __device__  double cinv (double a) { return 1.0 / a; }
  static inline __host__ __device__  double add  (double a, double b) { return a + b; }
  static inline __host__ __device__  double div  (double a, double b) { return a / b; }
  static inline __host__ __device__  double mul  (double a, double b) { return a * b; }
  static inline __host__ __device__  double sub  (double a, double b) { return a - b; }
  static inline __host__ __device__  double pow  (double a, double b) { return ::pow(a, b); }
};

/// `half` has some type conversion issues associated with it, since it
/// is a struct without a constructor/implicit conversion constructor.
/// We use this to convert scalar values to the given type that the
/// tensor expects.
template<typename In, typename Out>
struct ScalarConvert {
  __host__ __device__
  static
  Out to(const In& v) { return static_cast<Out>(v); }
};

#ifdef CUDA_HALF_TENSOR
  template<typename Out>
  struct ScalarConvert<half, Out> {
    __device__
    static
    Out to(half v)
    {
      #if defined(__HIP_PLATFORM_HCC__)
        return static_cast<Out>(v);
      #else
        return static_cast<Out>(__half2float(v));
      #endif
    }

    __host__
    static
    Out to(half v)
    {
      return static_cast<Out>(THC_half2float(v));
    }
  };

  template <typename In>
  struct ScalarConvert<In, half> {
    __device__
    static
    half to(In v)
    {
      #if defined(__HIP_PLATFORM_HCC__)
        return static_cast<half>(v);
      #else
        return __float2half(static_cast<float>(v));
      #endif
    }

    __host__
    static
    half to(In v)
    {
      return THC_float2half(static_cast<float>(v));
    }
  };

  template <>
  struct ScalarConvert<half, half> {
    __device__
    static
    half to(half v) { return v; }
  };
#endif

#endif // THC_NUMERICS_INC
