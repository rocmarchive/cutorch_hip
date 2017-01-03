#ifndef THC_THRUST_ALTERNATE_H
#define THC_THRUST_ALTERNATE_H


#include "THCTensor.h"
#include "THCGeneral.h"

#include "generic/THCThrustAlternate.h"
#include "THCGenerateAllTypes.h"



namespace thrust_alternate {
// Common Operators definition
template <class T> struct identity {
__device__ __host__  T operator() (const T& x) const  {return x;}
};

template <class T> struct sum {
 __device__ __host__  T operator() (const T& x, const T& y) const  {return x+y;}
};

template <class T> struct multiply {
 __device__ __host__  T operator() (const T& x, const T& y) const  {return x * y;}
};

template <class T> struct divide {
 __device__ __host__  T operator() (const T& x, const T& y) const  {return x/y;}
};

template <class T> struct maxval {
 __device__ __host__  T operator() (const T& x, const T& y) const  {
     if(x > y) return x; else return y;
  }
};

template <class T> struct minval {
 __device__ __host__  T operator() (const T& x, const T& y) const  {
     if(x < y) return x; else return y;
  }
};

template <class T> struct is_equal_to {
 __device__ __host__  bool operator() (const T& x, const T& y) const  {return x==y;}
};

template <class T> struct is_not_equal_to {
 __device__ __host__  bool operator() (const T& x, const T& y) const  {return x!=y;}
};


// pair holds two objects of arbitrary type.
template<class _T1, class _T2>
struct Pair
{
  typedef _T1 first_type;
  typedef _T2 second_type;

  _T1 first;
  _T2 second;

  __device__ __host__  Pair() : first(), second() { }
  __device__ __host__  Pair(const _T1& __a, const _T2& __b) : first(__a), second(__b) { }
};

template<class _T1, class _T2>
 __device__ __host__ Pair<_T1, _T2> Make_Pair(_T1 src, _T2 col)  {
  return (Pair<_T1, _T2>(src, col));
}





struct addvalue_functor {
  float value;
 __device__ __host__  addvalue_functor(float value_)  : value(value_) {}
 __device__ __host__  addvalue_functor(const addvalue_functor& other) 
    : value(other.value) {}
 __device__ __host__  addvalue_functor& operator = (const addvalue_functor&other)  {
    value = other.value;
    return *this;
  }
 __device__ __host__  float operator()(const float& x) const  {
    return (x + value);
  }
};

struct mse_functor {
 __device__ __host__  mse_functor()  {}
 __device__ __host__  float operator()(const float& x, const float& y) const  {
    float z = x - y;
    return z * z;
  }
};

struct mulvalue_functor {
  float value;
 __device__ __host__  mulvalue_functor(float value_) : value(value_) {}
 __device__ __host__  mulvalue_functor(const mulvalue_functor& other) 
    : value(other.value) {}
 __device__ __host__  mulvalue_functor& operator = (const mulvalue_functor&other)  {
    value = other.value;
    return *this;
  }
 __device__ __host__  float operator()(const float& x) const  {
    return (x * value);
  }
};

struct divvalue_functor {
  float value;
 __device__ __host__  divvalue_functor(float value_) : value(value_) {}
 __device__ __host__  divvalue_functor(const divvalue_functor& other) 
    : value(other.value) {}
 __device__ __host__  divvalue_functor& operator = (const divvalue_functor&other)  {
    value = other.value;
    return *this;
  }
  float operator()(const float& x) const  {
    return (x / value);
  }
};

struct pow_functor {
  float value;
 __device__ __host__  pow_functor(float value_)  : value(value_) {}
 __device__ __host__  pow_functor(const pow_functor& other)  : value(other.value) {}
 __device__ __host__  pow_functor& operator = (const pow_functor&other)  {
    value = other.value;
    return *this;
  }
 __device__ __host__  float operator()(const float& x) const  {
    return std::pow(x, value);
  }
};

struct tpow_functor {
  float value;
 __device__ __host__  tpow_functor(float value_) : value(value_) {}
 __device__ __host__  tpow_functor(const tpow_functor& other) : value(other.value) {}
 __device__ __host__  tpow_functor& operator = (const tpow_functor&other)  {
    value = other.value;
    return *this;
  }
 __device__ __host__  float operator()(const float& x) const  {
    return std::pow(value, x);
  }
};


struct cpow_functor {
 __device__ __host__  cpow_functor()  {}
 __device__ __host__  float operator()(const float& a, const float& b) const  {
    return std::pow(a, b);
  }
};


struct atan2_functor {
 __device__ __host__  atan2_functor()  {}
 __device__ __host__  float operator()(const float& x, const float& y) const  {
    return atan2f(x, y);
  }
};

struct clamp_functor {
  float min_value;
  float max_value;
 __device__ __host__  clamp_functor(float min_value_, float max_value_) : min_value(min_value_), max_value(max_value_) {}
 __device__ __host__  clamp_functor(const clamp_functor& other) : min_value(other.min_value), max_value(other.max_value) {}
 __device__ __host__  clamp_functor& operator = (const clamp_functor&other)  {
    min_value = other.min_value;
    max_value = other.max_value;
    return *this;
  }
 __device__ __host__  float operator()(const float& x) const  {
    if (x < min_value) {
      return min_value;
    }
    
    if (x > max_value) {
      return max_value;
    }
    
    return x;
  }
};

struct sign_functor {
 __device__ __host__  sign_functor()  {}
 __device__ __host__  float operator()(const float &v) const  {
    return (v > 0) - (v < 0);
  }
};

struct dist_functor {
  float exponent;
 __device__ __host__  dist_functor(float exponent_)  : exponent(exponent_) {}
 __device__ __host__  dist_functor(const dist_functor& other)  : exponent(other.exponent) {}
 __device__ __host__  dist_functor& operator = (const dist_functor&other)  {
    exponent = other.exponent;
    return *this;
  }
 __device__ __host__  float operator()(const float& x, const float& y) const  {
    return std::pow(std::fabs(x - y), exponent);
  }
};

struct norm_functor {
  float exponent;
 __device__ __host__  norm_functor(float exponent_) : exponent(exponent_) {}
 __device__ __host__  norm_functor(const norm_functor& other) : exponent(other.exponent) {}
 __device__ __host__  norm_functor& operator = (const norm_functor&other)  {
    exponent = other.exponent;
    return *this;
  }
 __device__ __host__  float operator()(const float& x) const  {
    return std::pow(std::fabs(x), exponent);
  }
};

struct partial_not_equal_functor {
  float rhs;
 __device__ __host__  partial_not_equal_functor(float rhs)  : rhs(rhs) {}
 __device__ __host__  partial_not_equal_functor(const partial_not_equal_functor& other)  : rhs(other.rhs) {}
 __device__ __host__  partial_not_equal_functor& operator = (const partial_not_equal_functor&other)  {
    rhs = other.rhs;
    return *this;
  }
 __device__ __host__  bool operator()(const float &lhs) const  {
    return lhs != rhs;
  }
};

struct mse_updateGradInput_functor {
  float norm;
 __device__ __host__  mse_updateGradInput_functor(float norm_)  : norm(norm_) {}
 __device__ __host__  mse_updateGradInput_functor(const mse_updateGradInput_functor &other)  : norm(other.norm) {}
 __device__ __host__  mse_updateGradInput_functor& operator = (const mse_updateGradInput_functor&other)  {
   norm = other.norm;
    return *this;
  }
 __device__ __host__  float operator()(const float& x, const float& y) const  {
    return norm * (x - y);
  }
};

struct binary_abs_functor {
 __device__ __host__  binary_abs_functor()  {}
 __device__ __host__  float operator()(const float& x, const float& y) const  {
    float z = x - y;
    return z >= 0 ? z : -z;
  }
};

struct abs_updateGradInput_functor {
  float norm;
 __device__ __host__  abs_updateGradInput_functor(float norm_) : norm(norm_) {}
 __device__ __host__  abs_updateGradInput_functor(const abs_updateGradInput_functor& other) : norm(other.norm) {}
 __device__ __host__  abs_updateGradInput_functor& operator = (const abs_updateGradInput_functor&other)  {
   norm = other.norm;
    return *this;
  }
 __device__ __host__  float operator()(const float& x, const float& y) const  {
    return (x - y) >= 0 ? norm : -norm;
  }
};

struct kl_functor {
 __device__ __host__ kl_functor()  {}
 __device__ __host__  float operator()(const float& x, const float& y) const  {
    return y > 0 ? y * (std::log(y) - x) : 0;
  }
};

struct kl_updateGradInput_functor {
  float norm;
 __device__ __host__ kl_updateGradInput_functor(float norm_)  : norm(norm_) {}
 __device__ __host__ kl_updateGradInput_functor(const kl_updateGradInput_functor& other)  : norm(other.norm) {}
 __device__ __host__ kl_updateGradInput_functor& operator = (const kl_updateGradInput_functor&other)  {
   norm = other.norm;
    return *this;
  }
 __device__ __host__ float operator()(const float& x, const float& y) const  {
    return y > 0 ? norm * (-y) : 0;
  }
};

} 
#endif

