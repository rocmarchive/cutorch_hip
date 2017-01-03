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

#endif

