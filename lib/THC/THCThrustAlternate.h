#ifndef THHC_FUNCTOR_OPNS_H
#define THHC_FUNCTOR_OPNS_H


#include "THCTensor.h"
#include "THCGeneral.h"
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
  __device__ __host__  ~Pair() { }
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
/*
// Define transform functions

// Unary transforms
template <typename UnaryFunction>
void transform(THCState* state, THCTensor* first, THCTensor* result, UnaryFunction op) {
  auto avData_first = first->get_device_data(state);
  auto avData_result = result->get_device_data(state);
  long size = THCTensor_nElement(state, result);
  unary_transform(state, avData_first, first->storageOffset, avData_result, result->storageOffset, size, op);
}

// Binary transform
template <typename BinaryFunction>
void transform(THCState* state, THCTensor* first1, THCTensor* first2, THCTensor* result, BinaryFunction op) {
  auto avData_first1 = first1->get_device_data(state);
  auto avData_first2 = first2->get_device_data(state);
  auto avData_result = result->get_device_data(state);
  long size = THCTensor_nElement(state, result);
  binary_transform(state, avData_first1, first1->storageOffset, avData_first2, first2->storageOffset,
                   avData_result, result->storageOffset, size, op);
}

template <typename BinaryFunction>
void binary_transform(THCState* state, float*& first1, long first1Offset,
                      float*& first2, long first2Offset,
                      float*& result, long resultOffset, long size,  BinaryFunction f) {
  if (size == 0) {
    return;
  }

  unsigned grdSz = (size + 255) & ~(255);
  hc::extent<1> grdExt(grdSz);
  hc::tiled_extent<1> t_ext = grdExt.tile(256);
  THCDeviceState* device_state = state->deviceState;
  hc::accelerator_view accl_view = device_state->get_current_accelerator_view();
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<1>& tidx)  {
    long index = tidx.global[0];

    if (index < size) {
      result[resultOffset + index] = (float) f(first1[first1Offset + index], first2[first2Offset + index]);
    }
  }).wait();
  return;
}

template <typename UnaryFunction>
void unary_transform(THCState* state, float*& first, long firstOffset,
                     float*& result, long resultOffset, long size, UnaryFunction f) {
  if (size == 0) {
    return;
  }

  unsigned grdSz = (size + 256) - (size % 256);
  hc::extent<1> grdExt(grdSz);
  hc::tiled_extent<1> t_ext = grdExt.tile(256);
  THCDeviceState* device_state = state->deviceState;
  hc::accelerator_view accl_view = device_state->get_current_accelerator_view();
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<1>& tidx)  {
    long index = tidx.global[0];

    if (index < size) {
      result[resultOffset + index] = f(first[firstOffset + index]);
    }
  }).wait();
  return;
}


// Reduce routines

#define BLOCK_SIZE 256
template <class T, typename BinaryFunction>
inline void reduce_operation(THCState* state, T *g_idata, T *g_odata, unsigned int n, T val,  BinaryFunction f) {
  
  THCDeviceState* device_state = state->deviceState;
  hc::accelerator accl = state->deviceState->get_current_accelerator();
  hc::accelerator_view accl_view = device_state->get_current_accelerator_view();
  long reduce_num_blocks = (n + (BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1))/BLOCK_SIZE;

  float* devPartialOut = hc::am_alloc(sizeof(T) * reduce_num_blocks, accl, 0);
 
  hc::extent<1> grdExt(reduce_num_blocks * BLOCK_SIZE);
  hc::tiled_extent<1> t_ext = grdExt.tile(BLOCK_SIZE);
  hc::parallel_for_each(accl_view, t_ext, [=] (hc::tiled_index<1>& tidx) 
  {
    tile_static T buf_tmp[BLOCK_SIZE];
    int idx = tidx.global[0];
    int block_idx = idx / BLOCK_SIZE;
    int thread_in_block_idx = idx % BLOCK_SIZE;
    int eidx = idx;
    T res = val;

    while(eidx < n)
    {
      res = f(res, g_idata[eidx]);
      eidx += reduce_num_blocks * BLOCK_SIZE;
    }
    buf_tmp[thread_in_block_idx] = res;
    tidx.barrier.wait();

    // Seqential part
    if (tidx.local[0] == 0)
    {
      res = val;
      for (uint i = 0; i < BLOCK_SIZE; i++)
      {
        res = f(res, buf_tmp[i]);
      }
      devPartialOut[block_idx] = res;
    }
  }).wait();

  hc::extent<1> grdExt1(1);
  hc::tiled_extent<1> t_ext1 = grdExt1.tile(1);
  hc::parallel_for_each(accl_view, t_ext1, [=] (hc::tiled_index<1>& tidx) 
  {
    T res = val;
    for (uint i = 0; i < reduce_num_blocks; i++)
    {
      res = f(res, devPartialOut[i]);
    }
    g_odata[0] = res;
  }).wait(); 

  hc::am_free(devPartialOut); 
}

template<class T, typename BinaryFunction>
inline T reduce(THCState* state, THCTensor* input, T init, BinaryFunction f) {
  T hRes, *dRes = NULL;
  auto dv_input_data = input->get_device_data(state);
  THCCheck(THCMalloc(state, (void**)&dRes, 1 * sizeof(T)));
  reduce_operation(state, dv_input_data + input->storageOffset, dRes, THCTensor_nElement(state, input), init, f);
  hc::am_copy(&hRes, dRes, 1*sizeof(T));
  THCFree(state, dRes);
  return hRes;
}



// Innerproduct
template <class T, typename BinaryFunction1, typename BinaryFunction2>
inline T inner_product(THCState* state, THCTensor* first1, THCTensor* first2, T init, BinaryFunction1 op1, BinaryFunction2 op2) {
  // Create temp contiguous array to store intermediate transform results  
  THCTensor* temp = THCTensor_newContiguous(state, first1);
  transform(state, first1, first2, temp, op2);
  return reduce<T>(state, temp, init, op1);
}*/
} 
#endif

