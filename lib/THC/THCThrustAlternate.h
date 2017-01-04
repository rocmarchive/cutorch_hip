#ifndef THC_THRUST_ALTERNATE_H
#define THC_THRUST_ALTERNATE_H


#include "THCTensor.h"
#include "THCGeneral.h"

#include "generic/THCThrustAlternate.h"
#include "THCGenerateAllTypes.h"
#include "hip/hip_runtime.h"

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

template <typename UnaryFunction>
__global__ void unary_transform_kernel(hipLaunchParm lp, float*& first, long firstOffset,
                     float*& result, long resultOffset, long size, UnaryFunction f) {
  if (size == 0) {
    return;
  }

  long index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

  if (index < size) {
      result[resultOffset + index] = f(first[firstOffset + index]);
  }
}

template <typename BinaryFunction>
__global__ void binary_transform_kernel(hipLaunchParm lp, float*& first1, long first1Offset,
                      float*& first2, long first2Offset,
                      float*& result, long resultOffset, long size,  BinaryFunction f) {
  if (size == 0) {
    return;
  }

  long index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

  if (index < size) {
    result[resultOffset + index] = (float) f(first1[first1Offset + index], first2[first2Offset + index]);
  }
}


// Define transform functions

// Unary transforms
template <typename UnaryFunction>
void transform(THCState* state, THCudaTensor* first, THCudaTensor* result, UnaryFunction op) {
  float* avData_first = THCudaTensor_data(state, first); 
  float* avData_result = THCudaTensor_data(state, result);
  long size = THCudaTensor_nElement(state, result);
  dim3 grid((size + 255)/256, 1, 1);
  dim3 block(256, 1, 1);
  hipLaunchKernel(HIP_KERNEL_NAME(unary_transform_kernel<UnaryFunction>), grid, block, 0, THCState_getCurrentStream(state), avData_first, first->storageOffset, avData_result, result->storageOffset, size, op);
}

// Binary transform
template <typename BinaryFunction>
void transform(THCState* state, THCudaTensor* first1, THCudaTensor* first2, THCudaTensor* result, BinaryFunction op) {
  float* avData_first1 = THCudaTensor_data(state, first1);
  float* avData_first2 = THCudaTensor_data(state, first2);
  float* avData_result = THCudaTensor_data(state, result);
  long size = THCudaTensor_nElement(state, result);
  dim3 grid((size + 255)/256, 1, 1);
  dim3 block(256, 1, 1);
  hipLaunchKernel(HIP_KERNEL_NAME(binary_transform_kernel<BinaryFunction>), grid, block, 0, THCState_getCurrentStream(state), avData_first1, first1->storageOffset, avData_first2, first2->storageOffset,
                   avData_result, result->storageOffset, size, op);
}




// Reduce routines
#define BLOCK_SIZE 256
template <typename BinaryFunction>
__global__ void reduce_kernel_pass1(hipLaunchParm lp, float *g_idata, float *devPartialOut, unsigned int n, unsigned int reduce_num_blocks, float val,  BinaryFunction f) { 
    __shared__ float buf_tmp[BLOCK_SIZE];
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int block_idx = idx / BLOCK_SIZE;
    int thread_in_block_idx = idx % BLOCK_SIZE;
    int eidx = idx;
    float res = val;

    while(eidx < n)
    {
      res = f(res, g_idata[eidx]);
      eidx += reduce_num_blocks * BLOCK_SIZE;
    }
    buf_tmp[thread_in_block_idx] = res;
    __syncthreads();

    // Seqential part
    if (hipThreadIdx_x == 0)
    {
      res = val;
      for (uint i = 0; i < BLOCK_SIZE; i++)
      {
        res = f(res, buf_tmp[i]);
      }
      devPartialOut[block_idx] = res;
    }
}

template <typename BinaryFunction>
__global__ void reduce_kernel_pass2(hipLaunchParm lp, float *devPartialOut, float *g_odata, unsigned int residualSize, float val,  BinaryFunction f) {
    float res = val;
    for (uint i = 0; i < residualSize; i++)
    {
      res = f(res, devPartialOut[i]);
    }
    g_odata[0] = res;
}

template<typename BinaryFunction>
float reduce(THCState* state, THCudaTensor* input, float init, BinaryFunction f) {
  float hRes, *dRes = NULL;
  float* dv_input_data = THCudaTensor_data(state, input);
  THCudaCheck(THCudaMalloc(state, (void**)&dRes, 1 * sizeof(float)));
  long n = THCudaTensor_nElement(state, input);
  long reduce_num_blocks = (n + (BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1))/BLOCK_SIZE;
  dim3 grid1(reduce_num_blocks, 1, 1);
  dim3 block1(BLOCK_SIZE, 1, 1);
  float* devPartialOut = NULL;
  THCudaCheck(THCudaMalloc(state, devPartialOut, sizeof(float) * reduce_num_blocks));
  hipLaunchKernel(HIP_KERNEL_NAME(reduce_kernel_pass1), grid1, block1, 0, THCState_getCurrentStream(state), dv_input_data + input->storageOffset, devPartialOut, n, reduce_num_blocks, init, f);
  dim3 grid2(1, 1, 1);
  dim3 block2(1, 1, 1);
  hipLaunchKernel(HIP_KERNEL_NAME(reduce_kernel_pass2), grid2, block2, 0, THCState_getCurrentStream(state), devPartialOut, dRes, reduce_num_blocks, init, f);
  THCudaCheck(hipMemcpy(&dRes[0], &hRes, 1*sizeof(float), hipMemcpyDeviceToHost));
  THCudaFree(state, dRes);
  return hRes;
}

// Innerproduct
template <typename BinaryFunction1, typename BinaryFunction2>
float inner_product(THCState* state, THCudaTensor* first1, THCudaTensor* first2, float init, BinaryFunction1 op1, BinaryFunction2 op2) {
  // Create temp contiguous array to store intermediate transform results  
  THCudaTensor* temp = THCudaTensor_newContiguous(state, first1);
  transform(state, first1, first2, temp, op2);
  return reduce(state, temp, init, op1);
}

}
#endif

