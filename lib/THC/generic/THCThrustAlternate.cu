#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCThrustAlternate.cu"
#else
#include "hip/hip_runtime.h"

template <typename UnaryFunction>
__global__ void unary_transform_kernel(hipLaunchParm lp, real*& first, long firstOffset,
                     real*& result, long resultOffset, long size, UnaryFunction f) {
  if (size == 0) {
    return;
  }

  long index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

  if (index < size) {
      result[resultOffset + index] = f(first[firstOffset + index]);
  }
}

template <typename BinaryFunction>
__global__ void binary_transform_kernel(hipLaunchParm lp, real*& first1, long first1Offset,
                      real*& first2, long first2Offset,
                      real*& result, long resultOffset, long size,  BinaryFunction f) {
  if (size == 0) {
    return;
  }

  long index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

  if (index < size) {
    result[resultOffset + index] = (real) f(first1[first1Offset + index], first2[first2Offset + index]);
  }
}


// Define transform functions

// Unary transforms
template <typename UnaryFunction>
void transform(THCState* state, THCTensor* first, THCTensor* result, UnaryFunction op) {
  real* avData_first = THCTensor_(data)(state, first); 
  real* avData_result = THCTensor_(data)(state, result);
  long size = THCTensor_(nElement)(state, result);
  dim3 grid((size + 255)/256, 1, 1);
  dim3 block(256, 1, 1);
  hipLaunchKernel(HIP_KERNEL_NAME(unary_transform_kernel<UnaryFunction>), grid, block, 0, THCState_getCurrentStream(state), avData_first, first->storageOffset, avData_result, result->storageOffset, size, op);
}

// Binary transform
template <typename BinaryFunction>
void transform(THCState* state, THCTensor* first1, THCTensor* first2, THCTensor* result, BinaryFunction op) {
  real* avData_first1 = THCTensor_(data)(state, first1);
  real* avData_first2 = THCTensor_(data)(state, first2);
  real* avData_result = THCTensor_(data)(state, result);
  long size = THCTensor_(nElement)(state, result);
  dim3 grid((size + 255)/256, 1, 1);
  dim3 block(256, 1, 1);
  hipLaunchKernel(HIP_KERNEL_NAME(binary_transform_kernel<BinaryFunction>), grid, block, 0, THCState_getCurrentStream(state), avData_first1, first1->storageOffset, avData_first2, first2->storageOffset,
                   avData_result, result->storageOffset, size, op);
}




// Reduce routines
#define BLOCK_SIZE 256
template <typename BinaryFunction>
__global__ void reduce_kernel_pass1(hipLaunchParm lp, THCState* state, real *g_idata, real *devPartialOut, unsigned int n, unsigned int reduce_num_blocks, real val,  BinaryFunction f) { 
    __shared__ real buf_tmp[BLOCK_SIZE];
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int block_idx = idx / BLOCK_SIZE;
    int thread_in_block_idx = idx % BLOCK_SIZE;
    int eidx = idx;
    real res = val;

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
__global__ void reduce_kernel_pass2(hipLaunchParm lp, real *devPartialOut, real *g_odata, unsigned int residualSize, real val,  BinaryFunction f) {
    real res = val;
    for (uint i = 0; i < residualSize; i++)
    {
      res = f(res, devPartialOut[i]);
    }
    g_odata[0] = res;
}

template<typename BinaryFunction>
real reduce(THCState* state, THCTensor* input, real init, BinaryFunction f) {
  real hRes, *dRes = NULL;
  real* dv_input_data = THCTensor_(data)(state, input);
  THCudaCheck(THCudaMalloc(state, (void**)&dRes, 1 * sizeof(real)));
  long n = THCTensor_(nElement)(state, input);
  long reduce_num_blocks = (n + (BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1))/BLOCK_SIZE;
  dim3 grid(reduce_num_blocks, 1, 1);
  dim3 block(BLOCK_SIZE, 1, 1);
  real* devPartialOut = NULL;
  THCudaCheck(THCudaMalloc(state, (void**)devPartialOut, sizeof(real) * reduce_num_blocks));
  reduce_kernel_pass1<BinaryFunction>(state, dv_input_data + input->storageOffset, dRes, n, reduce_num_blocks, init, f);
  reduce_kernel_pass2<BinaryFunction>(dRes, reduce_num_blocks, init, f);
  THCudaCheck(hipMemcpy(&dRes[0], &hRes, 1*sizeof(real), hipMemcpyDeviceToHost));
  THCudaFree(state, dRes);
  return hRes;
}



// Innerproduct
template <class T, typename BinaryFunction1, typename BinaryFunction2>
T inner_product(THCState* state, THCTensor* first1, THCTensor* first2, T init, BinaryFunction1 op1, BinaryFunction2 op2) {
  // Create temp contiguous array to store intermediate transform results  
  THCTensor* temp = THCTensor_(newContiguous)(state, first1);
  transform(state, first1, first2, temp, op2);
  return reduce(state, temp, init, op1);
}


#endif
