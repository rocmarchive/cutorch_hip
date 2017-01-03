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

template <class T, typename BinaryFunction>
__global__ void reduce_kernel(hipLaunchParm lp, THCState* state, T *g_idata, T *g_odata, unsigned int n, T val,  BinaryFunction f);  
/*
  THCDeviceState* device_state = state->deviceState;
  hc::accelerator accl = state->deviceState->get_current_accelerator();
  hc::accelerator_view accl_view = device_state->get_current_accelerator_view();
  long reduce_num_blocks = (n + (BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1))/BLOCK_SIZE;

  real* devPartialOut = hc::am_alloc(sizeof(T) * reduce_num_blocks, accl, 0);
 
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
}*/

template<class T, typename BinaryFunction>
T reduce(THCState* state, THCTensor* input, T init, BinaryFunction f) {
  T hRes, *dRes = NULL;
  real* dv_input_data = THCTensor_(data)(state, input);
  THCudaCheck(THCudaMalloc(state, (void**)&dRes, 1 * sizeof(T)));
  //reduce_kernel<T, BinaryFunction>(state, dv_input_data + input->storageOffset, dRes, THCTensor_(nElement)(state, input), init, f);
  //hc::am_copy(&hRes, dRes, 1*sizeof(T));
  THCudaFree(state, dRes);
  return hRes;
}



// Innerproduct
template <class T, typename BinaryFunction1, typename BinaryFunction2>
T inner_product(THCState* state, THCTensor* first1, THCTensor* first2, T init, BinaryFunction1 op1, BinaryFunction2 op2) {
  // Create temp contiguous array to store intermediate transform results  
  THCTensor* temp = THCTensor_(newContiguous)(state, first1);
  transform(state, first1, first2, temp, op2);
  return reduce<T>(state, temp, init, op1);
}


#endif
