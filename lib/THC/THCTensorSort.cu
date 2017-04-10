#include "THCTensorSort.cuh"

// Returns 2^(ceil(lg(n)) from Stanford bit twiddling hacks
unsigned long nextHighestPowerOf2(unsigned long n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
#ifndef _MSC_VER
  n |= n >> 32;
#endif
  n++;

  return n;
}

void THCudaLongTensor_fillSliceWithIndex(THCState* state,
                                         THCudaLongTensor* t,
                                         int dim) {
  long dims = THCudaLongTensor_nDimension(state, t);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);

  ptrdiff_t inElements = THCudaLongTensor_nElement(state, t);
  long sliceSize = THCudaLongTensor_size(state, t, dim);
  ptrdiff_t numSlices = inElements / sliceSize;

  dim3 grid;
  if (!THC_getGridFromTiles(numSlices, grid)) {
    THError("Slice to fill with indices is too large");
  }

  long maxThreads =
    THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
  long numThreads = sliceSize;
  if (numThreads > maxThreads) {
    numThreads = maxThreads;
  }

  dim3 block(numThreads);

#define FILL_INDEX(T, DIM)                                       \
  hipLaunchKernel(HIP_KERNEL_NAME(fillSliceWithIndex<T, DIM>),                                     \
      grid, block, 0, THCState_getCurrentStream(state),     \
      infoData, infoSizes, infoStrides, infoDims, numSlices, sliceSize, info.strides[collapseDim])

#ifdef CUDA_PATH
  if (TensorUtils<THCudaLongTensor>::canUse32BitIndexMath(state, t)) {
    TensorInfo<long, unsigned int> info =
      getTensorInfo<THCudaLongTensor, unsigned int>(state, t);
    long* infoData = info.data;
    unsigned int* infoSizes = info.dSizes;
    unsigned int* infoStrides = info.dStrides;
    int infoDims = info.dims;
    info.reduceDim(dim);
    int collapseDim = info.collapseDims(dim);
    if (info.isContiguous()) {
      FILL_INDEX(unsigned int, -2);
    } else {
      if (info.dims == 1) {
        FILL_INDEX(unsigned int, 1);
      } else if (info.dims == 2) {
        FILL_INDEX(unsigned int, 2);
      } else {
        FILL_INDEX(unsigned int, -1);
      }
    }
  } else {
    TensorInfo<long, unsigned long> info =
      getTensorInfo<THCudaLongTensor, unsigned long>(state, t);
    info.reduceDim(dim);
    int collapseDim = info.collapseDims(dim);
    long* infoData = info.data;
    unsigned long* infoSizes = info.dSizes;
    unsigned long* infoStrides = info.dStrides;
    int infoDims = info.dims;

    // catch-all implementation
    FILL_INDEX(unsigned long, -1);
  }
#endif

#undef FILL_INDEX

  THCudaCheck(hipGetLastError());
}
