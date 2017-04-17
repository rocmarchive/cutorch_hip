#include "hip/hip_runtime.h"
#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCApply.cuh"

// Compute the offsets into the given tensors for a linear index. For the 't2'
// tensor, dimension 'dim' is skipped. The tensors are assumed to have the same
// size (with the exception of 't2' in dimension 'dim').
// This version uses a static number of dimensions.
template <typename IndexType, typename Real, int Dims>
struct IndexToScatterGatherOffsets {
  static __device__ void compute(
      IndexType linearId, const int dim,
      IndexType* indexSizes, IndexType* indexStrides, IndexType* indexOffset,
      IndexType* t1Strides,  IndexType* t1Offset,
      IndexType* t2Strides, IndexType* t2Offset) {
    for (int d = Dims - 1; d >= 0; d--) {
      IndexType curDimIndex = linearId % indexSizes[d];
      *indexOffset += curDimIndex * indexStrides[d];
      *t1Offset += curDimIndex * t1Strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2Strides[d];
      }
      linearId /= indexSizes[d];
    }
  }

  static __device__ void compute(
      IndexType linearId, const int dim,
      IndexType* indexSizes, IndexType* indexStrides, IndexType* indexOffset,
      IndexType* t2Strides, IndexType* t2Offset) {
    for (int d = Dims - 1; d >= 0; d--) {
      IndexType curDimIndex = linearId % indexSizes[d];
      *indexOffset += curDimIndex * indexStrides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2Strides[d];
      }
      linearId /= indexSizes[d];
    }
  }
};

// Same as above but using a dynamic number of dimensions.
template <typename IndexType, typename Real>
struct IndexToScatterGatherOffsets<IndexType, Real, -1> {
  static __device__ void compute(
      IndexType linearId, const int dim,
      IndexType* indexSizes, IndexType* indexStrides, IndexType* indexOffset,
      IndexType* t1Strides, IndexType* t1Offset,
      IndexType* t2Strides, IndexType* t2Offset) {
    for (int d = dim - 1; d >= 0; d--) {
      IndexType curDimIndex = linearId % indexSizes[d];
      *indexOffset += curDimIndex * indexStrides[d];
      *t1Offset += curDimIndex * t1Strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2Strides[d];
      }
      linearId /= indexSizes[d];
    }
  }

  static __device__ void compute(
      IndexType linearId, const int dim,
      IndexType *indexSizes, IndexType *indexStrides, IndexType* indexOffset,
      IndexType *t2Strides, IndexType* t2Offset) {
    for (int d = dim - 1; d >= 0; d--) {
      IndexType curDimIndex = linearId % indexSizes[d];
      *indexOffset += curDimIndex * indexStrides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2Strides[d];
      }
      linearId /= indexSizes[d];
    }
  }
};

template <typename IndexType, typename Real, int Dims>
__global__ void THCudaTensor_gatherKernel(
    Real* tensorData, IndexType* tensorStrides,
    Real* srcData, IndexType* srcStrides,
    long* indexData, IndexType* indexSizes, IndexType* indexStrides, 
    const int dim,
    const IndexType totalElements) {
  for (IndexType linearId = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
       linearId < totalElements;
       linearId += hipGridDim_x * hipBlockDim_x) {
    IndexType tensorOffset = 0;
    IndexType srcOffset = 0;
    IndexType indexOffset = 0;

    IndexToScatterGatherOffsets<IndexType, Real, Dims>::compute(linearId, dim,
                                                          indexSizes, indexStrides, &indexOffset,
                                                          tensorStrides, &tensorOffset,
                                                          srcStrides, &srcOffset);

    IndexType indexValue = (IndexType)indexData[indexOffset] - TH_INDEX_BASE;
    srcOffset += indexValue * srcStrides[dim];

    tensorData[tensorOffset] = srcData[srcOffset];
  }
}

template <typename IndexType, typename Real, int Dims>
__global__ void THCudaTensor_scatterKernel(
    Real* tensorData, IndexType* tensorStrides,
    Real* srcData, IndexType* srcStrides,
    long* indexData, IndexType* indexSizes, IndexType* indexStrides, 
    const int dim,
    const IndexType totalElements) {
  for (IndexType linearId = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
       linearId < totalElements;
       linearId += hipGridDim_x * hipBlockDim_x) {
    IndexType tensorOffset = 0;
    IndexType srcOffset = 0;
    IndexType indexOffset = 0;

    IndexToScatterGatherOffsets<IndexType, Real, Dims>::compute(linearId, dim,
                                                          indexSizes, indexStrides, &indexOffset,
                                                          srcStrides, &srcOffset,
                                                          tensorStrides, &tensorOffset);

    IndexType indexValue = (IndexType)indexData[indexOffset] - TH_INDEX_BASE;
    tensorOffset += indexValue * tensorStrides[dim];

    tensorData[tensorOffset] = srcData[srcOffset];
  }
}

template <typename IndexType, typename Real, int Dims>
__global__ void THCudaTensor_scatterFillKernel(
    Real* tensorData, IndexType* tensorStrides,
    long* indexData, IndexType* indexSizes, IndexType* indexStrides, 
    Real value,
    const int dim,
    const IndexType totalElements) {
  for (IndexType linearId = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
       linearId < totalElements;
       linearId += hipGridDim_x * hipBlockDim_x) {
    IndexType tensorOffset = 0;
    IndexType indexOffset = 0;

    IndexToScatterGatherOffsets<IndexType, Real, Dims>::compute(linearId, dim,
                                                          indexSizes, indexStrides, &indexOffset,
                                                          tensorStrides, &tensorOffset);

    IndexType indexValue = (IndexType)indexData[indexOffset] - TH_INDEX_BASE;
    tensorOffset += indexValue * tensorStrides[dim];

    tensorData[tensorOffset] = value;
  }
}

#include "generic/THCTensorScatterGather.cu"
#include "THCGenerateAllTypes.h"
