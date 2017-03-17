#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCApply.cuh"

#include <hip/hip_runtime.h>

#include <GGL/grid_launch.hpp>

// Compute the offsets into the given tensors for a linear index. For the 't2'
// tensor, dimension 'dim' is skipped. The tensors are assumed to have the same
// size (with the exception of 't2' in dimension 'dim').
// This version uses a static number of dimensions.
template <typename IndexType, typename Real, int Dims>
struct IndexToScatterGatherOffsets {
  __device__
  static
  void compute(IndexType linearId,
               const int dim,
               //const TensorInfo<long, IndexType>& index,
               const long* indexData,
               const IndexType* indexSizes,
               const IndexType* indexStrides,
               int indexDims,
               IndexType* indexOffset,
               //const TensorInfo<Real, IndexType>& t1,
               const Real* t1Data,
               const IndexType* t1Sizes,
               const IndexType* t1Strides,
               int t1Dims,
               IndexType* t1Offset,
               //const TensorInfo<Real, IndexType>& t2,
               const Real* t2Data,
               const IndexType* t2Sizes,
               const IndexType* t2Strides,
               int t2Dims,
               IndexType* t2Offset)
  {
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

  __device__
  static void
  compute(IndexType linearId,
          const int dim,
          //const TensorInfo<long, IndexType>& index,
          const long* indexData,
          const IndexType* indexSizes,
          const IndexType* indexStrides,
          int indexDims,
          IndexType* indexOffset,
          //const TensorInfo<Real, IndexType>& t2,
          const Real* t2Data,
          const IndexType* t2Sizes,
          const IndexType* t2Strides,
          int t2Dims,
          IndexType* t2Offset)
  {
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
  __device__
  static
  void compute(IndexType linearId,
               const int dim,
               //const TensorInfo<long, IndexType>& index,
               const long* indexData,
               const IndexType* indexSizes,
               const IndexType* indexStrides,
               int indexDims,
               IndexType* indexOffset,
               //const TensorInfo<Real, IndexType>& t1,
               const Real* t1Data,
               const IndexType* t1Sizes,
               const IndexType* t1Strides,
               int t1Dims,
               IndexType* t1Offset,
               //const TensorInfo<Real, IndexType>& t2,
               const Real* t2Data,
               const IndexType* t2Sizes,
               const IndexType* t2Strides,
               int t2Dims,
               IndexType* t2Offset)
  {
    for (int d = indexDims - 1; d >= 0; d--) {
      IndexType curDimIndex = linearId % indexSizes[d];
      *indexOffset += curDimIndex * indexStrides[d];
      *t1Offset += curDimIndex * t1Strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2Strides[d];
      }
      linearId /= indexSizes[d];
    }
  }

  __device__
  static
  void compute(IndexType linearId,
               const int dim,
               //const TensorInfo<long, IndexType>& index,
               const long* indexData,
               const IndexType* indexSizes,
               const IndexType* indexStrides,
               int indexDims,
               IndexType* indexOffset,
               //const TensorInfo<Real, IndexType>& t2,
               const Real* t2Data,
               const IndexType* t2Sizes,
               const IndexType* t2Strides,
               int t2Dims,
               IndexType* t2Offset)
  {
    for (int d = indexDims - 1; d >= 0; d--) {
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
__global__
void THCudaTensor_gatherKernel(hipLaunchParm lp,
                               Real* tensorData,
                               IndexType* tensorSizes,
                               IndexType* tensorStrides,
                               int tensorDims, //TensorInfo<Real, IndexType> tensor,
                               Real* srcData,
                               IndexType* srcSizes,
                               IndexType* srcStrides,
                               int srcDims,//TensorInfo<Real, IndexType> src,
                               long* indexData,
                               IndexType* indexSizes,
                               IndexType* indexStrides,
                               int indexDims,//TensorInfo<long, IndexType> index,
                               const int dim,
                               const IndexType totalElements)
{
  for (IndexType linearId = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
       linearId < totalElements;
       linearId += hipGridDim_x * hipBlockDim_x) {
    IndexType tensorOffset = 0;
    IndexType srcOffset = 0;
    IndexType indexOffset = 0;

    IndexToScatterGatherOffsets<IndexType, Real, Dims>::compute(linearId,
                                                                dim,
                                                                indexData,
                                                                indexSizes,
                                                                indexStrides,
                                                                indexDims,
                                                                &indexOffset,
                                                                tensorData,
                                                                tensorSizes,
                                                                tensorStrides,
                                                                tensorDims,
                                                                &tensorOffset,
                                                                srcData,
                                                                srcSizes,
                                                                srcStrides,
                                                                srcDims,
                                                                &srcOffset);

    IndexType indexValue = (IndexType)indexData[indexOffset] - TH_INDEX_BASE;
    srcOffset += indexValue * srcStrides[dim];

    tensorData[tensorOffset] = srcData[srcOffset];
  }
}

template <typename IndexType, typename Real, int Dims>
__global__
void THCudaTensor_scatterKernel(hipLaunchParm lp,
                                //TensorInfo<Real, IndexType> tensor,
                                Real* tensorData,
                                IndexType* tensorSizes,
                                IndexType* tensorStrides,
                                int tensorDims,
                                //TensorInfo<Real, IndexType> src,
                                Real* srcData,
                                IndexType* srcSizes,
                                IndexType* srcStrides,
                                int srcDims,
                                //TensorInfo<long, IndexType> index,
                                long* indexData,
                                IndexType* indexSizes,
                                IndexType* indexStrides,
                                int indexDims,
                                int dim,
                                IndexType totalElements)
{
  for (IndexType linearId = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
       linearId < totalElements;
       linearId += hipGridDim_x * hipBlockDim_x) {
    IndexType tensorOffset = 0;
    IndexType srcOffset = 0;
    IndexType indexOffset = 0;

    IndexToScatterGatherOffsets<IndexType, Real, Dims>::compute(linearId,
                                                                dim,
                                                                indexData,
                                                                indexSizes,
                                                                indexStrides,
                                                                indexDims,
                                                                &indexOffset,
                                                                srcData,
                                                                srcSizes,
                                                                srcStrides,
                                                                srcDims,
                                                                &srcOffset,
                                                                tensorData,
                                                                tensorSizes,
                                                                tensorStrides,
                                                                tensorDims,
                                                                &tensorOffset);

    IndexType indexValue = (IndexType)indexData[indexOffset] - TH_INDEX_BASE;
    tensorOffset += indexValue * tensorStrides[dim];

    tensorData[tensorOffset] = srcData[srcOffset];
  }
}

template <typename IndexType, typename Real, int Dims>
__global__
inline
void THCudaTensor_scatterFillKernel(hipLaunchParm lp,
                                    //TensorInfo<Real, IndexType> tensor,
                                    Real* tensorData,
                                    IndexType* tensorSizes,
                                    IndexType* tensorStrides,
                                    int tensorDims,
                                    //TensorInfo<long, IndexType> index,
                                    long* indexData,
                                    IndexType* indexSizes,
                                    IndexType* indexStrides,
                                    int indexDims,
                                    Real value,
                                    int dim,
                                    IndexType totalElements)
{
  for (IndexType linearId = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
       linearId < totalElements;
       linearId += hipGridDim_x * hipBlockDim_x) {
    IndexType tensorOffset = 0;
    IndexType indexOffset = 0;

    IndexToScatterGatherOffsets<IndexType, Real, Dims>::compute(linearId,
                                                                dim,
                                                                indexData,
                                                                indexSizes,
                                                                indexStrides,
                                                                indexDims,
                                                                &indexOffset,
                                                                tensorData,
                                                                tensorSizes,
                                                                tensorStrides,
                                                                tensorDims,
                                                                &tensorOffset);

    IndexType indexValue = (IndexType)indexData[indexOffset] - TH_INDEX_BASE;
    tensorOffset += indexValue * tensorStrides[dim];

    tensorData[tensorOffset] = value;
  }
}

#include "generic/THCTensorScatterGather.cu"
#include "THCGenerateAllTypes.h"
