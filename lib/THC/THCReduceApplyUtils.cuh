#ifndef THC_REDUCE_APPLY_UTILS_INC
#define THC_REDUCE_APPLY_UTILS_INC

#include "THCGeneral.h"
#include "THCTensor.h"
#include "THCDeviceUtils.cuh"
#include "THCTensorInfo.cuh"

#include <hip/hip_runtime.h>

#include <assert.h>

// Enum that indicates whether tensor arguments are read/write or
// read-only
enum TensorArgType { ReadWrite, ReadOnly };

template <typename IndexType>
__device__ __forceinline__
static
IndexType getLinearBlockId() {
  return hipBlockIdx_z * hipGridDim_y * hipGridDim_x +
    hipBlockIdx_y * hipGridDim_x +
    hipBlockIdx_x;
}

// Block-wide reduction in shared memory helper; only hipThreadIdx_x == 0 will
// return the reduced value
// template <typename T, typename dressedT, typename ReduceOp>
template <typename T, typename ReduceOp>
__device__
static
inline
// T reduceBlock(dressedT* smem, int numVals, T threadVal, ReduceOp reduceOp, T init)
T reduceBlock(T* smem, int numVals, T threadVal, ReduceOp reduceOp, T init)
{
  if (numVals == 0) {
    return init;
  }

  if (hipThreadIdx_x < numVals) {
    smem[hipThreadIdx_x] = threadVal;
  }

  // First warp will perform reductions across warps
  __syncthreads();
  if ((hipThreadIdx_x / warpSize) == 0) {
    T r = hipThreadIdx_x < numVals ? smem[hipThreadIdx_x] : init;

    for (int i = warpSize + hipThreadIdx_x; i < numVals; i += warpSize) {
      r = reduceOp(r, smem[i]);
    }

    smem[hipThreadIdx_x] = r;
  }

  // First thread will perform reductions across the block
  __syncthreads();

  T r = init;
  if (hipThreadIdx_x == 0) {
    r = smem[0];

    int numLanesParticipating = min(numVals, warpSize);

    if (numLanesParticipating == warpSize) {
      // Unroll for warpSize == 32 and numVals >= 32
#pragma unroll
      for (int i = 1; i < warpSize; ++i) {
        r = reduceOp(r, smem[i]);
      }
    } else {
      for (int i = 1; i < numLanesParticipating; ++i) {
        r = reduceOp(r, smem[i]);
      }
    }
  }

  return r;
}

// Make sure the given tensor doesn't have too many dimensions
void THCCheckTensorDims(THCState* state, THCudaTensor* tensor, int arg);

// Produces a grid with at least one point per tile
THC_API bool THC_getGridFromTiles(ptrdiff_t gridTiles, dim3& grid);

#endif // THC_REDUCE_APPLY_UTILS_INC
