#include "hip/hip_runtime.h"
#ifndef THC_REDUCE_APPLY_UTILS_INC
#define THC_REDUCE_APPLY_UTILS_INC

#include <cuda.h>
#include <assert.h>
#include "THCGeneral.h"
#include "THCTensor.h"
#include "THCDeviceUtils.cuh"
#include "THCTensorInfo.cuh"

// Enum that indicates whether tensor arguments are read/write or
// read-only
enum TensorArgType { ReadWrite, ReadOnly };

template <typename IndexType>
__device__ __forceinline__ IndexType getLinearBlockId() {
  return hipBlockIdx_z * hipGridDim_y * hipGridDim_x +
    hipBlockIdx_y * hipGridDim_x +
    hipBlockIdx_x;
}

// Block-wide reduction in shared memory helper; only hipThreadIdx_x == 0 will
// return the reduced value
template <typename T, typename ReduceOp>
__device__ T reduceBlock(T* smem,
                         int numVals,
                         T threadVal,
                         ReduceOp reduceOp,
                         T init) {
  if (numVals == 0) {
    return init;
  }

  if (hipThreadIdx_x < numVals) {
    smem[hipThreadIdx_x] = threadVal;
  }

  // First warp will perform reductions across warps
  __syncthreads();
  if ((hipThreadIdx_x / hipWarpSize) == 0) {
    T r = hipThreadIdx_x < numVals ? smem[hipThreadIdx_x] : init;

    for (int i = hipWarpSize + hipThreadIdx_x; i < numVals; i += hipWarpSize) {
      r = reduceOp(r, smem[i]);
    }

    smem[hipThreadIdx_x] = r;
  }

  // First thread will perform reductions across the block
  __syncthreads();

  T r = init;
  if (hipThreadIdx_x == 0) {
    r = smem[0];

    int numLanesParticipating = min(numVals, hipWarpSize);

    if (numLanesParticipating == 32) {
      // Unroll for hipWarpSize == 32 and numVals >= 32
#pragma unroll
      for (int i = 1; i < 32; ++i) {
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
