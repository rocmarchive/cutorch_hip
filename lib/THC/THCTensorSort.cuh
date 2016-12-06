#pragma once
#ifndef THC_TENSORSORT_CUH
#define THC_TENSORSORT_CUH

#include "THCReduceApplyUtils.cuh"
#include "THCSortUtils.cuh"
#include "THCTensorCopy.h"
#include "THCTensorTypeUtils.cuh"

#include <hip/hip_runtime.h>
#include <bolt/amp/sort.h>

template <typename T>
struct ThrustGTOp {
  __device__ bool operator()(const T& lhs, const T& rhs) const {
    return THCNumerics<T>::gt(lhs, rhs);
  }
};

template <typename T>
struct ThrustLTOp {
  __device__ bool operator()(const T& lhs, const T& rhs) const {
    return THCNumerics<T>::lt(lhs, rhs);
  }
};

// `base` is the base address of a tensor
// For each slice (defined as a linear point of `out`, from 0 ->
// (sliceSize - 1) * sliceStride, we fill that slice from `0` to
// `sliceSize - 1`.
template <typename IndexType, int Dim>
__global__
inline
void
fillSliceWithIndex(hipLaunchParm lp,
                   TensorInfo<long, IndexType> out,
                   IndexType totalSlices,
                   IndexType sliceSize,
                   IndexType sliceStride) {
  IndexType slice = getLinearBlockId<IndexType>();

  if (slice >= totalSlices) {
    return;
  }

  const unsigned long offset =
    IndexToOffset<long, IndexType, Dim>::get(slice, out);
  long* base = &out.data[offset];

  for (long i = hipThreadIdx_x; i < sliceSize; i += hipBlockDim_x) {
    // Torch indices are 1-based (hence the +1)
    base[i * sliceStride] = i + TH_INDEX_BASE;
  }
}

// For slice sorting in Thrust; extracts a slice index from a linear
// index and uses that for comparison
struct SliceComp {
  explicit
  __host__ __device__
  SliceComp(long size) : sliceSize{size} {}

  __device__ bool operator()(const long& a, const long& b) const {
    // Since the slices are guaranteed to be innermost, the segment is
    // just via long division
    long segA = a / sliceSize;
    long segB = b / sliceSize;
    return segA < segB;
  }

  const long sliceSize;
};

// For sorting in Thurst; extracts a within-slice index from a linear index
struct GlobalIndexToPerSliceIndex {
  __host__ __device__
  explicit
  GlobalIndexToPerSliceIndex(long size) : sliceSize{size} {}

  __device__ inline void operator()(long& v) const {
    v = v % sliceSize + TH_INDEX_BASE;
  }

  long sliceSize;
};

unsigned long nextHighestPowerOf2(unsigned long n);
void THCudaLongTensor_fillSliceWithIndex(THCState* state,
                                         THCudaLongTensor* t,
                                         int dim);
#endif // THC_TENSORSORT_CUH
