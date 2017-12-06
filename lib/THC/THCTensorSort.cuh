#include "hip/hip_runtime.h"
#ifndef THC_TENSORSORT_CUH
#define THC_TENSORSORT_CUH

#include "THCReduceApplyUtils.cuh"
#include "THCSortUtils.cuh"
#include "THCTensorCopy.h"
#include "THCTensorTypeUtils.cuh"

#include "THCThrustAllocator.cuh"
#ifdef __HIP_PLATFORM_NVCC__
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#endif
#if CUDA_VERSION >= 7000
  #include <thrust/system/cuda/execution_policy.h>
#endif

template <typename T>
struct ThrustGTOp {
  __host__ __device__
  bool operator()(const T& lhs, const T& rhs) const {
    return THCNumerics<T>::gt(lhs, rhs);
  }
};

template <typename T>
struct ThrustLTOp {
  __host__ __device__
  bool operator()(const T& lhs, const T& rhs) const
  {
    return THCNumerics<T>::lt(lhs, rhs);
  }
};

// `base` is the base address of a tensor
// For each slice (defined as a linear point of `out`, from 0 ->
// (sliceSize - 1) * sliceStride, we fill that slice from `0` to
// `sliceSize - 1`.
template <typename IndexType, int Dim>
__global__
void fillSliceWithIndex(
    TensorInfo<long, IndexType> out,
    IndexType totalSlices,
    IndexType sliceSize,
    IndexType sliceStride)
{
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
  __host__ __device__
  explicit
  SliceComp(long size) : sliceSize(size) {}

  __device__
  bool operator()(long a, long b) const
  {
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
  GlobalIndexToPerSliceIndex(long size) : sliceSize(size) {}

  __device__
  void operator()(long& v) const {
    v = v % sliceSize + TH_INDEX_BASE;
  }

  const long sliceSize;
};

void THCudaLongTensor_fillSliceWithIndex(
    THCState* state, THCudaLongTensor* t, int dim);

template <typename T>
long partition (T* arr, long* indices, int low, int high, long stride)
{
    T pivot = arr[high * stride];    // pivot
    long i = (low - 1);  // Index of smaller element

    T t1;
    long t2;

    for (long j = low; j <= high- 1; j++)
    {
        if (arr[j * stride] <= pivot)
        {
            i++;
            t1 = arr[i * stride];
            arr[i * stride] = arr[j * stride];
            arr[j * stride] = t1;

            t2 = indices[i * stride];
            indices[i * stride] = indices[j * stride];
            indices[j * stride] = t2;
        }
    }
    t1 = arr[(i + 1) * stride];
    arr[(i + 1) * stride] = arr[high * stride];
    arr[high * stride] = t1;

    t2 = indices[(i + 1) * stride];
    indices[(i + 1) * stride] = indices[high * stride];
    indices[high * stride] = t2;
    return (i + 1);
}

template <typename T>
void quick_sort (T* arr, long* indices, int l, int h, long stride) {
    long stack[ h - l + 1 ];

    long top = -1;

    stack[ ++top ] = l;
    stack[ ++top ] = h;

    while ( top >= 0 )
    {
        h = stack[ top-- ];
        l = stack[ top-- ];

        long p = partition<T>( arr, indices, l, h, stride );

        if (p-1 > l)
        {
            stack[ ++top ] = l;
            stack[ ++top ] = p - 1;
        }

        if (p+1 < h)
        {
            stack[ ++top ] = p + 1;
            stack[ ++top ] = h;
        }
    }
}
#endif // THC_TENSORSORT_CUH
