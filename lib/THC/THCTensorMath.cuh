#ifndef THC_TENSORMATH_CUH
#define THC_TENSORMATH_CUH

// Copy the kth diagonal of a matrix B to a vector A.
template <typename T>
__global__ void THCTensor_copyFromDiagonal(T* a, T* b, ptrdiff_t start, ptrdiff_t size, ptrdiff_t strideSum, ptrdiff_t strideA) {
  for (ptrdiff_t linearIndex = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
       linearIndex < size;
       linearIndex += hipGridDim_x * hipBlockDim_x) {
    const ptrdiff_t bOffset = start + strideSum * linearIndex;
    a[strideA * linearIndex] = b[bOffset];
  }
}

// Copy vector B to the kth diagonal of a matrix A
template <typename T>
__global__ void THCTensor_copyToDiagonal(T* a, T* b, ptrdiff_t start, ptrdiff_t size, ptrdiff_t strideSum, ptrdiff_t strideB) {
  for (ptrdiff_t linearIndex = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
       linearIndex < size;
       linearIndex += hipGridDim_x * hipBlockDim_x) {
    const ptrdiff_t aOffset = start + strideSum * linearIndex;
    a[aOffset] = b[strideB * linearIndex];
  }
}

#define CAT_ARRAY_BATCH_SIZE 1024
#define CAT_ARRAY_MAX_INPUT_DIMS 4

// Similar to any other IndexToOffset calculation for copying along a given dimension.
template <typename IndexType, int Dims>
struct CatArrIndexToOffset {
  static inline __device__ IndexType compute(
      const IndexType outputSize[Dims],
      const IndexType outputStride[Dims],
      const IndexType dimSize,
      const unsigned int concatDim,
      IndexType linearIndex) {
    IndexType offset = 0;

#pragma unroll
    for (int i = Dims - 1; i >= 1; --i) {
      IndexType curDimSize = i == concatDim ? dimSize : outputSize[i];
      IndexType nextDimIndex = linearIndex / curDimSize;
      IndexType curDimIndex = linearIndex - curDimSize * nextDimIndex;
      IndexType curDimOffset = curDimIndex * outputStride[i];
      offset += curDimOffset;
      linearIndex = nextDimIndex;
    }

    return offset + linearIndex * outputStride[0];
  }
};

template <typename T, typename IndexType>
struct CatArrInputTensor {
  T* input;
  IndexType offset;
  IndexType dimSize;
  IndexType nElements;
};

template<typename IndexType, unsigned int MaxDims>
struct OutputTensorSizeStride {
  IndexType outputSize[MaxDims];
  IndexType outputStride[MaxDims];
  IndexType* devOutputSize;
  IndexType* devOutputStride;

  __host__ __device__ OutputTensorSizeStride(IndexType, IndexType, IndexType*, IndexType*) {}   
  // Create device Tensors
  __host__ __device__ OutputTensorSizeStride() {
#if __HIP_DEVICE_COMPILE__
   // Do Nothing
#else
    hipMalloc(&devOutputSize, MaxDims * sizeof(IndexType));
    hipMalloc(&devOutputStride, MaxDims * sizeof(IndexType));
#endif
  }

  // Destroy device tensors
  __host__ __device__ ~OutputTensorSizeStride() {
     //TODO Device Tensors destructions
   }  
};

/**
  * Kernel used to concatenated grimDim.y tensors into an output tensor. Uses a grid-stride loop based off of
  * the hipBlockIdx_x, hipThreadIdx_x for each input to copy each element from each input tensor into the output.
  *
  * output: base pointer to the storage associated with the output tensor
  * inputs: GPU-allocated array of input metadata for each input to concatenate in the kernel
  * os: the size/stride vectors for the output tensor
  * concatDim: dimension along which we are concatenating
  * dimStride: the stride of the output tensor at the concatDim
  *
  * The most important assumption made is that the input tensors are contiguous.
  */
template <typename T, typename IndexType, int Dims>
__global__ void CatArrayBatchedCopy(
    T* output,
    CatArrInputTensor<T, IndexType>* inputs,
    OutputTensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
    const int concatDim,
    IndexType* dimStride) {
  T* data = inputs[hipBlockIdx_y].input;
  IndexType offset = inputs[hipBlockIdx_y].offset;
  IndexType dimSize = inputs[hipBlockIdx_y].dimSize;
  IndexType nElements = inputs[hipBlockIdx_y].nElements;
  IndexType dataOffset = offset * dimStride[concatDim];

  for (IndexType linearIndex = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
      linearIndex < nElements;
      linearIndex += hipGridDim_x * hipBlockDim_x) {
    IndexType elementOffset = CatArrIndexToOffset<IndexType, Dims>::compute(
        os.devOutputSize, os.devOutputStride, dimSize, concatDim, linearIndex);
    output[dataOffset + elementOffset] = data[linearIndex];
  }
}

#endif
