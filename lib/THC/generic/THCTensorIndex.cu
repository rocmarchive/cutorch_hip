#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorIndex.cu"
#else

void THCTensor_(indexCopy_long)(THCState *state, THCTensor *dst, int dim, THLongTensor *indices, THCTensor *src)
{
  THAssert(THCTensor_(checkGPU)(state, 2, dst, src));

  THCudaLongTensor *indices_ = THCudaLongTensor_newWithSize1d(state, indices->size[0]);
  THCudaLongTensor_copyLong(state, indices_, indices);

  THCTensor_(indexCopy)(state, dst, dim, indices_, src);

  THCudaLongTensor_free(state, indices_);
}

void THCTensor_(indexCopy)(THCState *state, THCTensor *dst, int dim, THCudaLongTensor *indices, THCTensor *src)
{
  THAssert(THCTensor_(checkGPU)(state, 2, dst, src));
  THAssert(THCudaLongTensor_checkGPU(state, 1, indices));

  long dims = THCTensor_(nDimension)(state, dst);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  dims = THCTensor_(nDimension)(state, src);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 5, CUTORCH_DIM_WARNING);
  dims = THCudaLongTensor_nDimension(state, indices);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 4, CUTORCH_DIM_WARNING);

  ptrdiff_t numIndices = THCudaLongTensor_nElement(state, indices);

  long srcDims = THCTensor_(nDimension)(state, src);
  hipStream_t stream = THCState_getCurrentStream(state);

  THArgCheck(THCudaLongTensor_nDimension(state, indices) == 1, 3,
             "expecting vector of indices");
  THArgCheck(dim < srcDims, 4, "Indexing dim is out of bounds");
  THArgCheck(srcDims > 0, 2, "Source tensor is empty");
  THArgCheck(numIndices == src->size[dim], 4, "length of src.size[dim] is not equal to length of indices");

  int indContig = THCudaLongTensor_isContiguous(state, indices);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  ptrdiff_t srcTotalSize = THCTensor_(nElement)(state, src);
  long dstCopyDimSize = THCTensor_(size)(state, dst, dim);
  ptrdiff_t sliceSize = srcTotalSize / numIndices;

  int mpc = THCState_getCurrentDeviceProperties(state)->multiProcessorCount;

#define SMALL_INDEX(TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM) \
  invokeSmallIndex<TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM>(       \
    smallIndexGrid, smallIndexBlock, 0, stream,           \
      dstData, dstSizes, dstStrides, dstDims, \
      srcData, srcSizes, srcStrides, srcDims, \
      indData, indSizes, indStrides, indDims,\
      dstCopyDim, srcCopyDim, sliceSize, dstCopyDimSize);

#ifdef CUDA_PATH
#define LARGE_INDEX(TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM) \
  invokeLargeIndex<TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM>(       \
      largeIndexGrid, largeIndexBlock, 0, stream,          \
      dstData, dstSizes, dstStrides, dstDims, \
      srcData, srcSizes, srcStrides, srcDims, \
      indData, indSizes, indStrides, indDims,\
      dstCopyDim, srcCopyDim, sliceSize, dstCopyDimSize); 

#else
  #define LARGE_INDEX(TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM)
#endif 

  dim3 smallIndexGrid(std::min(THCCeilDiv(sliceSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 smallIndexBlock(std::min(sliceSize, (ptrdiff_t)128));

  dim3 largeIndexGrid(std::min(THCCeilDiv(srcTotalSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 largeIndexBlock(std::min(srcTotalSize, (ptrdiff_t)128));

  if (TensorUtils<THCTensor>::canUse32BitIndexMath(state, dst) &&
      TensorUtils<THCTensor>::canUse32BitIndexMath(state, src) &&
      TensorUtils<THCudaLongTensor>::canUse32BitIndexMath(state, indices)) {
    TensorInfo<real, unsigned int> dstInfo =
      getTensorInfo<THCTensor, unsigned int>(state, dst);
    int dstCopyDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstCopyDim);

    TensorInfo<real, unsigned int> srcInfo =
      getTensorInfo<THCTensor, unsigned int>(state, src);
    int srcCopyDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcCopyDim);

    TensorInfo<long, unsigned int> indicesInfo =
      getTensorInfo<THCudaLongTensor, unsigned int>(state, indices);
    indicesInfo.collapseDims();

    // Declaration of extra variables
    real* srcData, *dstData; 
    long *indData;
    unsigned int *srcSizes, *srcStrides, *dstSizes, *dstStrides, *indSizes, *indStrides;
    int srcDims, dstDims, indDims;
    // Assign value to data 
    srcData = srcInfo.data;
    dstData = dstInfo.data;
    indData = indicesInfo.data;
    srcStrides = srcInfo.dStrides;
    dstStrides = dstInfo.dStrides;
    indStrides = indicesInfo.dStrides;
    srcSizes = srcInfo.dSizes;
    dstSizes = dstInfo.dSizes;
    indSizes = indicesInfo.dSizes;
    srcDims = srcInfo.dims;
    dstDims = dstInfo.dims;
    indDims = indicesInfo.dims;

    // A reasonable choice for when to have each thread iterate over
    // indices to choose
#ifdef CUDA_PATH
    if (numIndices <= 16) {
      if (dstDims == 1 && srcDims == 1 && indContig) {
        SMALL_INDEX(real, unsigned int, 1, 1, -2);
      } else if (dstDims == 2 && srcDims == 2 && indContig) {
        SMALL_INDEX(real, unsigned int, 2, 2, -2);
      } else if (dstDims == 3 && srcDims == 3 && indContig) {
        SMALL_INDEX(real, unsigned int, 3, 3, -2);
      } else {
        SMALL_INDEX(real, unsigned int, -1, -1, -1);
      }
    } else {
      if (dstDims == 1 && srcDims == 1 && indContig) {
        LARGE_INDEX(real, unsigned int, 1, 1, -2);
      } else if (dstDims == 2 && srcDims == 2 && indContig) {
        LARGE_INDEX(real, unsigned int, 2, 2, -2);
      } else if (dstDims == 3 && srcDims == 3 && indContig) {
        LARGE_INDEX(real, unsigned int, 3, 3, -2);
      } else {
        LARGE_INDEX(real, unsigned int, -1, -1, -1);
      }
    }
#endif
  } else {
    TensorInfo<real, unsigned long> dstInfo =
      getTensorInfo<THCTensor, unsigned long>(state, dst);
    int dstCopyDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstCopyDim);

    TensorInfo<real, unsigned long> srcInfo =
      getTensorInfo<THCTensor, unsigned long>(state, src);
    int srcCopyDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcCopyDim);

    TensorInfo<long, unsigned long> indicesInfo =
      getTensorInfo<THCudaLongTensor, unsigned long>(state, indices);
    indicesInfo.collapseDims();
    // Declaration of extra variables
    real* srcData, *dstData; 
    long *indData;
    unsigned long *srcSizes, *srcStrides, *dstSizes, *dstStrides, *indSizes, *indStrides;
    int srcDims, dstDims, indDims;
    // Assign value to data 
    srcData = srcInfo.data;
    dstData = dstInfo.data;
    indData = indicesInfo.data;
    srcStrides = srcInfo.dStrides;
    dstStrides = dstInfo.dStrides;
    indStrides = indicesInfo.dStrides;
    srcSizes = srcInfo.dSizes;
    dstSizes = dstInfo.dSizes;
    indSizes = indicesInfo.dSizes;
    srcDims = srcInfo.dims;
    dstDims = dstInfo.dims;
    indDims = indicesInfo.dims;


    LARGE_INDEX(real, unsigned long, -1, -1, -1);
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
}

void THCTensor_(indexAdd_long)(THCState *state, THCTensor *dst, int dim, THLongTensor *indices, THCTensor *src)
{
  THAssert(THCTensor_(checkGPU)(state, 2, dst, src));

  THCudaLongTensor *indices_ = THCudaLongTensor_newWithSize1d(state, indices->size[0]);
  THCudaLongTensor_copyLong(state, indices_, indices);

  THCTensor_(indexAdd)(state, dst, dim, indices_, src);

  THCudaLongTensor_free(state, indices_);
}

void THCTensor_(indexAdd)(THCState *state, THCTensor *dst, int dim, THCudaLongTensor *indices, THCTensor *src)
{
  THAssert(THCTensor_(checkGPU)(state, 2, dst, src));
  THAssert(THCudaLongTensor_checkGPU(state, 1, indices));

  long dims = THCTensor_(nDimension)(state, dst);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  dims = THCTensor_(nDimension)(state, src);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 5, CUTORCH_DIM_WARNING);
  dims = THCudaLongTensor_nDimension(state, indices);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 4, CUTORCH_DIM_WARNING);

  ptrdiff_t numIndices = THCudaLongTensor_nElement(state, indices);

  long srcDims = THCTensor_(nDimension)(state, src);
  hipStream_t stream = THCState_getCurrentStream(state);

  THArgCheck(THCudaLongTensor_nDimension(state, indices) == 1, 3,
             "expecting vector of indices");
  THArgCheck(dim < srcDims, 4, "Indexing dim is out of bounds");
  THArgCheck(srcDims > 0, 2, "Source tensor is empty");
  THArgCheck(numIndices == src->size[dim], 4, "length of src.size[dim] is not equal to length of indices");

  int indContig = THCudaLongTensor_isContiguous(state, indices);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  ptrdiff_t srcTotalSize = THCTensor_(nElement)(state, src);
  long dstAddDimSize = THCTensor_(size)(state, dst, dim);
  ptrdiff_t sliceSize = srcTotalSize / numIndices;

  int mpc = THCState_getCurrentDeviceProperties(state)->multiProcessorCount;

#ifdef CUDA_PATH
#define SMALL_INDEX(TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM) \
  invokeAddSmallIndex<TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM>( \
      smallIndexGrid, smallIndexBlock, 0, stream,   \
      dstData, dstSizes, dstStrides, dstDims, \
      srcData, srcSizes, srcStrides, srcDims, \
      indData, indSizes, indStrides, indDims,\
      dstAddDim, srcAddDim, sliceSize, dstAddDimSize);
 
#else
  #define SMALL_INDEX(TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM)
#endif 

#ifdef CUDA_PATH
#define LARGE_INDEX(TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM) \
  invokeAddLargeIndex<TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM>( \
      largeIndexGrid, largeIndexBlock, 0, stream,   \
      dstData, dstSizes, dstStrides, dstDims, \
      srcData, srcSizes, srcStrides, srcDims, \
      indData, indSizes, indStrides, indDims,\
      dstAddDim, srcAddDim, sliceSize, dstAddDimSize); 

#else
#define LARGE_INDEX(TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM) 
#endif

  dim3 smallIndexGrid(std::min(THCCeilDiv(sliceSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 smallIndexBlock(std::min(sliceSize, (ptrdiff_t)128));

  dim3 largeIndexGrid(std::min(THCCeilDiv(srcTotalSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 largeIndexBlock(std::min(srcTotalSize, (ptrdiff_t)128));

  if (TensorUtils<THCTensor>::canUse32BitIndexMath(state, dst) &&
      TensorUtils<THCTensor>::canUse32BitIndexMath(state, src) &&
      TensorUtils<THCudaLongTensor>::canUse32BitIndexMath(state, indices)) {
    TensorInfo<real, unsigned int> dstInfo =
      getTensorInfo<THCTensor, unsigned int>(state, dst);
    int dstAddDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstAddDim);

    TensorInfo<real, unsigned int> srcInfo =
      getTensorInfo<THCTensor, unsigned int>(state, src);
    int srcAddDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcAddDim);

    TensorInfo<long, unsigned int> indicesInfo =
      getTensorInfo<THCudaLongTensor, unsigned int>(state, indices);
    indicesInfo.collapseDims();

    // Declaration of extra variables
    real* srcData, *dstData; 
    long *indData;
    unsigned int *srcSizes, *srcStrides, *dstSizes, *dstStrides, *indSizes, *indStrides;
    int srcDims, dstDims, indDims;
    // Assign value to data 
    srcData = srcInfo.data;
    dstData = dstInfo.data;
    indData = indicesInfo.data;
    srcStrides = srcInfo.dStrides;
    dstStrides = dstInfo.dStrides;
    indStrides = indicesInfo.dStrides;
    srcSizes = srcInfo.dSizes;
    dstSizes = dstInfo.dSizes;
    indSizes = indicesInfo.dSizes;
    srcDims = srcInfo.dims;
    dstDims = dstInfo.dims;
    indDims = indicesInfo.dims;

    // A reasonable choice for when to have each thread iterate over
    // indices to choose
#ifdef CUDA_PATH
    if (numIndices <= 16) {
      if (dstDims == 1 && srcDims == 1 && indContig) {
        SMALL_INDEX(real, unsigned int, 1, 1, -2);
      } else if (dstDims == 2 && srcDims == 2 && indContig) {
        SMALL_INDEX(real, unsigned int, 2, 2, -2);
      } else if (dstDims == 3 && srcDims == 3 && indContig) {
        SMALL_INDEX(real, unsigned int, 3, 3, -2);
      } else {
        SMALL_INDEX(real, unsigned int, -1, -1, -1);
      }
    } else {
      if (dstDims == 1 && srcDims == 1 && indContig) {
        LARGE_INDEX(real, unsigned int, 1, 1, -2);
      } else if (dstDims == 2 && srcDims == 2 && indContig) {
        LARGE_INDEX(real, unsigned int, 2, 2, -2);
      } else if (dstDims == 3 && srcDims == 3 && indContig) {
        LARGE_INDEX(real, unsigned int, 3, 3, -2);
      } else {
        LARGE_INDEX(real, unsigned int, -1, -1, -1);
      }
    }
#endif
  } else {
    TensorInfo<real, unsigned long> dstInfo =
      getTensorInfo<THCTensor, unsigned long>(state, dst);
    int dstAddDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstAddDim);

    TensorInfo<real, unsigned long> srcInfo =
      getTensorInfo<THCTensor, unsigned long>(state, src);
    int srcAddDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcAddDim);

    TensorInfo<long, unsigned long> indicesInfo =
      getTensorInfo<THCudaLongTensor, unsigned long>(state, indices);
    indicesInfo.collapseDims();
    // Declaration of extra variables
    real* srcData, *dstData; 
    long *indData;
    unsigned long *srcSizes, *srcStrides, *dstSizes, *dstStrides, *indSizes, *indStrides;
    int srcDims, dstDims, indDims;
    // Assign value to data 
    srcData = srcInfo.data;
    dstData = dstInfo.data;
    indData = indicesInfo.data;
    srcStrides = srcInfo.dStrides;
    dstStrides = dstInfo.dStrides;
    indStrides = indicesInfo.dStrides;
    srcSizes = srcInfo.dSizes;
    dstSizes = dstInfo.dSizes;
    indSizes = indicesInfo.dSizes;
    srcDims = srcInfo.dims;
    dstDims = dstInfo.dims;
    indDims = indicesInfo.dims;


    LARGE_INDEX(real, unsigned long, -1, -1, -1);
  }
#undef SMALL_INDEX
#undef LARGE_INDEX
}

void THCTensor_(indexFill_long)(THCState *state, THCTensor *dst, int dim, THLongTensor *indices, real val)
{
  THAssert(THCTensor_(checkGPU)(state, 1, dst));

  THCudaLongTensor *indices_ = THCudaLongTensor_newWithSize1d(state, indices->size[0]);
  THCudaLongTensor_copyLong(state, indices_, indices);

  THCTensor_(indexFill)(state, dst, dim, indices_, val);

  THCudaLongTensor_free(state, indices_);
}

void THCTensor_(indexFill)(THCState *state, THCTensor *dst, int dim, THCudaLongTensor *indices, real val)
{
  THAssert(THCTensor_(checkGPU)(state, 1, dst));
  THAssert(THCudaLongTensor_checkGPU(state, 1, indices));
  long dims = THCTensor_(nDimension)(state, dst);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  dims = THCudaLongTensor_nDimension(state, indices);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 4, CUTORCH_DIM_WARNING);

  ptrdiff_t numIndices = THCudaLongTensor_nElement(state, indices);

  long srcDims = THCTensor_(nDimension)(state, dst);
  hipStream_t stream = THCState_getCurrentStream(state);

  THArgCheck(THCudaLongTensor_nDimension(state, indices) == 1, 3,
             "expecting vector of indices");
  THArgCheck(dim < srcDims, 4, "Indexing dim is out of bounds");
  THArgCheck(srcDims > 0, 2, "Source tensor is empty");

  int indContig = THCudaLongTensor_isContiguous(state, indices);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  ptrdiff_t dstTotalSize = THCTensor_(nElement)(state, dst);
  long dstFillDimSize = THCTensor_(size)(state, dst, dim);
  ptrdiff_t sliceSize = dstTotalSize / dstFillDimSize;

  int mpc = THCState_getCurrentDeviceProperties(state)->multiProcessorCount;

#ifdef CUDA_PATH
#define SMALL_INDEX(TENSOR_TYPE, TYPE, DST_DIM, IDX_DIM)  \
  invokeFillSmallIndex<TENSOR_TYPE, TYPE, DST_DIM, IDX_DIM>( \
      smallIndexGrid, smallIndexBlock, 0, stream,   \
      dstData, dstSizes, dstStrides, dstDims, \
      indData, indSizes, indStrides, indDims,\
      dstFillDim, sliceSize, dstFillDimSize, val);
 
#else
  #define SMALL_INDEX(TENSOR_TYPE, TYPE, DST_DIM, IDX_DIM)  
#endif

#ifdef CUDA_PATH
#define LARGE_INDEX(TENSOR_TYPE, TYPE, DST_DIM, IDX_DIM)  \
  invokeFillLargeIndex<TENSOR_TYPE, TYPE, DST_DIM, IDX_DIM>(\
      largeIndexGrid, largeIndexBlock, 0, stream,   \
      dstData, dstSizes, dstStrides, dstDims, \
      indData, indSizes, indStrides, indDims,\
      dstFillDim, sliceSize, dstFillDimSize, val); 

#else
  #define LARGE_INDEX(TENSOR_TYPE, TYPE, DST_DIM, IDX_DIM)  
#endif

  dim3 smallIndexGrid(std::min(THCCeilDiv(sliceSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 smallIndexBlock(std::min(sliceSize, (ptrdiff_t)128));

  dim3 largeIndexGrid(std::min(THCCeilDiv(dstTotalSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 largeIndexBlock(std::min(dstTotalSize, (ptrdiff_t)128));

  if (TensorUtils<THCTensor>::canUse32BitIndexMath(state, dst) &&
      TensorUtils<THCudaLongTensor>::canUse32BitIndexMath(state, indices)) {
    TensorInfo<real, unsigned int> dstInfo =
      getTensorInfo<THCTensor, unsigned int>(state, dst);
    int dstFillDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstFillDim);

    TensorInfo<long, unsigned int> indicesInfo =
      getTensorInfo<THCudaLongTensor, unsigned int>(state, indices);
    indicesInfo.collapseDims();

    // Declaration of extra variables
    real* dstData; 
    long *indData;
    unsigned int *dstSizes, *dstStrides, *indSizes, *indStrides;
    int dstDims, indDims;
    // Assign value to data 
    dstData = dstInfo.data;
    indData = indicesInfo.data;
    dstStrides = dstInfo.dStrides;
    indStrides = indicesInfo.dStrides;
    dstSizes = dstInfo.dSizes;
    indSizes = indicesInfo.dSizes;
    dstDims = dstInfo.dims;
    indDims = indicesInfo.dims;

    // A reasonable choice for when to have each thread iterate over
    // indices to choose
#ifdef CUDA_PATH
    if (numIndices <= 16) {
      if (dstDims == 1 && indContig) {
        SMALL_INDEX(real, unsigned int, 1, -2);
      } else if (dstDims == 2 && indContig) {
        SMALL_INDEX(real, unsigned int, 2, -2);
      } else if (dstDims == 3 && indContig) {
        SMALL_INDEX(real, unsigned int, 3, -2);
      } else {
        SMALL_INDEX(real, unsigned int, -1, -1);
      }
    } else {
      if (dstDims == 1 && indContig) {
        LARGE_INDEX(real, unsigned int, 1, -2);
      } else if (dstDims == 2 && indContig) {
        LARGE_INDEX(real, unsigned int, 2, -2);
      } else if (dstDims == 3 && indContig) {
        LARGE_INDEX(real, unsigned int, 3, -2);
      } else {
        LARGE_INDEX(real, unsigned int, -1, -1);
      }
    }
#endif
  } else {
    TensorInfo<real, unsigned long> dstInfo =
      getTensorInfo<THCTensor, unsigned long>(state, dst);
    int dstFillDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstFillDim);

    TensorInfo<long, unsigned long> indicesInfo =
      getTensorInfo<THCudaLongTensor, unsigned long>(state, indices);
    indicesInfo.collapseDims();
    // Declaration of extra variables
    real* dstData; 
    long *indData;
    unsigned long *dstSizes, *dstStrides, *indSizes, *indStrides;
    int dstDims, indDims;
    // Assign value to data 
    dstData = dstInfo.data;
    indData = indicesInfo.data;
    dstStrides = dstInfo.dStrides;
    indStrides = indicesInfo.dStrides;
    dstSizes = dstInfo.dSizes;
    indSizes = indicesInfo.dSizes;
    dstDims = dstInfo.dims;
    indDims = indicesInfo.dims;


    LARGE_INDEX(real, unsigned long, -1, -1);
  }
#undef SMALL_INDEX
#undef LARGE_INDEX
}


void THCTensor_(indexSelect_long)(THCState *state, THCTensor *dst, THCTensor *src, int dim, THLongTensor *indices)
{
  THAssert(THCTensor_(checkGPU)(state, 2, dst, src));
  THArgCheck(indices->nDimension == 1, 3, "Index is supposed to be a vector");

  THCudaLongTensor *indices_ = THCudaLongTensor_newWithSize1d(state, indices->size[0]);
  THCudaLongTensor_copyLong(state, indices_, indices);

  THCTensor_(indexSelect)(state, dst, src, dim, indices_);

  THCudaLongTensor_free(state, indices_);
}

void THCTensor_(indexSelect)(THCState *state, THCTensor *dst, THCTensor *src, int dim, THCudaLongTensor *indices)
{
  THAssert(THCTensor_(checkGPU)(state, 3, dst, src, indices));

  long dims = THCTensor_(nDimension)(state, dst);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  dims = THCTensor_(nDimension)(state, src);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 3, CUTORCH_DIM_WARNING);
  dims = THCudaLongTensor_nDimension(state, indices);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 5, CUTORCH_DIM_WARNING);

  ptrdiff_t numIndices = THCudaLongTensor_nElement(state, indices);

  long srcDims = THCTensor_(nDimension)(state, src);
  hipStream_t stream = THCState_getCurrentStream(state);

  THArgCheck(THCudaLongTensor_nDimension(state, indices) == 1, 3,
             "expecting vector of indices");
  THArgCheck(dim < srcDims, 4, "Indexing dim is out of bounds");
  THArgCheck(srcDims > 0, 2, "Source tensor is empty");

  THLongStorage *newSize = THCTensor_(newSizeOf)(state, src);
  THLongStorage_set(newSize, dim, numIndices);
  THCTensor_(resize)(state, dst, newSize, NULL);
  THLongStorage_free(newSize);

  int indContig = THCudaLongTensor_isContiguous(state, indices);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  ptrdiff_t dstTotalSize = THCTensor_(nElement)(state, dst);
  long srcSelectDimSize = THCTensor_(size)(state, src, dim);
  ptrdiff_t sliceSize = dstTotalSize / numIndices;

  int mpc = THCState_getCurrentDeviceProperties(state)->multiProcessorCount;

#ifdef CUDA_PATH
#define SMALL_INDEX(TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM) \
  invokeSelectSmallIndex<TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM>(     \
      smallIndexGrid, smallIndexBlock, 0, stream,          \
      dstData, dstSizes, dstStrides, dstDims, \
      srcData, srcSizes, srcStrides, srcDims, \
      indData, indSizes, indStrides, indDims,\
      dstSelectDim, srcSelectDim, sliceSize, srcSelectDimSize);

#else
#define SMALL_INDEX(TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM)
#endif

#ifdef CUDA_PATH
#define LARGE_INDEX(TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM)         \
  invokeSelectLargeIndex<TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM>(     \
      largeIndexGrid, largeIndexBlock, 0, stream,                  \
      dstData, dstSizes, dstStrides, dstDims, \
      srcData, srcSizes, srcStrides, srcDims, \
      indData, indSizes, indStrides, indDims,\
      dstSelectDim, srcSelectDim, dstTotalSize, sliceSize, srcSelectDimSize);
 
#else
  #define LARGE_INDEX(TENSOR_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM)         
#endif

  dim3 smallIndexGrid(std::min(THCCeilDiv(sliceSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 smallIndexBlock(std::min(sliceSize, (ptrdiff_t)128));

  dim3 largeIndexGrid(std::min(THCCeilDiv(dstTotalSize, (ptrdiff_t)128), (ptrdiff_t)(mpc * 8)));
  dim3 largeIndexBlock(std::min(dstTotalSize, (ptrdiff_t)128));

  if (TensorUtils<THCTensor>::canUse32BitIndexMath(state, dst) &&
      TensorUtils<THCTensor>::canUse32BitIndexMath(state, src) &&
      TensorUtils<THCudaLongTensor>::canUse32BitIndexMath(state, indices)) {
    TensorInfo<real, unsigned int> dstInfo =
      getTensorInfo<THCTensor, unsigned int>(state, dst);
    int dstSelectDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstSelectDim);

    TensorInfo<real, unsigned int> srcInfo =
      getTensorInfo<THCTensor, unsigned int>(state, src);
    int srcSelectDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcSelectDim);

    TensorInfo<long, unsigned int> indicesInfo =
      getTensorInfo<THCudaLongTensor, unsigned int>(state, indices);
    indicesInfo.collapseDims();

    // Declaration of extra variables
    real* srcData, *dstData; 
    long *indData;
    unsigned int *srcSizes, *srcStrides, *dstSizes, *dstStrides, *indSizes, *indStrides;
    int srcDims, dstDims, indDims;
    // Assign value to data 
    srcData = srcInfo.data;
    dstData = dstInfo.data;
    indData = indicesInfo.data;
    srcStrides = srcInfo.dStrides;
    dstStrides = dstInfo.dStrides;
    indStrides = indicesInfo.dStrides;
    srcSizes = srcInfo.dSizes;
    dstSizes = dstInfo.dSizes;
    indSizes = indicesInfo.dSizes;
    srcDims = srcInfo.dims;
    dstDims = dstInfo.dims;
    indDims = indicesInfo.dims;


    // A reasonable choice for when to have each thread iterate over
    // indices to choose
#ifdef CUDA_PATH
    if (numIndices <= 16) {
      if (dstDims == 1 && srcDims == 1 && indContig) {
        SMALL_INDEX(real, unsigned int, 1, 1, -2);
      } else if (dstDims == 2 && srcDims == 2 && indContig) {
        SMALL_INDEX(real, unsigned int, 2, 2, -2);
      } else if (dstDims == 3 && srcDims == 3 && indContig) {
        SMALL_INDEX(real, unsigned int, 3, 3, -2);
      } else {
        SMALL_INDEX(real, unsigned int, -1, -1, -1);
      }
    } else {
      if (dstDims == 1 && srcDims == 1 && indContig) {
        LARGE_INDEX(real, unsigned int, 1, 1, -2);
      } else if (dstDims == 2 && srcDims == 2 && indContig) {
        LARGE_INDEX(real, unsigned int, 2, 2, -2);
      } else if (dstDims == 3 && srcDims == 3 && indContig) {
        LARGE_INDEX(real, unsigned int, 3, 3, -2);
      } else {
        LARGE_INDEX(real, unsigned int, -1, -1, -1);
      }
    }
#endif
  } else {
    TensorInfo<real, unsigned long> dstInfo =
      getTensorInfo<THCTensor, unsigned long>(state, dst);
    int dstSelectDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstSelectDim);

    TensorInfo<real, unsigned long> srcInfo =
      getTensorInfo<THCTensor, unsigned long>(state, src);
    int srcSelectDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcSelectDim);

    TensorInfo<long, unsigned long> indicesInfo =
      getTensorInfo<THCudaLongTensor, unsigned long>(state, indices);
    indicesInfo.collapseDims();
    // Declaration of extra variables
    real* srcData, *dstData; 
    long *indData;
    unsigned long *srcSizes, *srcStrides, *dstSizes, *dstStrides, *indSizes, *indStrides;
    int srcDims, dstDims, indDims;
    // Assign value to data 
    srcData = srcInfo.data;
    dstData = dstInfo.data;
    indData = indicesInfo.data;
    srcStrides = srcInfo.dStrides;
    dstStrides = dstInfo.dStrides;
    indStrides = indicesInfo.dStrides;
    srcSizes = srcInfo.dSizes;
    dstSizes = dstInfo.dSizes;
    indSizes = indicesInfo.dSizes;
    srcDims = srcInfo.dims;
    dstDims = dstInfo.dims;
    indDims = indicesInfo.dims;


    LARGE_INDEX(real, unsigned long, -1, -1, -1);
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
}

#endif
