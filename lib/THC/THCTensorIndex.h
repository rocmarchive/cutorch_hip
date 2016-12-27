// TypeCast Template Specializations

// REAL == unsigned char

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSmallIndex(dim3, dim3, int, hipStream_t, unsigned char *, unsigned int *, unsigned int *, int, unsigned char *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeLargeIndex(dim3, dim3, int, hipStream_t, unsigned char *, unsigned int *, unsigned int *, int, unsigned char *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeLargeIndex(dim3, dim3, int, hipStream_t, unsigned char *, unsigned long *, unsigned long *, int, unsigned char *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeAddSmallIndex(dim3, dim3, int, hipStream_t, unsigned char *, unsigned int *, unsigned int *, int, unsigned char *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeAddLargeIndex(dim3, dim3, int, hipStream_t, unsigned char *, unsigned int *, unsigned int *, int, unsigned char *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeAddLargeIndex(dim3, dim3, int, hipStream_t, unsigned char *, unsigned long *, unsigned long *, int, unsigned char *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSelectSmallIndex(dim3, dim3, int, hipStream_t, unsigned char *, unsigned int *, unsigned int *, int, unsigned char *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSelectLargeIndex(dim3, dim3, int, hipStream_t, unsigned char *, unsigned int *, unsigned int *, int, unsigned char *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, ptrdiff_t, long);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSelectLargeIndex(dim3, dim3, int, hipStream_t, unsigned char *, unsigned long *, unsigned long *, int, unsigned char *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, int, ptrdiff_t, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillSmallIndex(dim3, dim3, int, hipStream_t, unsigned char *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, ptrdiff_t, long, unsigned char);

template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillLargeIndex(dim3, dim3, int, hipStream_t, unsigned char *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, ptrdiff_t, long, unsigned char);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillLargeIndex(dim3, dim3, int, hipStream_t, unsigned char *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, ptrdiff_t, long, unsigned char);

// REAL == char

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSmallIndex(dim3, dim3, int, hipStream_t, char *, unsigned int *, unsigned int *, int, char *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeLargeIndex(dim3, dim3, int, hipStream_t, char *, unsigned int *, unsigned int *, int, char *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeLargeIndex(dim3, dim3, int, hipStream_t, char *, unsigned long *, unsigned long *, int, char *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeAddSmallIndex(dim3, dim3, int, hipStream_t, char *, unsigned int *, unsigned int *, int, char *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeAddLargeIndex(dim3, dim3, int, hipStream_t, char *, unsigned int *, unsigned int *, int, char *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeAddLargeIndex(dim3, dim3, int, hipStream_t, char *, unsigned long *, unsigned long *, int, char *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSelectSmallIndex(dim3, dim3, int, hipStream_t, char *, unsigned int *, unsigned int *, int, char *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSelectLargeIndex(dim3, dim3, int, hipStream_t, char *, unsigned int *, unsigned int *, int, char *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, ptrdiff_t, long);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSelectLargeIndex(dim3, dim3, int, hipStream_t, char *, unsigned long *, unsigned long *, int, char *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, int, ptrdiff_t, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillSmallIndex(dim3, dim3, int, hipStream_t, char *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, ptrdiff_t, long, char);

template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillLargeIndex(dim3, dim3, int, hipStream_t, char *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, ptrdiff_t, long, char);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillLargeIndex(dim3, dim3, int, hipStream_t, char *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, ptrdiff_t, long, char);

// REAL == int

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSmallIndex(dim3, dim3, int, hipStream_t, int *, unsigned int *, unsigned int *, int, int *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeLargeIndex(dim3, dim3, int, hipStream_t, int *, unsigned int *, unsigned int *, int, int *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeLargeIndex(dim3, dim3, int, hipStream_t, int *, unsigned long *, unsigned long *, int, int *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeAddSmallIndex(dim3, dim3, int, hipStream_t, int *, unsigned int *, unsigned int *, int, int *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeAddLargeIndex(dim3, dim3, int, hipStream_t, int *, unsigned int *, unsigned int *, int, int *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeAddLargeIndex(dim3, dim3, int, hipStream_t, int *, unsigned long *, unsigned long *, int, int *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSelectSmallIndex(dim3, dim3, int, hipStream_t, int *, unsigned int *, unsigned int *, int, int *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSelectLargeIndex(dim3, dim3, int, hipStream_t, int *, unsigned int *, unsigned int *, int, int *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, ptrdiff_t, long);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSelectLargeIndex(dim3, dim3, int, hipStream_t, int *, unsigned long *, unsigned long *, int, int *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, int, ptrdiff_t, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillSmallIndex(dim3, dim3, int, hipStream_t, int *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, ptrdiff_t, long, int);

template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillLargeIndex(dim3, dim3, int, hipStream_t, int *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, ptrdiff_t, long, int);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillLargeIndex(dim3, dim3, int, hipStream_t, int *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, ptrdiff_t, long, int);

// REAL == half

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSmallIndex(dim3, dim3, int, hipStream_t, half *, unsigned int *, unsigned int *, int, half *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeLargeIndex(dim3, dim3, int, hipStream_t, half *, unsigned int *, unsigned int *, int, half *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeLargeIndex(dim3, dim3, int, hipStream_t, half *, unsigned long *, unsigned long *, int, half *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeAddSmallIndex(dim3, dim3, int, hipStream_t, half *, unsigned int *, unsigned int *, int, half *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeAddLargeIndex(dim3, dim3, int, hipStream_t, half *, unsigned int *, unsigned int *, int, half *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeAddLargeIndex(dim3, dim3, int, hipStream_t, half *, unsigned long *, unsigned long *, int, half *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSelectSmallIndex(dim3, dim3, int, hipStream_t, half *, unsigned int *, unsigned int *, int, half *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSelectLargeIndex(dim3, dim3, int, hipStream_t, half *, unsigned int *, unsigned int *, int, half *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, ptrdiff_t, long);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSelectLargeIndex(dim3, dim3, int, hipStream_t, half *, unsigned long *, unsigned long *, int, half *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, int, ptrdiff_t, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillSmallIndex(dim3, dim3, int, hipStream_t, half *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, ptrdiff_t, long, half);

template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillLargeIndex(dim3, dim3, int, hipStream_t, half *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, ptrdiff_t, long, half);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillLargeIndex(dim3, dim3, int, hipStream_t, half *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, ptrdiff_t, long, half);

// REAL == float

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSmallIndex(dim3, dim3, int, hipStream_t, float *, unsigned int *, unsigned int *, int, float *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeLargeIndex(dim3, dim3, int, hipStream_t, float *, unsigned int *, unsigned int *, int, float *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeLargeIndex(dim3, dim3, int, hipStream_t, float *, unsigned long *, unsigned long *, int, float *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeAddSmallIndex(dim3, dim3, int, hipStream_t, float *, unsigned int *, unsigned int *, int, float *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeAddLargeIndex(dim3, dim3, int, hipStream_t, float *, unsigned int *, unsigned int *, int, float *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeAddLargeIndex(dim3, dim3, int, hipStream_t, float *, unsigned long *, unsigned long *, int, float *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSelectSmallIndex(dim3, dim3, int, hipStream_t, float *, unsigned int *, unsigned int *, int, float *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSelectLargeIndex(dim3, dim3, int, hipStream_t, float *, unsigned int *, unsigned int *, int, float *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, ptrdiff_t, long);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSelectLargeIndex(dim3, dim3, int, hipStream_t, float *, unsigned long *, unsigned long *, int, float *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, int, ptrdiff_t, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillSmallIndex(dim3, dim3, int, hipStream_t, float *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, ptrdiff_t, long, float);

template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillLargeIndex(dim3, dim3, int, hipStream_t, float *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, ptrdiff_t, long, float);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillLargeIndex(dim3, dim3, int, hipStream_t, float *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, ptrdiff_t, long, float);

// REAL == double

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSmallIndex(dim3, dim3, int, hipStream_t, double *, unsigned int *, unsigned int *, int, double *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeLargeIndex(dim3, dim3, int, hipStream_t, double *, unsigned int *, unsigned int *, int, double *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeLargeIndex(dim3, dim3, int, hipStream_t, double *, unsigned long *, unsigned long *, int, double *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeAddSmallIndex(dim3, dim3, int, hipStream_t, double *, unsigned int *, unsigned int *, int, double *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeAddLargeIndex(dim3, dim3, int, hipStream_t, double *, unsigned int *, unsigned int *, int, double *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeAddLargeIndex(dim3, dim3, int, hipStream_t, double *, unsigned long *, unsigned long *, int, double *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSelectSmallIndex(dim3, dim3, int, hipStream_t, double *, unsigned int *, unsigned int *, int, double *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSelectLargeIndex(dim3, dim3, int, hipStream_t, double *, unsigned int *, unsigned int *, int, double *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, ptrdiff_t, long);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSelectLargeIndex(dim3, dim3, int, hipStream_t, double *, unsigned long *, unsigned long *, int, double *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, int, ptrdiff_t, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillSmallIndex(dim3, dim3, int, hipStream_t, double *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, ptrdiff_t, long, double);

template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillLargeIndex(dim3, dim3, int, hipStream_t, double *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, ptrdiff_t, long, double);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillLargeIndex(dim3, dim3, int, hipStream_t, double *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, ptrdiff_t, long, double);


// REAL == short

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSmallIndex(dim3, dim3, int, hipStream_t, short *, unsigned int *, unsigned int *, int, short *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeLargeIndex(dim3, dim3, int, hipStream_t, short *, unsigned int *, unsigned int *, int, short *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeLargeIndex(dim3, dim3, int, hipStream_t, short *, unsigned long *, unsigned long *, int, short *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeAddSmallIndex(dim3, dim3, int, hipStream_t, short *, unsigned int *, unsigned int *, int, short *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeAddLargeIndex(dim3, dim3, int, hipStream_t, short *, unsigned int *, unsigned int *, int, short *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeAddLargeIndex(dim3, dim3, int, hipStream_t, short *, unsigned long *, unsigned long *, int, short *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSelectSmallIndex(dim3, dim3, int, hipStream_t, short *, unsigned int *, unsigned int *, int, short *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSelectLargeIndex(dim3, dim3, int, hipStream_t, short *, unsigned int *, unsigned int *, int, short *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, int, ptrdiff_t, ptrdiff_t, long);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
void invokeSelectLargeIndex(dim3, dim3, int, hipStream_t, short *, unsigned long *, unsigned long *, int, short *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, int, ptrdiff_t, ptrdiff_t, long);

template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillSmallIndex(dim3, dim3, int, hipStream_t, short *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, ptrdiff_t, long, short);

template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillLargeIndex(dim3, dim3, int, hipStream_t, short *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, ptrdiff_t, long, short);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillLargeIndex(dim3, dim3, int, hipStream_t, short *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, ptrdiff_t, long, short);

template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillSmallIndex(dim3, dim3, int, hipStream_t, long *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, ptrdiff_t, long, long);

template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillLargeIndex(dim3, dim3, int, hipStream_t, long *, unsigned int *, unsigned int *, int, long *, unsigned int *, unsigned int *, int, int, ptrdiff_t, long, long);

// Indextype = unsigned long
template <typename T, typename IndexType, int DstDim, int IdxDim>
void invokeFillLargeIndex(dim3, dim3, int, hipStream_t, long *, unsigned long *, unsigned long *, int, long *, unsigned long *, unsigned long *, int, int, ptrdiff_t, long, long);





















