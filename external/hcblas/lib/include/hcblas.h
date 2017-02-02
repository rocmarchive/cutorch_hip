#ifndef HCBLAS_H
#define HCBLAS_H

#ifdef __cplusplus
extern "C" {
#endif //(__cplusplus)
//2.2.1. hcblasHandle_t

// The hcblasHandle_t type is a pointer to an opaque structure holding the hcBLAS library context. 
// The hcBLAS library context must be initialized using hcblasCreate() and the returned handle must be 
// passed to all subsequent library function calls. The context should be destroyed at the end using 
// hcblasDestroy().

namespace hc {
  class accelerator;
  class accelerator_view;
};

typedef struct  Hcblaslibrary* hcblasHandle_t;

// 2.2.2. hcblasStatus_t

// The type  hcblasStatus  is used for function status returns. HCBLAS 
// helper functions return status directly, while the status of HCBLAS 
// core functions can be retrieved via  hcblasGetError() . Currently, the 
// following values are defined: 

enum hcblasStatus_t {
  HCBLAS_STATUS_SUCCESS,          // Function succeeds
  HCBLAS_STATUS_NOT_INITIALIZED,  // HCBLAS library not initialized
  HCBLAS_STATUS_ALLOC_FAILED,     // resource allocation failed
  HCBLAS_STATUS_INVALID_VALUE,    // unsupported numerical value was passed to function
  HCBLAS_STATUS_MAPPING_ERROR,    // access to GPU memory space failed
  HCBLAS_STATUS_EXECUTION_FAILED, // GPU program failed to execute
  HCBLAS_STATUS_INTERNAL_ERROR    // an internal HCBLAS operation failed
};

// 2.2.3. hcblasOperation_t

// The hcblasOperation_t type indicates which operation needs to be performed with 
// the dense matrix. Its values correspond to Fortran characters ‘N’ or ‘n’ (non-transpose),
// ‘T’ or ‘t’ (transpose) and ‘C’ or ‘c’ (conjugate transpose) that are often used as parameters 
// to legacy BLAS implementations.

enum hcblasOperation_t {
  HCBLAS_OP_N,  // The Non transpose operation is selected
  HCBLAS_OP_T,  // Transpose operation is selected
  HCBLAS_OP_C   // Conjugate transpose operation is selected
};
  
// 2.2.4. hcComplex

// hcComplex is used in Complex-precision functions
struct float_2_ {
  float x;
  float y;
};

struct double_2_ {
  double x;
  double y;
};

typedef float_2_ hcFloatComplex;
typedef hcFloatComplex hcComplex;
typedef double_2_ hcDoubleComplex;
typedef hcDoubleComplex hcDoubleComplex;

// hcblas Helper functions 

// 1. hcblasCreate()

// This function initializes the HCBLAS library and creates a handle to an opaque structure
// holding the HCBLAS library context.
// Create the handle for use on the specified GPU.

// Return Values
// --------------------------------------------------------------------
// HCBLAS_STATUS_SUCCESS            initialization succeeded
// HCBLAS_STATUS_ALLOC_FAILED       the resources could not be allocated  

hcblasStatus_t hcblasCreate(hcblasHandle_t *handle, hc::accelerator *acc=nullptr);

// 2. hcblasDestory()

// This function releases hardware resources used by the HCBLAS library. 
// This function is usually the last call with a particular handle to the HCBLAS library.

// Return Values
// ---------------------------------------------------------------------
// HCBLAS_STATUS_SUCCESS            the shut down succeeded
// HCBLAS_STATUS_NOT_INITIALIZED    the library was not initialized

hcblasStatus_t hcblasDestroy(hcblasHandle_t *handle);


//hcblasSetAcclView()
//This function sets the hcBLAS library stream, which will be used to execute all subsequent calls to the hcBLAS library functions. If the hcBLAS library stream is not set, all kernels use the defaultNULL stream. In particular, this routine can be used to change the stream between kernel launches and then to reset the hcBLAS library stream back to NULL.
//Return Value 	Meaning

// Returns
// HCBLAS_STATUS_SUCCESS         :the stream was set successfully
// HCBLAS_STATUS_NOT_INITIALIZED :the library was not initialized
hcblasStatus_t hcblasSetAcclView(hcblasHandle_t handle, hc::accelerator_view accl_view, void* streamId = nullptr); 

//hcblasGetAcclView()
// This function gets the hcBLAS library stream, which is being used to execute all calls to the hcBLAS library functions. If the hcBLAS library stream is not set, all kernels use the defaultNULL stream.
// Return Value 	
// HCBLAS_STATUS_SUCCESS : the stream was returned successfully
// HCBLAS_STATUS_NOT_INITIALIZED : the library was not initialized

hcblasStatus_t  hcblasGetAcclView(hcblasHandle_t handle, hc::accelerator_view **accl_view, void **streamId); 

// 3. hcblasSetVector()

// This function copies n elements from a vector x in host memory space to a vector y in GPU memory space. 
// Elements in both vectors are assumed to have a size of elemSize bytes. The storage spacing between 
// consecutive elements is given by incx for the source vector x and by incy for the destination vector y.

// Return Values
// --------------------------------------------------------------------
// HCBLAS_STATUS_SUCCESS            the operation completed successfully
// HCBLAS_STATUS_NOT_INITIALIZED    the library was not initialized
// HCBLAS_STATUS_INVALID_VALUE      the parameters incx, incy, elemSize<=0
// HCBLAS_STATUS_MAPPING_ERROR      there was an error accessing GPU memory

hcblasStatus_t hcblasSetVector(hcblasHandle_t handle, int n, int elemSize, const void *x, int incx, void *y, int incy);

// 4. hcblasGetVector()

// This function copies n elements from a vector x in GPU memory space to a vector y in host memory space. 
// Elements in both vectors are assumed to have a size of elemSize bytes. The storage spacing between 
// consecutive elements is given by incx for the source vector and incy for the destination vector y.

// Return Values
// --------------------------------------------------------------------
// HCBLAS_STATUS_SUCCESS            the operation completed successfully
// HCBLAS_STATUS_NOT_INITIALIZED    the library was not initialized
// HCBLAS_STATUS_INVALID_VALUE      the parameters incx, incy, elemSize<=0
// HCBLAS_STATUS_MAPPING_ERROR      there was an error accessing GPU memory

hcblasStatus_t hcblasGetVector(hcblasHandle_t handle, int n, int elemSize, const void *x, int incx, void *y, int incy);

// 5. hcblasSetMatrix()

// This function copies a tile of rows x cols elements from a matrix A in host memory space to a 
// matrix B in GPU memory space. It is assumed that each element requires storage of elemSize bytes 
// and that both matrices are stored in column-major format, with the leading dimension of the source 
// matrix A and destination matrix B given in lda and ldb, respectively.

// Return Values
// --------------------------------------------------------------------
// HCBLAS_STATUS_SUCCESS            the operation completed successfully
// HCBLAS_STATUS_NOT_INITIALIZED    the library was not initialized
// HCBLAS_STATUS_INVALID_VALUE      the parameters rows, cols<0 or elemSize, lda, ldb<=0
// HCBLAS_STATUS_MAPPING_ERROR      there was an error accessing GPU memory

hcblasStatus_t hcblasSetMatrix(hcblasHandle_t handle, int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);

// 6. hcblasGetMatrix()

// This function copies a tile of rows x cols elements from a matrix A in GPU memory space to 
// a matrix B in host memory space. It is assumed that each element requires storage of elemSize 
// bytes and that both matrices are stored in column-major format, with the leading dimension of 
// the source matrix A and destination matrix B given in lda and ldb, respectively.

// Return Values
// --------------------------------------------------------------------
// HCBLAS_STATUS_SUCCESS            the operation completed successfully
// HCBLAS_STATUS_NOT_INITIALIZED    the library was not initialized
// HCBLAS_STATUS_INVALID_VALUE      the parameters rows, cols<0 or elemSize, lda, ldb<=0
// HCBLAS_STATUS_MAPPING_ERROR      there was an error accessing GPU memory
 
hcblasStatus_t hcblasGetMatrix(hcblasHandle_t handle, int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);


// HCBLAS Level-1 function reference

// Level-1 Basic Linear Algebra Subprograms (BLAS1) functions perform scalar and vector based operations. 
// We will use abbreviations <type> for type and <t> for the corresponding short type to make a more concise 
// and clear presentation of the implemented functions. 
// Unless otherwise specified <type> and <t> have the following meanings:

// <type> 	<t> 	     Meaning
// ---------------------------------------------------
// float     ‘s’ or ‘S’      real single-precision
// double    ‘d’ or ‘D’      real double-precision
// hcComplex ‘c’ or ‘C’      complex single-precision

// The abbreviation Re(.) and Im(.) will stand for the real and imaginary part of a number, respectively.

// 1. hcblas<t>asum() and  hcblas<t>asumBatched()

// This function computes the sum of the absolute values of the elements of vector x.

// Param.       Memory           In/out         Meaning
// -------------------------------------------------------------------------------------
// handle       host             input          handle to the HCBLAS library context.
// n            host             input          number of elements in the vector x.
// x            device           input          <type> vector with elements.
// incx         host             input          stride between consecutive elements of x.
// result       host or device   output         the resulting index, which is 0.0 if n,incx<=0.
// batchCount   host             input          number of pointers contained in input and output arrays.

// Return Values
// --------------------------------------------------------------------
// HCBLAS_STATUS_SUCCESS           the operation completed successfully
// HCBLAS_STATUS_NOT_INITIALIZED   the library was not initialized
// HCBLAS_STATUS_EXECUTION_FAILED  the function failed to launch on the GPU

hcblasStatus_t  hcblasSasum(hcblasHandle_t handle, int n,
                            float           *x, int incx, float  *result);
hcblasStatus_t  hcblasDasum(hcblasHandle_t handle, int n,
                            double          *x, int incx, double *result);
hcblasStatus_t  hcblasSasumBatched(hcblasHandle_t handle, int n,
                            float           *x, int incx, float  *result, int batchCount);
hcblasStatus_t  hcblasDasumBatched(hcblasHandle_t handle, int n,
                            double          *x, int incx, double *result, int batchCount);

// 2. hcblas<t>axpy() and hcblas<t>axpyBatched()

// This function multiplies the vector x by the scalar α and adds it to the vector y overwriting 
// the latest vector with the result.

// Param. 	Memory 	         In/out 	Meaning
// -------------------------------------------------------------------------------------
// handle       host             input          handle to the HCBLAS library context.
// alpha        host or device   input          <type> scalar used for multiplication.
// n            host             input          number of elements in the vector x and y.
// x            device           input          <type> vector with n elements.
// incx         host             input          stride between consecutive elements of x.
// y            device           in/out         <type> vector with n elements.
// incy         host             input          stride between consecutive elements of y.
// batchCount   host             input          number of pointers contained in input and output arrays.

// Return Values
// --------------------------------------------------------------------
// HCBLAS_STATUS_SUCCESS           the operation completed successfully
// HCBLAS_STATUS_NOT_INITIALIZED   the library was not initialized
// HCBLAS_STATUS_EXECUTION_FAILED  the function failed to launch on the GPU

hcblasStatus_t hcblasSaxpy(hcblasHandle_t handle, int n,
                           const float           *alpha,
                           const float           *x, int incx,
                           float                 *y, int incy);
hcblasStatus_t hcblasDaxpy(hcblasHandle_t handle, int n,
                           const double           *alpha,
                           const double           *x, int incx,
                           double                 *y, int incy);
hcblasStatus_t hcblasSaxpyBatched(hcblasHandle_t handle, int n,
                           const float           *alpha,
                           const float           *x, int incx,
                           float                 *y, int incy, int batchCount);


// 3. hcblas<t>copy() and and hcblas<t>copyBatched()

// This function copies the vector x into the vector y.

// Param.       Memory           In/out         Meaning
// -------------------------------------------------------------------------------------
// handle       host             input          handle to the HCBLAS library context.
// n            host             input          number of elements in the vector x and y.
// x            device           input          <type> vector with n elements.
// incx         host             input          stride between consecutive elements of x.
// y            device           in/out         <type> vector with n elements.
// incy         host             input          stride between consecutive elements of y.
// batchCount   host             input          number of pointers contained in input and output arrays.

// Return Values
// --------------------------------------------------------------------
// HCBLAS_STATUS_SUCCESS           the operation completed successfully
// HCBLAS_STATUS_NOT_INITIALIZED   the library was not initialized
// HCBLAS_STATUS_EXECUTION_FAILED  the function failed to launch on the GPU

hcblasStatus_t hcblasScopy(hcblasHandle_t handle, int n,
                           const float           *x, int incx,
                           float                 *y, int incy);
hcblasStatus_t hcblasDcopy(hcblasHandle_t handle, int n,
                           const double          *x, int incx,
                           double                *y, int incy);
hcblasStatus_t hcblasScopyBatched(hcblasHandle_t handle, int n,
                           const float           *x, int incx,
                           float                 *y, int incy, int batchCount);
hcblasStatus_t hcblasDcopyBatched(hcblasHandle_t handle, int n,
                           const double          *x, int incx,
                           double                *y, int incy, int batchCount);


// 4. hcblas<t>dot() and hcblas<t>dotBatched()

// This function computes the dot product of vectors x and y.

// Param.       Memory           In/out         Meaning
// -------------------------------------------------------------------------------------
// handle       host             input          handle to the HCBLAS library context.
// n            host             input          number of elements in the vector x and y.
// x            device           input          <type> vector with n elements.
// incx         host             input          stride between consecutive elements of x.
// y            device           in/out         <type> vector with n elements.
// incy         host             input          stride between consecutive elements of y.
// result       host or device   output         the resulting dot product, which is 0.0 if n<=0.
// batchCount   host             input          number of pointers contained in input and output arrays.

// Return Values
// --------------------------------------------------------------------
// HCBLAS_STATUS_SUCCESS           the operation completed successfully
// HCBLAS_STATUS_NOT_INITIALIZED   the library was not initialized
// HCBLAS_STATUS_EXECUTION_FAILED  the function failed to launch on the GPU

hcblasStatus_t hcblasSdot (hcblasHandle_t handle, int n,
                           const float           *x, int incx,
                           const float           *y, int incy,
                           float           *result);
hcblasStatus_t hcblasDdot (hcblasHandle_t handle, int n,
                           const double          *x, int incx,
                           const double          *y, int incy,
                           double          *result);
hcblasStatus_t hcblasSdotBatched (hcblasHandle_t handle, int n,
                           const float           *x, int incx,
                           const float           *y, int incy,
                           float           *result, int batchCount);
hcblasStatus_t hcblasDdotBatched (hcblasHandle_t handle, int n,
                           const double          *x, int incx,
                           const double          *y, int incy,
                           double          *result, int batchCount);


// 5. hcblas<t>scal() and hcblas<t>scalBatched()

// This function scales the vector x by the scalar α and overwrites it with the result.

// Param.       Memory           In/out         Meaning
// -------------------------------------------------------------------------------------
// handle       host             input          handle to the HCBLAS library context.
// alpha        host or device   input          <type> scalar used for multiplication.
// n            host             input          number of elements in the vector x and y.
// x            device           input          <type> vector with n elements.
// incx         host             input          stride between consecutive elements of x.
// batchCount   host             input          number of pointers contained in input and output arrays.

// Return Values
// --------------------------------------------------------------------
// HCBLAS_STATUS_SUCCESS           the operation completed successfully
// HCBLAS_STATUS_NOT_INITIALIZED   the library was not initialized
// HCBLAS_STATUS_EXECUTION_FAILED  the function failed to launch on the GPU

hcblasStatus_t  hcblasSscal(hcblasHandle_t handle, int n,
                            const float           *alpha,
                            float           *x, int incx);
hcblasStatus_t  hcblasDscal(hcblasHandle_t handle, int n,
                            const double          *alpha,
                            double          *x, int incx);
hcblasStatus_t  hcblasSscalBatched(hcblasHandle_t handle, int n,
                            const float           *alpha,
                            float           *x, int incx, int batchCount);
hcblasStatus_t  hcblasDscalBatched(hcblasHandle_t handle, int n,
                            const double          *alpha,
                            double          *x, int incx, int batchCount);

 
// HCBLAS Level-2 Function Reference

// The Level-2 Basic Linear Algebra Subprograms (BLAS2) functions perform matrix-vector operations.
// Unless otherwise specified <type> and <t> have the following meanings:

// <type>       <t>          Meaning
// ---------------------------------------------------
// float     ‘s’ or ‘S’      real single-precision
// double    ‘d’ or ‘D’      real double-precision
// hcComplex ‘c’ or ‘C’      complex single-precision

// 1. hcblas<t>gemv() and and hcblas<t>gemvBatched()

// This function performs the matrix-vector multiplication
// y = α op ( A ) x + β y
// where A is a m × n matrix stored in column-major format, x and y are vectors, and α and β are scalars. Also, for matrix A
// op ( A ) = A             if transa == HCBLAS_OP_N 
//            A^T           if transa == HCBLAS_OP_T   
//            A^H           if transa == HCBLAS_OP_C

// Param.       Memory           In/out         Meaning
// -------------------------------------------------------------------------------------
// handle       host             input          handle to the HCBLAS library context.
// trans        host             input          operation op(A) that is non- or (conj.) transpose.
// m            host             input          number of rows of matrix A.
// n            host             input          number of columns of matrix A.
// alpha        host or device   input          <type> scalar used for multiplication.
// A            device           input          <type> array of dimension lda x n with lda >= max(1,m)
//                                              if transa==HCBLAS_OP_N and lda x m with lda >= max(1,n) otherwise.
// lda          host             input          leading dimension of two-dimensional array used to store matrix A.
// x            device           input          <type> vector with n elements if transa==HCBLAS_OP_N and m elements otherwise.
// incx         host             input          stride between consecutive elements of x.
// beta         host or device   input          <type> scalar used for multiplication, if beta==0 
//                                              then y does not have to be a valid input.
// y            device           in/out         <type> vector with m elements if transa==HCBLAS_OP_N and n elements otherwise.
// incy         host             input          stride between consecutive elements of y.
// batchCount   host             input          number of pointers contained in input and output arrays.

// Return Values
// --------------------------------------------------------------------
// HCBLAS_STATUS_SUCCESS           the operation completed successfully
// HCBLAS_STATUS_NOT_INITIALIZED   the library was not initialized
// HCBLAS_STATUS_INVALID_VALUE     the parameters m,n<0 or incx,incy=0
// HCBLAS_STATUS_EXECUTION_FAILED  the function failed to launch on the GPU

hcblasStatus_t hcblasSgemv(hcblasHandle_t handle, hcblasOperation_t trans,
                           int m, int n,
                           const float           *alpha,
                           float           *A, int lda,
                           float           *x, int incx,
                           const float           *beta,
                           float           *y, int incy);
hcblasStatus_t hcblasSgemvBatched(hcblasHandle_t handle, hcblasOperation_t trans,
                           int m, int n,
                           const float           *alpha,
                           float           *A, int lda,
                           float           *x, int incx,
                           const float           *beta,
                           float           *y, int incy, int batchCount);

hcblasStatus_t hcblasDgemv(hcblasHandle_t handle, hcblasOperation_t trans,
                           int m, int n,
                           const double           *alpha,
                           double           *A, int lda,
                           double           *x, int incx,
                           const double           *beta,
                           double           *y, int incy);
hcblasStatus_t hcblasDgemvBatched(hcblasHandle_t handle, hcblasOperation_t trans,
                           int m, int n,
                           const double           *alpha,
                           double           *A, int lda,
                           double           *x, int incx,
                           const double           *beta,
                           double           *y, int incy, int batchCount);


// 2. hcblas<t>ger() and hcblas<t>gerBatched()

// This function performs the rank-1 update
// A = α x y T + A if ger(),geru() is called 
//     α x y H + A if gerc() is called
// where A is a m × n matrix stored in column-major format, x and y are vectors, and α is a scalar.

// Param.       Memory           In/out         Meaning
// -------------------------------------------------------------------------------------
// handle       host             input          handle to the HCBLAS library context.
// m            host             input          number of rows of matrix A.
// n            host             input          number of columns of matrix A.
// alpha        host or device   input          <type> scalar used for multiplication.
// x            device           input          <type> vector with m elements.
// incx         host             input          stride between consecutive elements of x.
// y            device           in/out         <type> vector with n elements.
// incy         host             input          stride between consecutive elements of y.
// A            device           in/out         <type> array of dimension lda x n with lda >= max(1,m).
// lda          host             input          leading dimension of two-dimensional array used to store matrix A.
// batchCount   host             input          number of pointers contained in input and output arrays.

// Return Values
// --------------------------------------------------------------------
// HCBLAS_STATUS_SUCCESS           the operation completed successfully
// HCBLAS_STATUS_NOT_INITIALIZED   the library was not initialized
// HCBLAS_STATUS_INVALID_VALUE     the parameters m,n<0 or incx,incy=0
// HCBLAS_STATUS_EXECUTION_FAILED  the function failed to launch on the GPU

hcblasStatus_t  hcblasSger(hcblasHandle_t handle, int m, int n,
                           const float           *alpha,
                           const float           *x, int incx,
                           const float           *y, int incy,
                           float           *A, int lda);
hcblasStatus_t  hcblasSgerBatched(hcblasHandle_t handle, int m, int n,
                           const float           *alpha,
                           const float           *x, int incx,
                           const float           *y, int incy,
                           float           *A, int lda, int batchCount);

// HCBLAS Level-3 Function Reference

// The Level-3 Basic Linear Algebra Subprograms (BLAS3) functions perform matrix-matrix operations.
// Unless otherwise specified <type> and <t> have the following meanings:

// <type>       <t>          Meaning
// ---------------------------------------------------
// float     ‘s’ or ‘S’      real single-precision
// double    ‘d’ or ‘D’      real double-precision
// hcComplex ‘c’ or ‘C’      complex single-precision

// 1. hcblas<t>gemm()

// This function performs the matrix-matrix multiplication
// C = α op ( A ) op ( B ) + β C
// where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions 
// op ( A ) m × k , op ( B ) k × n and C m × n , respectively. Also, for matrix A
// op ( A ) = A   if  transa == HCBLAS_OP_N 
//            A^T if  transa == HCBLAS_OP_T 
//            A^H if  transa == HCBLAS_OP_C
// and op ( B ) is defined similarly for matrix B .

// Param.       Memory           In/out         Meaning
// -------------------------------------------------------------------------------------
// handle       host             input          handle to the HCBLAS library context.
// transa       host             input          operation op(A) that is non- or (conj.) transpose.
// transb       host             input          operation op(B) that is non- or (conj.) transpose.
// m            host             input          number of rows of matrix op(A) and C.
// n            host             input          number of rows of matrix op(B) and C.
// k            host             input          number of columns of op(A) and rows of op(B).
// alpha        host or device   input          <type> scalar used for multiplication.
// A            device           input          <type> array of dimensions lda x k with lda>=max(1,m) 
//                                              if transa == HCBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.
// lda          host             input          leading dimension of two-dimensional array used to store the matrix A.
// B            device           input          <type> array of dimension ldb x n with ldb>=max(1,k) 
//                                              if transa == HCBLAS_OP_N and ldb x k with ldb>=max(1,n) otherwise.
// ldb          host             input          leading dimension of two-dimensional array used to store matrix B.
// beta         host or device   input          <type> scalar used for multiplication. If beta==0, C does not have to be a valid input.
// C            device           in/out         <type> array of dimensions ldc x n with ldc>=max(1,m).
// ldc          host             input          leading dimension of a two-dimensional array used to store the matrix C.

// Return Values
// --------------------------------------------------------------------
// HCBLAS_STATUS_SUCCESS           the operation completed successfully
// HCBLAS_STATUS_NOT_INITIALIZED   the library was not initialized
// HCBLAS_STATUS_INVALID_VALUE     the parameters m,n,k<0 
// HCBLAS_STATUS_EXECUTION_FAILED  the function failed to launch on the GPU

hcblasStatus_t hcblasSgemm(hcblasHandle_t handle,
                           hcblasOperation_t transa, hcblasOperation_t transb,
                           int m, int n, int k,
                           const float           *alpha,
                           float           *A, int lda,
                           float           *B, int ldb,
                           const float           *beta,
                           float           *C, int ldc);

hcblasStatus_t hcblasDgemm(hcblasHandle_t handle,
                           hcblasOperation_t transa, hcblasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           double           *A, int lda,
                           double           *B, int ldb,
                           const double           *beta,
                           double           *C, int ldc);

hcblasStatus_t hcblasCgemm(hcblasHandle_t handle,
                           hcblasOperation_t transa, hcblasOperation_t transb,
                           int m, int n, int k,
                           const hcComplex       *alpha,
                           hcComplex       *A, int lda,
                           hcComplex       *B, int ldb,
                           const hcComplex       *beta,
                           hcComplex       *C, int ldc);

hcblasStatus_t hcblasZgemm(hcblasHandle_t handle,
                           hcblasOperation_t transa, hcblasOperation_t transb,
                           int m, int n, int k,
                           const hcDoubleComplex       *alpha,
                           hcDoubleComplex       *A, int lda,
                           hcDoubleComplex       *B, int ldb,
                           const hcDoubleComplex       *beta,
                           hcDoubleComplex       *C, int ldc);

// 2. hcblas<t>gemmBatched()

// This function performs the matrix-matrix multiplications of an array of matrices.
// C [ i ] = α op ( A [ i ] ) op ( B [ i ] ) + β C [ i ] ,  for i  ∈ [ 0 , batchCount − 1 ]
// where α and β are scalars, and A , B and C are arrays of pointers to matrices stored in 
// column-major format with dimensions op ( A [ i ] ) m × k , op ( B [ i ] ) k × n and C [ i ] m × n , 
// respectively. Also, for matrix A
// op ( A ) = A   if  transa == HCBLAS_OP_N 
//            A^T if  transa == HCBLAS_OP_T 
//            A^H if  transa == HCBLAS_OP_C
// and op ( B [ i ] ) is defined similarly for matrix B [ i ] .

// This function is intended to be used for matrices of small sizes where the launch overhead is a significant factor.

// Param.       Memory           In/out         Meaning
// -------------------------------------------------------------------------------------
// handle       host             input          handle to the HCBLAS library context.
// transa       host             input          operation op(A) that is non- or (conj.) transpose.
// transb       host             input          operation op(B) that is non- or (conj.) transpose.
// m            host             input          number of rows of matrix op(A) and C.
// n            host             input          number of rows of matrix op(B) and C.
// k            host             input          number of columns of op(A) and rows of op(B).
// alpha        host or device   input          <type> scalar used for multiplication.
// Aarray       device           input          array of pointers to <type> array, with each array of dim. lda x k with lda>=max(1,m)
//                                              if transa == HCBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.
// lda          host             input          leading dimension of two-dimensional array used to store the matrix A[i].
// Barray       device           input          array of pointers to <type> array, with each array of dim.ldb x n with ldb>=max(1,k)
//                                              if transa == HCBLAS_OP_N and ldb x k with ldb>=max(1,n) otherwise.
// ldb          host             input          leading dimension of two-dimensional array used to store matrix B[i].
// beta         host or device   input          <type> scalar used for multiplication. If beta==0, C does not have to be a valid input.
// Carray       device           in/out         array of pointers to <type> array, with each array of dim.ldc x n with ldc>=max(1,m).
// ldc          host             input          leading dimension of a two-dimensional array used to store the matrix C[i].
// batchCount   host             input          number of pointers contained in Aarray, Barray and Carray.

// Return Values
// --------------------------------------------------------------------
// HCBLAS_STATUS_SUCCESS           the operation completed successfully
// HCBLAS_STATUS_NOT_INITIALIZED   the library was not initialized
// HCBLAS_STATUS_INVALID_VALUE     the parameters m,n,k,batchCount<0
// HCBLAS_STATUS_EXECUTION_FAILED  the function failed to launch on the GPU

hcblasStatus_t hcblasSgemmBatched(hcblasHandle_t handle,
                                  hcblasOperation_t transa, hcblasOperation_t transb,
                                  int m, int n, int k,
                                  const float           *alpha,
                                  float           *Aarray, int lda,
                                  float           *Barray, int ldb,
                                  const float           *beta,
                                  float           *Carray, int ldc, int batchCount);

hcblasStatus_t hcblasCgemmBatched(hcblasHandle_t handle,
                                  hcblasOperation_t transa, hcblasOperation_t transb,
                                  int m, int n, int k,
                                  const hcComplex       *alpha,
                                  hcComplex       *Aarray, int lda,
                                  hcComplex       *Barray, int ldb,
                                  const hcComplex       *beta,
                                  hcComplex       *Carray, int ldc, int batchCount);

hcblasStatus_t hcblasDgemmBatched(hcblasHandle_t handle,
                                  hcblasOperation_t transa, hcblasOperation_t transb,
                                  int m, int n, int k,
                                  const double           *alpha,
                                  double           *Aarray, int lda,
                                  double           *Barray, int ldb,
                                  const double           *beta,
                                  double           *Carray, int ldc, int batchCount);

hcblasStatus_t hcblasZgemmBatched(hcblasHandle_t handle,
                                  hcblasOperation_t transa, hcblasOperation_t transb,
                                  int m, int n, int k,
                                  const hcDoubleComplex       *alpha,
                                  hcDoubleComplex       *Aarray, int lda,
                                  hcDoubleComplex       *Barray, int ldb,
                                  const hcDoubleComplex       *beta,
                                  hcDoubleComplex       *Carray, int ldc, int batchCount);
#ifdef __cplusplus
}
#endif //(__cplusplus)
#endif //(HCBLAS_H)
