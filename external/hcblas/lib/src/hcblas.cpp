#include "hcblaslib.h"
#include "hcblas.h"
// hcblas Helper functions 

// 1. hcblasCreate()

// This function initializes the HCBLAS library and creates a handle to an opaque structure
// holding the HCBLAS library context.

// Return Values
// --------------------------------------------------------------------
// HCBLAS_STATUS_SUCCESS            initialization succeeded
// HCBLAS_STATUS_ALLOC_FAILED       the resources could not be allocated  

hcblasStatus_t hcblasCreate(hcblasHandle_t *handle, hc::accelerator *acc) {
  if (handle == NULL) { 
    handle = new hcblasHandle_t();
  }
  *handle = new Hcblaslibrary(acc);

  if(*handle == NULL) {
    return HCBLAS_STATUS_ALLOC_FAILED;
  }
  return HCBLAS_STATUS_SUCCESS;  
}

// 2. hcblasDestory()

// This function releases hardware resources used by the HCBLAS library.
// This function is usually the last call with a particular handle to the HCBLAS library.

// Return Values
// ---------------------------------------------------------------------
// HCBLAS_STATUS_SUCCESS            the shut down succeeded
// HCBLAS_STATUS_NOT_INITIALIZED    the library was not initialized

hcblasStatus_t hcblasDestroy(hcblasHandle_t *handle){
  if(handle == nullptr || *handle == nullptr || (*handle)->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;
  delete *handle;
  *handle = nullptr;
  handle = nullptr;
  return HCBLAS_STATUS_SUCCESS;
}


//hcblasSetAcclView()
//This function sets the hcBLAS library stream, which will be used to execute all subsequent calls to the hcBLAS library functions. If the hcBLAS library stream is not set, all kernels use the defaultNULL stream. In particular, this routine can be used to change the stream between kernel launches and then to reset the hcBLAS library stream back to NULL.
//Return Value 	Meaning

// Returns
// HCBLAS_STATUS_SUCCESS         :the stream was set successfully
// HCBLAS_STATUS_NOT_INITIALIZED :the library was not initialized
hcblasStatus_t hcblasSetAcclView(hcblasHandle_t handle, hc::accelerator_view accl_view, void* stream) {
  if (handle == nullptr || handle->initialized == false) {
    return HCBLAS_STATUS_NOT_INITIALIZED;    
  }
  handle->currentAcclView = accl_view;
  handle->currentStream = stream;
  return HCBLAS_STATUS_SUCCESS;
} 

//hcblasGetAcclView()
// This function gets the hcBLAS library stream, which is being used to execute all calls to the hcBLAS library functions. If the hcBLAS library stream is not set, all kernels use the defaultNULL stream.
// Return Value 	
// HCBLAS_STATUS_SUCCESS : the stream was returned successfully
// HCBLAS_STATUS_NOT_INITIALIZED : the library was not initialized

hcblasStatus_t  hcblasGetAcclView(hcblasHandle_t handle, hc::accelerator_view **accl_view, void** stream) {
  if (handle == nullptr) {
    return HCBLAS_STATUS_NOT_INITIALIZED;    
  }
  *accl_view = &handle->currentAcclView;
  stream = &(handle->currentStream);
  return HCBLAS_STATUS_SUCCESS;
}



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

hcblasStatus_t hcblasSetVector(hcblasHandle_t handle, int n, int elemSize, const void *x, int incx, void *y, int incy) {
  std::vector<accelerator> accs = accelerator::get_all();
  if(accs.size() == 0) {
    std::wcout << "There is no acclerator!\n";
    // Since this case is to test on GPU device, skip if there is CPU only
    return HCBLAS_STATUS_MAPPING_ERROR;
  }
  assert(accs.size() && "Number of Accelerators == 0!");

  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;

  if( incx <= 0 || incy <=0 || elemSize <=0 ) {
    return HCBLAS_STATUS_INVALID_VALUE;
  }
 
  handle->currentAcclView.copy(x, y, elemSize * n);   
  return HCBLAS_STATUS_SUCCESS;
}

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

hcblasStatus_t hcblasGetVector(hcblasHandle_t handle, int n, int elemSize, const void *x, int incx, void *y, int incy) {
 std::vector<accelerator> accs = accelerator::get_all();
  if(accs.size() == 0) {
    std::wcout << "There is no acclerator!\n";
    // Since this case is to test on GPU device, skip if there is CPU only
    return HCBLAS_STATUS_MAPPING_ERROR;
  }
  assert(accs.size() && "Number of Accelerators == 0!");

  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;

  if( incx <= 0 || incy <=0 || elemSize <=0 ) {
    return HCBLAS_STATUS_INVALID_VALUE;
  }

  handle->currentAcclView.copy(x, y, elemSize * n);
  return HCBLAS_STATUS_SUCCESS;
}

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

hcblasStatus_t hcblasSetMatrix(hcblasHandle_t handle, int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb) {
  std::vector<accelerator> accs = accelerator::get_all();
  if(accs.size() == 0) {
    std::wcout << "There is no acclerator!\n";
    // Since this case is to test on GPU device, skip if there is CPU only
    return HCBLAS_STATUS_MAPPING_ERROR;
  }
  assert(accs.size() && "Number of Accelerators == 0!");

  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;

  if( rows < 0 || cols < 0 ||  lda <=0 || ldb <=0 || elemSize <=0 ) {
    return HCBLAS_STATUS_INVALID_VALUE;
  }

  handle->currentAcclView.copy(A, B, elemSize * rows * cols);
 
  return HCBLAS_STATUS_SUCCESS;
}

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

hcblasStatus_t hcblasGetMatrix(hcblasHandle_t handle, int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb) {
  std::vector<accelerator> accs = accelerator::get_all();
  if(accs.size() == 0) {
    std::wcout << "There is no acclerator!\n";
    // Since this case is to test on GPU device, skip if there is CPU only
    return HCBLAS_STATUS_MAPPING_ERROR;
  }
  assert(accs.size() && "Number of Accelerators == 0!");

  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;

  if( rows < 0 || cols < 0 ||  lda <=0 || ldb <=0 || elemSize <=0 ) {
    return HCBLAS_STATUS_INVALID_VALUE;
  }

  handle->currentAcclView.copy(A, B, elemSize * rows * cols);

  return HCBLAS_STATUS_SUCCESS;
}

// HCBLAS Level-1 function reference

// Level-1 Basic Linear Algebra Subprograms (BLAS1) functions perform scalar and vector based operations. 
// We will use abbreviations <type> for type and <t> for the corresponding short type to make a more concise 
// and clear presentation of the implemented functions. 
// Unless otherwise specified <type> and <t> have the following meanings:

// <type>       <t>          Meaning
// ---------------------------------------------------
// float     ‘s’ or ‘S’      real single-precision
// double    ‘d’ or ‘D’      real double-precision
// hcComplex ‘c’ or ‘C’      complex single-precision

// The abbreviation Re(.) and Im(.) will stand for the real and imaginary part of a number, respectively.

// 1. hcblas<t>asum() and hcblas<t>asumBatched()

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

hcblasStatus_t  hcblasSasum(hcblasHandle_t handle, const int n,
                            float           *x, const int incx, float  *result) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;
  long xOffset = 0;
  hcblasStatus status;
  status = handle->hcblas_sasum(handle->currentAcclView, n, x, incx, xOffset, result);
  if(status == HCBLAS_SUCCEEDS) 
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

hcblasStatus_t  hcblasSasumBatched(hcblasHandle_t handle, const int n,
                                   float           *x, const int incx, float  *result, int batchCount) {

  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;
  long xOffset = 0;
  long X_batchOffset = n;
  hcblasStatus status;
  status= handle->hcblas_sasum(handle->currentAcclView, n, x, incx, xOffset, result, X_batchOffset, batchCount);
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

hcblasStatus_t  hcblasDasum(hcblasHandle_t handle, const int n,
                            double          *x, const int incx, double *result) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;
  long xOffset = 0;
  hcblasStatus status;
  status = handle->hcblas_dasum(handle->currentAcclView, n, x, incx, xOffset, result);
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

hcblasStatus_t  hcblasDasumBatched(hcblasHandle_t handle, const int n,
                                   double          *x, const int incx, double *result, int batchCount) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;
  long xOffset = 0;
  long X_batchOffset = n;
  hcblasStatus status;
  status = handle->hcblas_dasum(handle->currentAcclView, n, x, incx, xOffset, result, X_batchOffset, batchCount);
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

// 2. hcblas<t>axpy() and hcblas<t>axpyBatched()

// This function multiplies the vector x by the scalar α and adds it to the vector y overwriting 
// the latest vector with the result.

// Param.       Memory           In/out         Meaning
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
                           float                 *y, int incy) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;
  long xOffset = 0;
  long yOffset = 0;
  hcblasStatus status;
  status = handle->hcblas_saxpy(handle->currentAcclView, n, *alpha, x, incx, y, incy , xOffset, yOffset);
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

hcblasStatus_t hcblasDaxpy(hcblasHandle_t handle, int n,
                           const double           *alpha,
                           const double           *x, int incx,
                           double                 *y, int incy) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;
  long xOffset = 0;
  long yOffset = 0;
  hcblasStatus status;
  status = handle->hcblas_daxpy(handle->currentAcclView, n, *alpha, x, incx, y, incy , xOffset, yOffset);
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

hcblasStatus_t hcblasSaxpyBatched(hcblasHandle_t handle, int n,
                                  const float           *alpha,
                                  const float           *x, int incx,
                                  float                 *y, int incy, int batchCount) {
  if(handle == nullptr  || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;
  long xOffset = 0;
  long yOffset = 0;
  long X_batchOffset = n;
  long Y_batchOffset = n;
  hcblasStatus status;
  status= handle->hcblas_saxpy(handle->currentAcclView, n, *alpha, x, incx, X_batchOffset, y, incy, Y_batchOffset, xOffset, yOffset, batchCount);
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

// 3. hcblas<t>copy() and hcblas<t>copyBatched()

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
                           float                 *y, int incy) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;
  long xOffset = 0;
  long yOffset = 0;
  hcblasStatus status;
  status = handle->hcblas_scopy(handle->currentAcclView, n, x, incx, xOffset, y, incy, yOffset);
  if(status == HCBLAS_SUCCEEDS) 
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

hcblasStatus_t hcblasScopyBatched(hcblasHandle_t handle, int n,
                                  const float           *x, int incx,
                                  float                 *y, int incy, int batchCount) {
  if(handle == nullptr  || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;
  long xOffset = 0;
  long yOffset = 0;
  long X_batchOffset = n;
  long Y_batchOffset = n;
  hcblasStatus status;
  status = handle->hcblas_scopy(handle->currentAcclView, n, x, incx, xOffset, y, incy, yOffset, X_batchOffset, Y_batchOffset, batchCount);
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

hcblasStatus_t hcblasDcopy(hcblasHandle_t handle, int n,
                           const double          *x, int incx,
                           double                *y, int incy) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;
  long xOffset = 0;
  long yOffset = 0;
  hcblasStatus status;
  status = handle->hcblas_dcopy(handle->currentAcclView, n, x, incx, xOffset, y, incy, yOffset);
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

hcblasStatus_t hcblasDcopyBatched(hcblasHandle_t handle, int n,
                                  const double          *x, int incx,
                                  double                *y, int incy, int batchCount) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;
  long xOffset = 0;
  long yOffset = 0;
  long X_batchOffset = n;
  long Y_batchOffset = n;
  hcblasStatus status;
  status = handle->hcblas_dcopy(handle->currentAcclView, n, x, incx, xOffset, y, incy, yOffset, X_batchOffset, Y_batchOffset, batchCount);
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

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
                           float           *result) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;
  long xOffset = 0;
  long yOffset = 0;
  hcblasStatus status;
  status = handle->hcblas_sdot(handle->currentAcclView, n, x, incx, xOffset, y, incy, yOffset, *result);
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

hcblasStatus_t hcblasSdotBatched (hcblasHandle_t handle, int n,
                                  const float           *x, int incx,
                                  const float           *y, int incy,
                                  float           *result, int batchCount) {
  if(handle == nullptr  || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;
  long xOffset = 0;
  long yOffset = 0;
  long X_batchOffset = n;
  long Y_batchOffset = n;
  hcblasStatus status;
  status = handle->hcblas_sdot(handle->currentAcclView, n, x, incx, xOffset, y, incy, yOffset, *result, X_batchOffset, Y_batchOffset, batchCount);
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

hcblasStatus_t hcblasDdot (hcblasHandle_t handle, int n,
                           const double          *x, int incx,
                           const double          *y, int incy,
                           double          *result) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;
  long xOffset = 0;
  long yOffset = 0;
  hcblasStatus status;
  status = handle->hcblas_ddot(handle->currentAcclView, n, x, incx, xOffset, y, incy, yOffset, *result);
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

hcblasStatus_t hcblasDdotBatched (hcblasHandle_t handle, int n,
                                  const double          *x, int incx,
                                  const double          *y, int incy,
                                  double          *result, int batchCount) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;
  long xOffset = 0;
  long yOffset = 0;
  long X_batchOffset = n;
  long Y_batchOffset = n;
  hcblasStatus status;
  status = handle->hcblas_ddot(handle->currentAcclView, n, x, incx, xOffset, y, incy, yOffset, *result, X_batchOffset, Y_batchOffset, batchCount);
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

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
                            float           *x, int incx) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;
  long xOffset = 0;
  hcblasStatus status;
  status = handle->hcblas_sscal(handle->currentAcclView, n, *alpha, x, incx, xOffset);
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

hcblasStatus_t  hcblasSscalBatched(hcblasHandle_t handle, int n,
                                   const float           *alpha,
                                   float           *x, int incx, int batchCount) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;
  long xOffset = 0;
  long X_batchOffset = n;
  hcblasStatus status;
  status = handle->hcblas_sscal(handle->currentAcclView, n, *alpha, x, incx, xOffset, X_batchOffset, batchCount);
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

hcblasStatus_t  hcblasDscal(hcblasHandle_t handle, int n,
                            const double          *alpha,
                            double          *x, int incx) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;
  long xOffset = 0;
  hcblasStatus status;
  status = handle->hcblas_dscal(handle->currentAcclView, n, *alpha, x, incx, xOffset);
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

hcblasStatus_t  hcblasDscalBatched(hcblasHandle_t handle, int n,
                                   const double          *alpha,
                                   double          *x, int incx, int batchCount) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;
  long xOffset = 0;
  long X_batchOffset = n;
  hcblasStatus status;
  status = handle->hcblas_dscal(handle->currentAcclView, n, *alpha, x, incx, xOffset, X_batchOffset, batchCount);
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

// HCBLAS Level-2 Function Reference

// The Level-2 Basic Linear Algebra Subprograms (BLAS2) functions perform matrix-vector operations.
// Unless otherwise specified <type> and <t> have the following meanings:

// <type>       <t>          Meaning
// ---------------------------------------------------
// float     ‘s’ or ‘S’      real single-precision
// double    ‘d’ or ‘D’      real double-precision
// hcComplex ‘c’ or ‘C’      complex single-precision

// 1. hcblas<t>gemv() and hcblas<t>gemvBatched()

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
                           float           *y, int incy) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;

  if(m < 0 || n < 0 || incx == 0 || incy == 0)
    return HCBLAS_STATUS_INVALID_VALUE;

  long aOffset = 0;
  long xOffset = 0;
  long yOffset = 0;
  hcblasStatus status;
  hcblasTranspose transA;
  transA =  (trans == HCBLAS_OP_N)? NoTrans : Trans;
  status =  handle->hcblas_sgemv(handle->currentAcclView, handle->Order, transA, m, n, *alpha, A, aOffset, lda, x, xOffset, incx, *beta, y, yOffset, incy);
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

hcblasStatus_t hcblasSgemvBatched(hcblasHandle_t handle, hcblasOperation_t trans,
                                  int m, int n,
                                  const float           *alpha,
                                  float           *A, int lda,
                                  float           *x, int incx,
                                  const float           *beta,
                                  float           *y, int incy, int batchCount) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;

  if(m < 0 || n < 0 || incx == 0 || incy == 0)
    return HCBLAS_STATUS_INVALID_VALUE;

  long aOffset = 0;
  long xOffset = 0;
  long yOffset = 0;
  hcblasStatus status;
  hcblasTranspose transA;
  transA =  (trans == HCBLAS_OP_N)? NoTrans : Trans;
  int row, col;
  if(transA == NoTrans){
        row = n;
        col = m;
  }
  else{
        row = m;
        col = n;
  }
  long X_batchOffset = row;
  long Y_batchOffset = col;
  long A_batchOffset = row * col;
  status =  handle->hcblas_sgemv(handle->currentAcclView, handle->Order, transA, m, n, *alpha, A, aOffset, A_batchOffset, lda, x, xOffset, X_batchOffset, incx, *beta, y, yOffset, Y_batchOffset, incy, batchCount);
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

//Dgemv routines
hcblasStatus_t hcblasDgemv(hcblasHandle_t handle, hcblasOperation_t trans,
                           int m, int n,
                           const double           *alpha,
                           double           *A, int lda,
                           double           *x, int incx,
                           const double           *beta,
                           double           *y, int incy) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;

  if(m < 0 || n < 0 || incx == 0 || incy == 0)
    return HCBLAS_STATUS_INVALID_VALUE;

  long aOffset = 0;
  long xOffset = 0;
  long yOffset = 0;
  hcblasStatus status;
  hcblasTranspose transA;
  transA =  (trans == HCBLAS_OP_N)? NoTrans : Trans;
  status =  handle->hcblas_dgemv(handle->currentAcclView, handle->Order, transA, m, n, *alpha, A, aOffset, lda, x, xOffset, incx, *beta, y, yOffset, incy);
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

hcblasStatus_t hcblasDgemvBatched(hcblasHandle_t handle, hcblasOperation_t trans,
                                  int m, int n,
                                  const double           *alpha,
                                  double           *A, int lda,
                                  double           *x, int incx,
                                  const double           *beta,
                                  double           *y, int incy, int batchCount) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;

  if(m < 0 || n < 0 || incx == 0 || incy == 0)
    return HCBLAS_STATUS_INVALID_VALUE;

  long aOffset = 0;
  long xOffset = 0;
  long yOffset = 0;
  hcblasStatus status;
  hcblasTranspose transA;
  transA =  (trans == HCBLAS_OP_N)? NoTrans : Trans;
  int row, col;
  if(transA == NoTrans){
        row = n;
        col = m;
  }
  else{
        row = m;
        col = n;
  }
  long X_batchOffset = row;
  long Y_batchOffset = col;
  long A_batchOffset = row * col;
  status =  handle->hcblas_dgemv(handle->currentAcclView, handle->Order, transA, m, n, *alpha, A, aOffset, A_batchOffset, lda, x, xOffset, X_batchOffset, incx, *beta, y, yOffset, Y_batchOffset, incy, batchCount);
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

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
                           float           *A, int lda) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;

  if(m < 0 || n < 0 || incx == 0 || incy == 0)
    return HCBLAS_STATUS_INVALID_VALUE;

  long xOffset = 0;
  long yOffset = 0;
  long aOffset = 0;
  hcblasStatus status;
  status = handle->hcblas_sger(handle->currentAcclView, handle->Order, m, n, *alpha, x, xOffset, incx, y, yOffset, incy, A, aOffset, lda );
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

hcblasStatus_t  hcblasSgerBatched(hcblasHandle_t handle, int m, int n,
                                  const float           *alpha,
                                  const float           *x, int incx,
                                  const float           *y, int incy,
                                  float           *A, int lda, int batchCount) {
  if(handle == nullptr  || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;

  if(m < 0 || n < 0 || incx == 0 || incy == 0)
    return HCBLAS_STATUS_INVALID_VALUE;

  long xOffset = 0;
  long yOffset = 0;
  long aOffset = 0;
  long X_batchOffset = m;
  long Y_batchOffset = n;
  long A_batchOffset = m * n;
  hcblasStatus status;
  status = handle->hcblas_sger(handle->currentAcclView, handle->Order, m, n, *alpha, x, xOffset, X_batchOffset, incx, y, yOffset, Y_batchOffset, incy, A, aOffset, A_batchOffset, lda, batchCount);
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

//Dger routines
hcblasStatus_t  hcblasDger(hcblasHandle_t handle, int m, int n,
                           const double           *alpha,
                           const double           *x, int incx,
                           const double           *y, int incy,
                           double           *A, int lda) {
  if(handle == nullptr  || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;

  if(m < 0 || n < 0 || incx == 0 || incy == 0)
    return HCBLAS_STATUS_INVALID_VALUE;

  long xOffset = 0;
  long yOffset = 0;
  long aOffset = 0;
  hcblasStatus status;
  status = handle->hcblas_dger(handle->currentAcclView, handle->Order, m, n, *alpha, x, xOffset, incx, y, yOffset, incy, A, aOffset, lda );
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

hcblasStatus_t  hcblasDgerBatched(hcblasHandle_t handle, int m, int n,
                                  const double           *alpha,
                                  const double           *x, int incx,
                                  const double           *y, int incy,
                                  double           *A, int lda, int batchCount) {
  if(handle == nullptr  || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;

  if(m < 0 || n < 0 || incx == 0 || incy == 0)
    return HCBLAS_STATUS_INVALID_VALUE;

  long xOffset = 0;
  long yOffset = 0;
  long aOffset = 0;
  long X_batchOffset = m;
  long Y_batchOffset = n;
  long A_batchOffset = m * n;
  hcblasStatus status;
  status = handle->hcblas_dger(handle->currentAcclView, handle->Order, m, n, *alpha, x, xOffset, X_batchOffset, incx, y, yOffset, Y_batchOffset, incy, A, aOffset, A_batchOffset, lda, batchCount);
  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

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
// batchCount   host             input          number of pointers contained in input and output arrays.

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
                           float           *C, int ldc) {
  if(handle == nullptr  || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;

  if(m < 0 || n < 0 || k < 0)
    return HCBLAS_STATUS_INVALID_VALUE;

  long aOffset = 0;
  long bOffset = 0;
  long cOffset = 0;
  hcblasStatus status;
  hcblasTranspose transA, transB;
  transA = (transa == HCBLAS_OP_N) ? NoTrans : Trans;
  transB = (transb == HCBLAS_OP_N) ? NoTrans : Trans;
  status = handle->hcblas_sgemm(handle->currentAcclView, handle->Order, transA, transB, m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc, aOffset, bOffset, cOffset);
  if(status == HCBLAS_SUCCEEDS) 
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

hcblasStatus_t hcblasCgemm(hcblasHandle_t handle,
                           hcblasOperation_t transa, hcblasOperation_t transb,
                           int m, int n, int k,
                           const hcComplex       *alpha,
                           hcComplex       *A, int lda,
                           hcComplex       *B, int ldb,
                           const hcComplex       *beta,
                           hcComplex       *C, int ldc) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;

  if(m < 0 || n < 0 || k < 0)
    return HCBLAS_STATUS_INVALID_VALUE;


  long aOffset = 0;
  long bOffset = 0;
  long cOffset = 0;

  hcblasStatus status;

  hcblasTranspose transA, transB;
  transA = (transa == HCBLAS_OP_N) ? NoTrans : Trans;
  transB = (transb == HCBLAS_OP_N) ? NoTrans : Trans;

  status = handle->hcblas_cgemm(handle->currentAcclView, handle->Order, transA, transB, m, n, k, *(reinterpret_cast<const float2*>(alpha)), reinterpret_cast<float2*>(A), aOffset, lda, reinterpret_cast<float2*>(B), bOffset, ldb, *(reinterpret_cast<const float2*>(beta)), reinterpret_cast<float2*>(C), cOffset, ldc);

  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

hcblasStatus_t hcblasDgemm(hcblasHandle_t handle,
                           hcblasOperation_t transa, hcblasOperation_t transb,
                           int m, int n, int k,
                           const double           *alpha,
                           double           *A, int lda,
                           double           *B, int ldb,
                           const double           *beta,
                           double           *C, int ldc) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;

  if(m < 0 || n < 0 || k < 0)
    return HCBLAS_STATUS_INVALID_VALUE;

  long aOffset = 0;
  long bOffset = 0;
  long cOffset = 0;
  hcblasStatus status;
  hcblasTranspose transA, transB;
  transA = (transa == HCBLAS_OP_N) ? NoTrans : Trans;
  transB = (transb == HCBLAS_OP_N) ? NoTrans : Trans;
  status = handle->hcblas_dgemm(handle->currentAcclView, handle->Order, transA, transB, m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc, aOffset, bOffset, cOffset);
  if(status == HCBLAS_SUCCEEDS) 
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

hcblasStatus_t hcblasZgemm(hcblasHandle_t handle,
                           hcblasOperation_t transa, hcblasOperation_t transb,
                           int m, int n, int k,
                           const hcDoubleComplex       *alpha,
                           hcDoubleComplex       *A, int lda,
                           hcDoubleComplex       *B, int ldb,
                           const hcDoubleComplex       *beta,
                           hcDoubleComplex       *C, int ldc) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;

  if(m < 0 || n < 0 || k < 0)
    return HCBLAS_STATUS_INVALID_VALUE;


  long aOffset = 0;
  long bOffset = 0;
  long cOffset = 0;

  hcblasStatus status;

  hcblasTranspose transA, transB;
  transA = (transa == HCBLAS_OP_N) ? NoTrans : Trans;
  transB = (transb == HCBLAS_OP_N) ? NoTrans : Trans;

  status = handle->hcblas_zgemm(handle->currentAcclView, handle->Order, transA, transB, m, n, k, *(reinterpret_cast<const double2*>(alpha)), reinterpret_cast<double2*>(A), aOffset, lda, reinterpret_cast<double2*>(B), bOffset, ldb, *(reinterpret_cast<const double2*>(beta)), reinterpret_cast<double2*>(C), cOffset, ldc);

  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}
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
                                  float           *Carray, int ldc, int batchCount) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;

  if(m < 0 || n < 0 || k < 0 || batchCount < 0)
    return HCBLAS_STATUS_INVALID_VALUE;


  long aOffset = 0;
  long bOffset = 0;
  long cOffset = 0;
  long A_batchOffset = 0;
  long B_batchOffset = 0;
  long C_batchOffset = m * n;

  hcblasStatus status;
  hcblasTranspose transA, transB;
  transA = (transa == HCBLAS_OP_N) ? NoTrans : Trans;
  transB = (transb == HCBLAS_OP_N) ? NoTrans : Trans;

  status = handle->hcblas_sgemm(handle->currentAcclView, handle->Order, transA, transB, m, n, k, *alpha, Aarray, lda, A_batchOffset, Barray, ldb, B_batchOffset, *beta, Carray, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchCount);

  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

hcblasStatus_t hcblasCgemmBatched(hcblasHandle_t handle,
                                  hcblasOperation_t transa, hcblasOperation_t transb,
                                  int m, int n, int k,
                                  const hcComplex       *alpha,
                                  hcComplex       *Aarray, int lda,
                                  hcComplex       *Barray, int ldb,
                                  const hcComplex       *beta,
                                  hcComplex       *Carray, int ldc, int batchCount) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;

  if(m < 0 || n < 0 || k < 0 || batchCount < 0)
    return HCBLAS_STATUS_INVALID_VALUE;

  long aOffset = 0;
  long bOffset = 0;
  long cOffset = 0;
  long A_batchOffset = 0;
  long B_batchOffset = 0;
  long C_batchOffset = m * n;

  hcblasStatus status;
  hcblasTranspose transA, transB;
  transA = (transa == HCBLAS_OP_N) ? NoTrans : Trans;
  transB = (transb == HCBLAS_OP_N) ? NoTrans : Trans;

  status = handle->hcblas_cgemm(handle->currentAcclView, handle->Order, transA, transB, m, n, k, *(reinterpret_cast<const float2*>(alpha)), reinterpret_cast<float2*>(Aarray), aOffset, A_batchOffset, lda, reinterpret_cast<float2*>(Barray), bOffset, B_batchOffset, ldb, *(reinterpret_cast<const float2*>(beta)), reinterpret_cast<float2*>(Carray), cOffset, C_batchOffset, ldc, batchCount);

  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

// Batche Double Implementations
hcblasStatus_t hcblasDgemmBatched(hcblasHandle_t handle,
                                  hcblasOperation_t transa, hcblasOperation_t transb,
                                  int m, int n, int k,
                                  const double           *alpha,
                                  double           *Aarray, int lda,
                                  double           *Barray, int ldb,
                                  const double           *beta,
                                  double           *Carray, int ldc, int batchCount) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;

  if(m < 0 || n < 0 || k < 0 || batchCount < 0)
    return HCBLAS_STATUS_INVALID_VALUE;


  long aOffset = 0;
  long bOffset = 0;
  long cOffset = 0;
  long A_batchOffset = 0;
  long B_batchOffset = 0;
  long C_batchOffset = m * n;

  hcblasStatus status;
  hcblasTranspose transA, transB;
  transA = (transa == HCBLAS_OP_N) ? NoTrans : Trans;
  transB = (transb == HCBLAS_OP_N) ? NoTrans : Trans;

  status = handle->hcblas_dgemm(handle->currentAcclView, handle->Order, transA, transB, m, n, k, *alpha, Aarray, lda, A_batchOffset, Barray, ldb, B_batchOffset, *beta, Carray, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchCount);

  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}

hcblasStatus_t hcblasZgemmBatched(hcblasHandle_t handle,
                                  hcblasOperation_t transa, hcblasOperation_t transb,
                                  int m, int n, int k,
                                  const hcDoubleComplex       *alpha,
                                  hcDoubleComplex       *Aarray, int lda,
                                  hcDoubleComplex       *Barray, int ldb,
                                  const hcDoubleComplex       *beta,
                                  hcDoubleComplex       *Carray, int ldc, int batchCount) {
  if(handle == nullptr || handle->initialized == false)
    return HCBLAS_STATUS_NOT_INITIALIZED;

  if(m < 0 || n < 0 || k < 0 || batchCount < 0)
    return HCBLAS_STATUS_INVALID_VALUE;

  long aOffset = 0;
  long bOffset = 0;
  long cOffset = 0;
  long A_batchOffset = 0;
  long B_batchOffset = 0;
  long C_batchOffset = m * n;

  hcblasStatus status;
  hcblasTranspose transA, transB;
  transA = (transa == HCBLAS_OP_N) ? NoTrans : Trans;
  transB = (transb == HCBLAS_OP_N) ? NoTrans : Trans;

  status = handle->hcblas_zgemm(handle->currentAcclView, handle->Order, transA, transB, m, n, k, *(reinterpret_cast<const double2*>(alpha)), reinterpret_cast<double2*>(Aarray), aOffset, A_batchOffset, lda, reinterpret_cast<double2*>(Barray), bOffset, B_batchOffset, ldb, *(reinterpret_cast<const double2*>(beta)), reinterpret_cast<double2*>(Carray), cOffset, C_batchOffset, ldc, batchCount);

  if(status == HCBLAS_SUCCEEDS)
        return HCBLAS_STATUS_SUCCESS;
  else
        return HCBLAS_STATUS_EXECUTION_FAILED;
}
