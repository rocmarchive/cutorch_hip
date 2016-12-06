#include "hipblas.h"
#include <hip/hcc_detail/hcc_acc.h>

#ifdef __cplusplus
extern "C" {
#endif

//hipblasSetStream()
//This function sets the hipBLAS library stream, which will be used to execute all subsequent calls to the hipBLAS library functions. If the hipBLAS library stream is not set, all kernels use the defaultNULL stream. In particular, this routine can be used to change the stream between kernel launches and then to reset the hipBLAS library stream back to NULL.
//Return Value 	Meaning

// Returns
// HIPBLAS_STATUS_SUCCESS         :the stream was set successfully
// HIPBLAS_STATUS_NOT_INITIALIZED :the library was not initialized
hipblasStatus_t hipblasSetStream(hipblasHandle_t handle, hipStream_t streamId) {
  if (handle == nullptr) {
    return HIPBLAS_STATUS_NOT_INITIALIZED;    
  }
  hc::accelerator_view *pAcclView;
  hipError_t err = hipHccGetAcceleratorView(streamId, &pAcclView);
  if (err != hipSuccess)
  { 
    return HIPBLAS_STATUS_NOT_INITIALIZED;
  }
  return hipHCBLASStatusToHIPStatus(hcblasSetAcclView(handle, *pAcclView, static_cast<void*>(&streamId)));
} 

//hipblasGetStream()
// This function gets the hipBLAS library stream, which is being used to execute all calls to the hipBLAS library functions. If the hipBLAS library stream is not set, all kernels use the defaultNULL stream.
// Return Value 	
// HIPBLAS_STATUS_SUCCESS : the stream was returned successfully
// HIPBLAS_STATUS_NOT_INITIALIZED : the library was not initialized

hipblasStatus_t  hipblasGetStream(hipblasHandle_t handle, hipStream_t *streamId) {
  if (handle == nullptr) {
    return HIPBLAS_STATUS_NOT_INITIALIZED;    
  }
  hc::accelerator_view **ppAcclView;
  return hipHCBLASStatusToHIPStatus(hcblasGetAcclView(handle, ppAcclView, (void**)(&streamId)));
}

hcblasOperation_t hipOperationToHCCOperation( hipblasOperation_t op)
{
	switch (op)
	{
		case HIPBLAS_OP_N:
			return HCBLAS_OP_N;
		
		case HIPBLAS_OP_T:
			return HCBLAS_OP_T;
			
		case HIPBLAS_OP_C:
			return HCBLAS_OP_C;
			
		default:
			throw "Non existent OP";
	}
}

hipblasOperation_t HCCOperationToHIPOperation( hcblasOperation_t op)
{
	switch (op)
	{
		case HCBLAS_OP_N :
			return HIPBLAS_OP_N;
		
		case HCBLAS_OP_T :
			return HIPBLAS_OP_T;
			
		case HCBLAS_OP_C :
			return HIPBLAS_OP_C;
			
		default:
			throw "Non existent OP";
	}
}


hipblasStatus_t hipHCBLASStatusToHIPStatus(hcblasStatus_t hcStatus) 
{
	switch(hcStatus)
	{
		case HCBLAS_STATUS_SUCCESS:
			return HIPBLAS_STATUS_SUCCESS;
		case HCBLAS_STATUS_NOT_INITIALIZED:
			return HIPBLAS_STATUS_NOT_INITIALIZED;
		case HCBLAS_STATUS_ALLOC_FAILED:
			return HIPBLAS_STATUS_ALLOC_FAILED;
		case HCBLAS_STATUS_INVALID_VALUE:
			return HIPBLAS_STATUS_INVALID_VALUE;
		case HCBLAS_STATUS_MAPPING_ERROR:
			return HIPBLAS_STATUS_MAPPING_ERROR;
		case HCBLAS_STATUS_EXECUTION_FAILED:
			return HIPBLAS_STATUS_EXECUTION_FAILED;
		case HCBLAS_STATUS_INTERNAL_ERROR:
			return HIPBLAS_STATUS_INTERNAL_ERROR;
		default:
			throw "Unimplemented status";
	}
}

hipblasStatus_t hipblasDestroy(hipblasHandle_t handle) {
    return hipHCBLASStatusToHIPStatus(hcblasDestroy(&handle)); 
}

hipblasStatus_t hipblasSetVector(int n, int elemSize, const void *x, int incx, void *y, int incy){
	return hipHCBLASStatusToHIPStatus(hcblasSetVector(dummyGlobal, n, elemSize, x, incx, y, incy)); //HGSOS no need for handle moving forward
}

hipblasStatus_t hipblasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy){
	return hipHCBLASStatusToHIPStatus(hcblasGetVector(dummyGlobal, n, elemSize, x, incx, y, incy)); //HGSOS no need for handle
}

hipblasStatus_t hipblasSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb){
	return hipHCBLASStatusToHIPStatus(hcblasSetMatrix(dummyGlobal, rows, cols, elemSize, A, lda, B, ldb));
}

hipblasStatus_t hipblasGetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb){
	return hipHCBLASStatusToHIPStatus(hcblasGetMatrix(dummyGlobal, rows, cols, elemSize, A, lda, B, ldb));
}

hipblasStatus_t  hipblasSasum(hipblasHandle_t handle, int n, const float *x, int incx, float  *result){
	return hipHCBLASStatusToHIPStatus(hcblasSasum(handle, n, const_cast<float*>(x), incx, result));
}

hipblasStatus_t  hipblasDasum(hipblasHandle_t handle, int n, const double *x, int incx, double *result){
	return hipHCBLASStatusToHIPStatus(hcblasDasum(handle, n, const_cast<double*>(x), incx, result));
}

hipblasStatus_t  hipblasSasumBatched(hipblasHandle_t handle, int n, float *x, int incx, float  *result, int batchCount){
	return hipHCBLASStatusToHIPStatus(hcblasSasumBatched( handle, n, x, incx, result, batchCount));
}

hipblasStatus_t  hipblasDasumBatched(hipblasHandle_t handle, int n, double *x, int incx, double *result, int batchCount){
	return hipHCBLASStatusToHIPStatus(hcblasDasumBatched(handle, n, x, incx, result, batchCount));
}

hipblasStatus_t hipblasSaxpy(hipblasHandle_t handle, int n, const float *alpha,   const float *x, int incx, float *y, int incy) {
	return hipHCBLASStatusToHIPStatus(hcblasSaxpy(handle, n, alpha, x, incx, y, incy));
}

hipblasStatus_t hipblasDaxpy(hipblasHandle_t handle, int n, const double *alpha,   const double *x, int incx, double *y, int incy) {
	return hipHCBLASStatusToHIPStatus(hcblasDaxpy(handle, n, alpha, x, incx, y, incy));
}

hipblasStatus_t hipblasSaxpyBatched(hipblasHandle_t handle, int n, const float *alpha, const float *x, int incx,  float *y, int incy, int batchCount){
	return hipHCBLASStatusToHIPStatus(hcblasSaxpyBatched(handle, n, alpha, x, incx, y, incy, batchCount));
}

hipblasStatus_t hipblasScopy(hipblasHandle_t handle, int n, const float *x, int incx, float *y, int incy){
	return hipHCBLASStatusToHIPStatus(hcblasScopy( handle, n, x, incx, y, incy));
}

hipblasStatus_t hipblasDcopy(hipblasHandle_t handle, int n, const double *x, int incx, double *y, int incy){
	return hipHCBLASStatusToHIPStatus(hcblasDcopy( handle, n, x, incx, y, incy));
}

hipblasStatus_t hipblasScopyBatched(hipblasHandle_t handle, int n, const float *x, int incx, float *y, int incy, int batchCount){
	return hipHCBLASStatusToHIPStatus(hcblasScopyBatched( handle, n, x, incx, y, incy, batchCount));
}

hipblasStatus_t hipblasDcopyBatched(hipblasHandle_t handle, int n, const double *x, int incx, double *y, int incy, int batchCount){
	return hipHCBLASStatusToHIPStatus(hcblasDcopyBatched( handle, n, x, incx, y, incy, batchCount));
}

hipblasStatus_t hipblasSdot (hipblasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result){
	return 	hipHCBLASStatusToHIPStatus(hcblasSdot(handle, n, x, incx, y, incy, result));			   
}

hipblasStatus_t hipblasDdot (hipblasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result){
	return 	hipHCBLASStatusToHIPStatus(hcblasDdot(handle, n, x, incx, y, incy, result));			   
}

hipblasStatus_t hipblasSdotBatched (hipblasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result, int batchCount){
	return 	hipHCBLASStatusToHIPStatus(hcblasSdotBatched(handle, n, x, incx, y, incy, result, batchCount));			   
}

hipblasStatus_t hipblasDdotBatched (hipblasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result, int batchCount){
	return 	hipHCBLASStatusToHIPStatus(hcblasDdotBatched ( handle, n, x, incx, y, incy, result, batchCount));			   
}

hipblasStatus_t  hipblasSscal(hipblasHandle_t handle, int n, const float *alpha,  float *x, int incx){
	return hipHCBLASStatusToHIPStatus(hcblasSscal(handle, n, alpha,  x, incx));
}

hipblasStatus_t  hipblasDscal(hipblasHandle_t handle, int n, const double *alpha,  double *x, int incx){
	return hipHCBLASStatusToHIPStatus(hcblasDscal(handle, n, alpha,  x, incx));
}

hipblasStatus_t  hipblasSscalBatched(hipblasHandle_t handle, int n, const float *alpha,  float *x, int incx, int batchCount){
	return hipHCBLASStatusToHIPStatus(hcblasSscalBatched(handle, n, alpha,  x, incx, batchCount));
}

hipblasStatus_t  hipblasDscalBatched(hipblasHandle_t handle, int n, const double *alpha,  double *x, int incx, int batchCount){
	return hipHCBLASStatusToHIPStatus(hcblasDscalBatched(handle, n, alpha,  x, incx, batchCount));
}

hipblasStatus_t hipblasSgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda,
                           const float *x, int incx,  const float *beta,  float *y, int incy){
        // TODO: Remove const_cast
	return hipHCBLASStatusToHIPStatus(hcblasSgemv(handle, hipOperationToHCCOperation(trans),  m,  n, alpha, const_cast<float*>(A), lda, const_cast<float*>(x), incx, beta,  y, incy));						   
}

hipblasStatus_t hipblasDgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda,
                           const double *x, int incx,  const double *beta,  double *y, int incy){
        // TODO: Remove const_cast
	return hipHCBLASStatusToHIPStatus(hcblasDgemv(handle, hipOperationToHCCOperation(trans),  m,  n, alpha, const_cast<double*>(A), lda, const_cast<double*>(x), incx, beta,  y, incy));						   
}

hipblasStatus_t hipblasSgemvBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const float *alpha, float *A, int lda,
                           float *x, int incx,  const float *beta,  float *y, int incy, int batchCount){
	return hipHCBLASStatusToHIPStatus(hcblasSgemvBatched(handle, hipOperationToHCCOperation(trans),  m,  n, alpha, A, lda, x, incx, beta,  y, incy, batchCount));						   
}

hipblasStatus_t  hipblasSger(hipblasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda){
	return hipHCBLASStatusToHIPStatus(hcblasSger(handle, m, n, alpha, x, incx, y, incy, A, lda));
}

hipblasStatus_t  hipblasSgerBatched(hipblasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda, int batchCount){
	return hipHCBLASStatusToHIPStatus(hcblasSgerBatched(handle, m, n, alpha, x, incx, y, incy, A, lda, batchCount));
}

hipblasStatus_t hipblasSgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc){
   // TODO: Remove const_cast
	return hipHCBLASStatusToHIPStatus(hcblasSgemm( handle, hipOperationToHCCOperation(transa),  hipOperationToHCCOperation(transb), m,  n,  k, alpha, const_cast<float*>(A),  lda, const_cast<float*>(B),  ldb, beta, C,  ldc));
}

hipblasStatus_t hipblasDgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc){
	return hipHCBLASStatusToHIPStatus(hcblasDgemm( handle, hipOperationToHCCOperation(transa),  hipOperationToHCCOperation(transb), m,  n,  k, alpha, const_cast<double*>(A),  lda, const_cast<double*>(B),  ldb, beta, C,  ldc));
}

hipblasStatus_t hipblasCgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const hipComplex *alpha, hipComplex *A, int lda, hipComplex *B, int ldb, const hipComplex *beta, hipComplex *C, int ldc){
	return hipHCBLASStatusToHIPStatus(hcblasCgemm( handle, hipOperationToHCCOperation(transa),  hipOperationToHCCOperation(transb), m,  n,  k, alpha, A,  lda, B,  ldb, beta, C,  ldc));
}

hipblasStatus_t hipblasSgemmBatched(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const float *alpha, float *A, int lda, float *B, int ldb, const float *beta, float *C, int ldc, int batchCount){
	return hipHCBLASStatusToHIPStatus(hcblasSgemmBatched( handle, hipOperationToHCCOperation(transa),  hipOperationToHCCOperation(transb), m,  n,  k, alpha, A,  lda, B,  ldb, beta, C,  ldc, batchCount));
}

hipblasStatus_t hipblasDgemmBatched(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const double *alpha, double *A, int lda, double *B, int ldb, const double *beta, double *C, int ldc, int batchCount){
	return hipHCBLASStatusToHIPStatus(hcblasDgemmBatched( handle, hipOperationToHCCOperation(transa),  hipOperationToHCCOperation(transb), m,  n,  k, alpha, A,  lda, B,  ldb, beta, C,  ldc, batchCount));
}

hipblasStatus_t hipblasCgemmBatched(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const hipComplex *alpha, hipComplex *A, int lda, hipComplex *B, int ldb, const hipComplex *beta, hipComplex *C, int ldc, int batchCount){
	return hipHCBLASStatusToHIPStatus(hcblasCgemmBatched( handle, hipOperationToHCCOperation(transa),  hipOperationToHCCOperation(transb), m,  n,  k, alpha, A,  lda, B,  ldb, beta, C,  ldc, batchCount));
}


// TODO - review use of this handle:
hipblasHandle_t dummyGlobal;

hipblasStatus_t hipblasCreate(hipblasHandle_t* handle) {
  int deviceId;
  hipError_t err;
  hipblasStatus_t retval = HIPBLAS_STATUS_SUCCESS;

  err = hipGetDevice(&deviceId);
  if (err == hipSuccess) {
    hc::accelerator acc;
    err = hipHccGetAccelerator(deviceId, &acc);
    if (err == hipSuccess) {
      retval = hipHCBLASStatusToHIPStatus(hcblasCreate(&*handle, &acc));
      dummyGlobal = *handle;
    } else {
      retval = HIPBLAS_STATUS_INVALID_VALUE;
    }
  }
  return retval;
}

#ifdef __cplusplus
}
#endif
