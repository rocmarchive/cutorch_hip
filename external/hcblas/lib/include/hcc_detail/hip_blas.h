/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma once

#include <hip/hip_runtime_api.h>
#include <hcblas.h>

//HGSOS for Kalmar leave it as C++, only cublas needs C linkage.

#ifdef __cplusplus
extern "C" {
#endif

typedef hcblasHandle_t hipblasHandle_t;
typedef hcComplex hipComplex ;

extern  hipblasHandle_t dummyGlobal;
/* Unsupported types
		"cublasFillMode_t",
		"cublasDiagType_t",
		"cublasSideMode_t",
		"cublasPointerMode_t",
		"cublasAtomicsMode_t",
		"cublasDataType_t"
*/	

hcblasOperation_t hipOperationToHCCOperation( hipblasOperation_t op);

hipblasOperation_t HCCOperationToHIPOperation( hcblasOperation_t op);

hipblasStatus_t hipHCBLASStatusToHIPStatus(hcblasStatus_t hcStatus); 

hipblasStatus_t hipblasCreate(hipblasHandle_t* handle);

hipblasStatus_t hipblasDestroy(hipblasHandle_t handle);

hipblasStatus_t hipblasSetStream(hipblasHandle_t handle, hipStream_t streamId);

hipblasStatus_t  hipblasGetStream(hipblasHandle_t handle, hipStream_t *streamId);

hipblasStatus_t hipblasSetVector(int n, int elemSize, const void *x, int incx, void *y, int incy);

hipblasStatus_t hipblasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy);

hipblasStatus_t hipblasSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);

hipblasStatus_t hipblasGetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);

hipblasStatus_t  hipblasSasum(hipblasHandle_t handle, int n, const float *x, int incx, float  *result);

hipblasStatus_t  hipblasDasum(hipblasHandle_t handle, int n, const double *x, int incx, double *result);

hipblasStatus_t  hipblasSasumBatched(hipblasHandle_t handle, int n, float *x, int incx, float  *result, int batchCount);

hipblasStatus_t  hipblasDasumBatched(hipblasHandle_t handle, int n, double *x, int incx, double *result, int batchCount);

hipblasStatus_t hipblasSaxpy(hipblasHandle_t handle, int n, const float *alpha,   const float *x, int incx, float *y, int incy);

hipblasStatus_t hipblasDaxpy(hipblasHandle_t handle, int n, const double *alpha,   const double *x, int incx, double *y, int incy) ;

hipblasStatus_t hipblasSaxpyBatched(hipblasHandle_t handle, int n, const float *alpha, const float *x, int incx,  float *y, int incy, int batchCount);

hipblasStatus_t hipblasScopy(hipblasHandle_t handle, int n, const float *x, int incx, float *y, int incy);

hipblasStatus_t hipblasDcopy(hipblasHandle_t handle, int n, const double *x, int incx, double *y, int incy);

hipblasStatus_t hipblasScopyBatched(hipblasHandle_t handle, int n, const float *x, int incx, float *y, int incy, int batchCount);

hipblasStatus_t hipblasDcopyBatched(hipblasHandle_t handle, int n, const double *x, int incx, double *y, int incy, int batchCount);

hipblasStatus_t hipblasSdot (hipblasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result);

hipblasStatus_t hipblasDdot (hipblasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result);

hipblasStatus_t hipblasSdotBatched (hipblasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result, int batchCount);

hipblasStatus_t hipblasDdotBatched (hipblasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result, int batchCount);

hipblasStatus_t  hipblasSscal(hipblasHandle_t handle, int n, const float *alpha,  float *x, int incx);

hipblasStatus_t  hipblasDscal(hipblasHandle_t handle, int n, const double *alpha,  double *x, int incx);

hipblasStatus_t  hipblasSscalBatched(hipblasHandle_t handle, int n, const float *alpha,  float *x, int incx, int batchCount);

hipblasStatus_t  hipblasDscalBatched(hipblasHandle_t handle, int n, const double *alpha,  double *x, int incx, int batchCount);

hipblasStatus_t hipblasSgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda,
                           const float *x, int incx,  const float *beta,  float *y, int incy);

hipblasStatus_t hipblasDgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda,
                           const double *x, int incx,  const double *beta,  double *y, int incy);

hipblasStatus_t hipblasSgemvBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const float *alpha, float *A, int lda,
                           float *x, int incx,  const float *beta,  float *y, int incy, int batchCount);

hipblasStatus_t  hipblasSger(hipblasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda);

hipblasStatus_t  hipblasSgerBatched(hipblasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda, int batchCount);

hipblasStatus_t hipblasSgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);

hipblasStatus_t hipblasDgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc);

hipblasStatus_t hipblasCgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const hipComplex *alpha, hipComplex *A, int lda, hipComplex *B, int ldb, const hipComplex *beta, hipComplex *C, int ldc);

hipblasStatus_t hipblasSgemmBatched(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const float *alpha, float *A, int lda, float *B, int ldb, const float *beta, float *C, int ldc, int batchCount);

hipblasStatus_t hipblasDgemmBatched(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const double *alpha, double *A, int lda, double *B, int ldb, const double *beta, double *C, int ldc, int batchCount);

hipblasStatus_t hipblasCgemmBatched(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const hipComplex *alpha, hipComplex *A, int lda, hipComplex *B, int ldb, const hipComplex *beta, hipComplex *C, int ldc, int batchCount);




#ifdef __cplusplus
}
#endif


