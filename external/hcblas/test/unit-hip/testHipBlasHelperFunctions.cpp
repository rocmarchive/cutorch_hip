#include "gtest/gtest.h"
#include "hipblas.h"

TEST(hipblasCreateTest, return_Check_hipblasCreate) {
 // Case I: Input to the API is null handle
 hipblasHandle_t handle = NULL;
 // Passing a Null handle to the API
 hipblasStatus_t status = hipblasCreate(&handle); 
 // Assert if the handle is still NULL after allocation
 EXPECT_TRUE(handle != NULL);
 // If allocation succeeds we must expect a success status
 if (handle != NULL)
   EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS); 
 else
   EXPECT_EQ(status, HIPBLAS_STATUS_ALLOC_FAILED);
}

TEST(hipblasDestroyTest, return_Check_hipblasDestroy) {
 hipblasHandle_t handle = NULL;
 // Passing a Null handle to the API
 hipblasStatus_t status = hipblasCreate(&handle);
 //hipblasDestroy
 status = hipblasDestroy(handle);
 EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
 // Destory again
 status = hipblasDestroy(handle);
 EXPECT_EQ(status, HIPBLAS_STATUS_NOT_INITIALIZED);
}


/*TEST(hipblasSetGetAcclViewTest, func_and_return_check_hipblasSetGetAcclView) {
 // Case I: Input to the API is null handle
 hipblasHandle_t handle = NULL;
 hc::accelerator default_acc;
 hc::accelerator_view default_acc_view = default_acc.get_default_view();
 hc::accelerator_view* accl_view = NULL;
 hipblasStatus_t status = hipblasSetAcclView(handle, default_acc_view);
 EXPECT_EQ(status, HIPBLAS_STATUS_NOT_INITIALIZED);
 status = hipblasGetAcclView(handle, &accl_view);
 EXPECT_EQ(status, HIPBLAS_STATUS_NOT_INITIALIZED);
 // Now create the handle
 status = hipblasCreate(&handle);
 // Assert if the handle is still NULL after allocation
 EXPECT_TRUE(handle != NULL);
 // If allocation succeeds we must expect a success status
 if (handle != NULL)
   EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS); 
 else
   EXPECT_EQ(status, HIPBLAS_STATUS_ALLOC_FAILED);

 status = hipblasSetAcclView(handle, default_acc_view);
 EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
 // Now Get the Accl_view
 status = hipblasGetAcclView(handle, &accl_view);
 EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
 EXPECT_TRUE(accl_view != NULL);
 if (default_acc_view == *accl_view) {
   EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
 }
 // We must expect the accl_view obtained is what that's being set
 //EXPECT_EQ(default_acc_view, *accl_view);
}*/

TEST(hipblasSetVectorTest, return_Check_hipblasSetVector) {
 int n = 10;
 int incx = 1, incy = 1;
 float *x1 = (float*) calloc(n, sizeof(float));
 double *x2 = (double*) calloc(n, sizeof(double));
 hipblasStatus_t status;
 hipblasHandle_t handle = NULL;
 status= hipblasCreate(&handle);
 float *y1 = NULL;
 double *y2 = NULL;
 hipError_t err = hipMalloc(&y1, n);
 err = hipMalloc(&y2, n);
 // HIPBLAS_STATUS_SUCCESS
 // float type memory transfer from host to device
/* status = hipblasSetVector(n, sizeof(x1), x1 , incx, y1, incy);
 EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
 // double type memory transfer from host to device
 status = hipblasSetVector(n, sizeof(x2), x2 , incx, y2, incy);
 EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
 */
 // HIPBLAS_STATUS_INVALID_VALUE 
 // incx is 0
 status = hipblasSetVector(n, sizeof(x1), x1 , 0, y1, incy);
 EXPECT_EQ(status, HIPBLAS_STATUS_INVALID_VALUE);
 // incy is 0
 status = hipblasSetVector(n, sizeof(x1), x1 , incx, y1, 0);
 EXPECT_EQ(status, HIPBLAS_STATUS_INVALID_VALUE);
 // elemSize is 0
 status = hipblasSetVector(n, 0, x1 , incx, y1, incy);
 EXPECT_EQ(status, HIPBLAS_STATUS_INVALID_VALUE);
 
// HIPBLAS_STATUS_MAPPING_ERROR
 /*handle->deviceId = 0;
 status = hipblasSetVector(n, sizeof(x1), x1 , incx, y2, incy);
 EXPECT_EQ(status, HIPBLAS_STATUS_MAPPING_ERROR);
*/
 // HIPBLAS_STATUS_NOT_INITIALIZED  
 hipblasDestroy(handle);
 status = hipblasSetVector(n, sizeof(x1), x1 , incx, y1, incy);
 EXPECT_EQ(status, HIPBLAS_STATUS_NOT_INITIALIZED);

 free(x1);
 free(x2);
 hipFree(y1);
 hipFree(y2);
}

TEST(hipblasGetVectorTest, return_Check_hipblasGetVector) {
 int n = 10;
 int incx = 1, incy = 1;
 float *y1 = (float*) calloc(n, sizeof(float));
 double *y2 = (double*) calloc(n, sizeof(double));
 hipblasStatus_t status;
 hipblasHandle_t handle = NULL;
 status= hipblasCreate(&handle);
 float *x1 = NULL;
 double *x2 = NULL;
 hipError_t err = hipMalloc(&x1, n);
 err = hipMalloc(&x2, n);
 // HIPBLAS_STATUS_SUCCESS
 // float type memory transfer from host to device
/*
 status = hipblasSetVector(n, sizeof(y1), x1 , incx, y1, incy);
 EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
 // double type memory transfer from host to device
 status = hipblasSetVector(n, sizeof(y2), x2 , incx, y2, incy);
 EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);
*/
 // HIPBLAS_STATUS_INVALID_VALUE
 // incx is 0
 status = hipblasSetVector(n, sizeof(y1), x1 , 0, y1, incy);
 EXPECT_EQ(status, HIPBLAS_STATUS_INVALID_VALUE);
 // incy is 0
 status = hipblasSetVector(n, sizeof(y1), x1 , incx, y1, 0);
 EXPECT_EQ(status, HIPBLAS_STATUS_INVALID_VALUE);
 // elemSize is 0
 status = hipblasSetVector(n, 0, x1 , incx, y1, incy);
 EXPECT_EQ(status, HIPBLAS_STATUS_INVALID_VALUE);

 // HIPBLAS_STATUS_MAPPING_ERROR
/* handle->deviceId = 0;
 status = hipblasSetVector(n, sizeof(y1), x2 , incx, y1, incy);
 EXPECT_EQ(status, HIPBLAS_STATUS_MAPPING_ERROR);
*/
 // HIPBLAS_STATUS_NOT_INITIALIZED
 hipblasDestroy(handle);
 status = hipblasSetVector(n, sizeof(y1), x1 , incx, y1, incy);
 EXPECT_EQ(status, HIPBLAS_STATUS_NOT_INITIALIZED);

 free(y1);
 free(y2);
 hipFree(x1);
 hipFree(x2);
}

TEST(hipblasSetMatrixTest, return_Check_hipblasSetMatrix) {
 int rows = 10;
 int cols = 10;
 int lda = 1, ldb = 1;
 float *x1 = (float*) calloc(rows * cols, sizeof(float));
 double *x2 = (double*) calloc(rows * cols, sizeof(double));
 hipblasStatus_t status;
 hipblasHandle_t handle = NULL;
 status= hipblasCreate(&handle);
 float*y1 = NULL;
 double *y2 = NULL;
 hipError_t err = hipMalloc(&y1, rows * cols);
 err = hipMalloc(&y2, rows * cols);

 // HIPBLAS_STATUS_INVALID_VALUE 
 // lda is 0
 status = hipblasSetMatrix(rows, cols, sizeof(x1), x1 , 0, y1, ldb);
 EXPECT_EQ(status, HIPBLAS_STATUS_INVALID_VALUE);
 // ldb is 0
 status = hipblasSetMatrix(rows, cols, sizeof(x1), x1 , lda, y1, 0);
 EXPECT_EQ(status, HIPBLAS_STATUS_INVALID_VALUE);
 // elemSize is 0
 status = hipblasSetMatrix(rows, cols, 0, x1 , lda, y1, ldb);
 EXPECT_EQ(status, HIPBLAS_STATUS_INVALID_VALUE);
 
// HIPBLAS_STATUS_MAPPING_ERROR
/* handle->deviceId = 0;
 status = hipblasSetMatrix(rows, cols, sizeof(x1), x1 , lda, y2, ldb);
 EXPECT_EQ(status, HIPBLAS_STATUS_MAPPING_ERROR);
*/
 // HIPBLAS_STATUS_NOT_INITIALIZED  
 hipblasDestroy(handle);
 status = hipblasSetMatrix(rows, cols, sizeof(x1), x1 , lda, y1, ldb);
 EXPECT_EQ(status, HIPBLAS_STATUS_NOT_INITIALIZED);

 free(x1);
 free(x2);
 hipFree(y1);
 hipFree(y2);
}

TEST(hipblasGetMatrixTest, return_Check_hipblasGetMatrix) {
 int rows = 10;
 int cols = 10;
 int lda = 1, ldb = 1;
 float *y1 = (float*) calloc(cols * rows, sizeof(float));
 double *y2 = (double*) calloc(cols * rows, sizeof(double));
 hipblasStatus_t status;
 hipblasHandle_t handle = NULL;
 status= hipblasCreate(&handle);
 float*x1 = NULL, *x2 = NULL;
 hipError_t err = hipMalloc(&x1, rows * cols);
 err = hipMalloc(&x2, rows * cols);

 // HIPBLAS_STATUS_INVALID_VALUE
 // lda is 0
 status = hipblasSetMatrix(rows, cols, sizeof(y1), x1 , 0, y1, ldb);
 EXPECT_EQ(status, HIPBLAS_STATUS_INVALID_VALUE);
 // ldb is 0
 status = hipblasSetMatrix(rows, cols, sizeof(y1), x1 , lda, y1, 0);
 EXPECT_EQ(status, HIPBLAS_STATUS_INVALID_VALUE);
 // elemSize is 0
 status = hipblasSetMatrix(rows, cols, 0, x1 , lda, y1, ldb);
 EXPECT_EQ(status, HIPBLAS_STATUS_INVALID_VALUE);

/* // HIPBLAS_STATUS_MAPPING_ERROR
 status = hipblasSetMatrix(rows, cols, sizeof(y1), x2 , lda, y1, ldb);
 EXPECT_EQ(status, HIPBLAS_STATUS_MAPPING_ERROR);
*/
 // HIPBLAS_STATUS_NOT_INITIALIZED
 hipblasDestroy(handle);
 status = hipblasSetMatrix(rows, cols, sizeof(y1), x1 , lda, y1, ldb);
 EXPECT_EQ(status, HIPBLAS_STATUS_NOT_INITIALIZED);

 free(y1);
 free(y2);
 hipFree(x1);
 hipFree(x2);
}


