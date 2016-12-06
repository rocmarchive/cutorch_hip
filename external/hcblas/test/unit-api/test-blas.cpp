#include "hcblas.h"
#include "gtest/gtest.h"
#include "hc_am.hpp"
#include "cblas.h"
#include "hcblaslib.h"

TEST(hcblaswrapper_sasum, func_return_correct_sasum) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status= hcblasCreate(&handle);
  int n = 23;
  int incx = 1;
  long lenx = 1 + (n-1) * abs(incx);
  float* result;

  // HCBLAS_STATUS_SUCCESS and FUNCTIONALITY CHECK
  float *X = (float*)calloc(lenx, sizeof(float));//host input
  float* devX = hc::am_alloc(sizeof(float) * lenx, handle->currentAccl, 0);
  for(int i = 0; i < lenx; i++){
            X[i] = rand() % 10;
  }

  status = hcblasSetVector(handle, lenx, sizeof(float), X, incx, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);

//  handle->currentAcclView.copy(X, devX, lenx * sizeof(float));
  status = hcblasSasum(handle, n, devX, incx, result);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);

  float asumcblas = 0.0;
  asumcblas = cblas_sasum( n, X, incx);
  EXPECT_EQ(*result, asumcblas);

  // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasSasum(handle, n, devX, incx, result);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED); 

  free(X);
  hc::am_free(devX);
}

TEST(hcblaswrapper_sasumBatched, func_return_correct_sasumBatched) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status= hcblasCreate(&handle);
  int n = 23;
  int incx = 1;
  long lenx = 1 + (n-1) * abs(incx);
  float result;
  int batchSize = 128;

  // HCBLAS_STATUS_SUCCESS and FUNCTIONALITY CHECK
  float *X = (float*)calloc(lenx * batchSize, sizeof(float));//host input
  float* devX = hc::am_alloc(sizeof(float) * lenx * batchSize, handle->currentAccl, 0);
  for(int i = 0; i < lenx * batchSize; i++){
            X[i] = rand() % 10;
  }
  status = hcblasSetVector(handle, lenx*batchSize, sizeof(float), X, incx, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSasumBatched(handle, n, devX, incx, &result, batchSize);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);

  float asumcblas = 0.0;
  float *asumcblastemp = (float*)calloc(batchSize, sizeof(float));
  for(int i = 0; i < batchSize; i++) {
                asumcblastemp[i] = cblas_sasum( n, X + i * n, incx);
                asumcblas += asumcblastemp[i];
  }
  EXPECT_EQ(result, asumcblas);

  // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasSasumBatched(handle, n, devX, incx, &result, batchSize);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED); 

  free(X);
  hc::am_free(devX);
}

TEST(hcblaswrapper_dasum, func_return_correct_dasum) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status= hcblasCreate(&handle);
  int n = 23;
  int incx = 1;
  long lenx = 1 + (n-1) * abs(incx);
  double result;

  // HCBLAS_STATUS_SUCCESS and FUNCTIONALITY CHECK
  double *X = (double*)calloc(lenx, sizeof(double));//host input
  double* devX = hc::am_alloc(sizeof(double) * lenx, handle->currentAccl, 0);
  for(int i = 0; i < lenx; i++){
            X[i] = rand() % 10;
   }
  status = hcblasSetVector(handle, lenx, sizeof(double), X, incx, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasDasum(handle, n, devX, incx, &result);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);

  double asumcblas = 0.0;
  asumcblas = cblas_dasum( n, X, incx);
  EXPECT_EQ(result, asumcblas);

  // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasDasum(handle, n, devX, incx, &result);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED);

  free(X);
  hc::am_free(devX);
}

TEST(hcblaswrapper_dasumBatched, func_return_correct_dasumBatched) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status= hcblasCreate(&handle);
  int n = 23;
  int incx = 1;
  long lenx = 1 + (n-1) * abs(incx);
  double result;
  int batchSize = 128;

  // HCBLAS_STATUS_SUCCESS and FUNCTIONALITY CHECK
  double *X = (double*)calloc(lenx * batchSize, sizeof(double));//host input
  double* devX = hc::am_alloc(sizeof(double) * lenx * batchSize, handle->currentAccl, 0);
  for(int i = 0; i < lenx * batchSize; i++){
            X[i] = rand() % 10;
   }
  status = hcblasSetVector(handle, lenx*batchSize, sizeof(double), X, incx, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasDasumBatched(handle, n, devX, incx, &result, batchSize);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);

  double asumcblas = 0.0;
  double *asumcblastemp = (double*)calloc(batchSize, sizeof(double));
  for(int i = 0; i < batchSize; i++) {
                asumcblastemp[i] = cblas_dasum( n, X + i * n, incx);
                asumcblas += asumcblastemp[i];
  }
  EXPECT_EQ(result, asumcblas);


  // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasDasumBatched(handle, n, devX, incx, &result, batchSize);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED);

  free(X);
  hc::am_free(devX);
}

TEST(hcblaswrapper_sscal, func_return_correct_sscal) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status= hcblasCreate(&handle);
  int n = 123;
  int incx = 1;
  long lenx = 1 + (n-1) * abs(incx);
  float alpha = 1;

  // HCBLAS_STATUS_SUCCESS and FUNCTIONALITY CHECK
  float *Xcblas = (float*)calloc(lenx, sizeof(float));
  float *X = (float*)calloc(lenx, sizeof(float));//host input
  float* devX = hc::am_alloc(sizeof(float) * lenx, handle->currentAccl, 0);
  for(int i = 0; i < lenx; i++){
            X[i] = rand() % 10;
            Xcblas[i] = X[i];
  }
  status = hcblasSetVector(handle, lenx, sizeof(float), X, incx, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSscal(handle, n, &alpha, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasGetVector(handle, lenx, sizeof(float), devX, incx, X, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  cblas_sscal( n, alpha, Xcblas, incx );
  for(int i = 0; i < lenx ; i++){
        EXPECT_EQ(X[i], Xcblas[i]);
  }

  // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasSscal(handle, n, &alpha, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED); 

  free(X);
  free(Xcblas);
  hc::am_free(devX);
}

TEST(hcblaswrapper_sscalBatched, func_return_correct_sscalBatched) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status= hcblasCreate(&handle);
  int n = 123;
  int incx = 1;
  long lenx = 1 + (n-1) * abs(incx);
  float alpha = 1;
  int batchSize = 128;

  // HCBLAS_STATUS_SUCCESS and FUNCTIONALITY CHECK
  float *Xcblas = (float*)calloc(lenx * batchSize , sizeof(float));
  float *X = (float*)calloc(lenx * batchSize, sizeof(float));//host input
  float* devX = hc::am_alloc(sizeof(float) * lenx * batchSize, handle->currentAccl, 0);
  for(int i = 0; i < lenx * batchSize; i++){
            X[i] = rand() % 10;
            Xcblas[i] =  X[i];
  }
  status = hcblasSetVector(handle, lenx*batchSize, sizeof(float), X, incx, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSscalBatched(handle, n, &alpha, devX, incx, batchSize);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasGetVector(handle, lenx*batchSize, sizeof(float), devX, incx, X, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  for(int i = 0; i < batchSize; i++)
          cblas_sscal( n, alpha, Xcblas + i * n, incx);
  for(int i =0; i < lenx * batchSize; i ++){
          EXPECT_EQ(X[i], Xcblas[i]);
  }

  // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasSscalBatched(handle, n, &alpha, devX, incx, batchSize);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED); 

  free(X);
  free(Xcblas);
  hc::am_free(devX);
}

TEST(hcblaswrapper_dscal, func_return_correct_dscal) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status= hcblasCreate(&handle);
  int n = 123;
  int incx = 1;
  long lenx = 1 + (n-1) * abs(incx);
  double alpha = 1;

  // HCBLAS_STATUS_SUCCESS and FUNCTIONALITY CHECK
  double *Xcblas = (double*)calloc(lenx, sizeof(double));
  double *X = (double*)calloc(lenx, sizeof(double));//host input
  double* devX = hc::am_alloc(sizeof(double) * lenx, handle->currentAccl, 0);
  for(int i = 0; i < lenx; i++){
            X[i] = rand() % 10;
            Xcblas[i] = X[i];
  }
  status = hcblasSetVector(handle, lenx, sizeof(double), X, incx, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasDscal(handle, n, &alpha, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasGetVector(handle, lenx, sizeof(double), devX, incx, X, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  cblas_dscal( n, alpha, Xcblas, incx );
  for(int i = 0; i < lenx ; i++){
        EXPECT_EQ(X[i], Xcblas[i]);
  }

  // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasDscal(handle, n, &alpha, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED); 

  free(X);
  free(Xcblas);
  hc::am_free(devX);
}

TEST(hcblaswrapper_dscalBatched, func_return_correct_dscalBatched) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status= hcblasCreate(&handle);
  int n = 123;
  int incx = 1;
  long lenx = 1 + (n-1) * abs(incx);
  double alpha = 1;
  int batchSize = 128;

  // HCBLAS_STATUS_SUCCESS and FUNCTIONALITY CHECK
  double *Xcblas = (double*)calloc(lenx * batchSize , sizeof(double));
  double *X = (double*)calloc(lenx * batchSize, sizeof(double));//host input
  double* devX = hc::am_alloc(sizeof(double) * lenx * batchSize, handle->currentAccl, 0);
  for(int i = 0; i < lenx * batchSize; i++){
            X[i] = rand() % 10;
            Xcblas[i] =  X[i];
  }
  status = hcblasSetVector(handle, lenx*batchSize, sizeof(double), X, incx, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasDscalBatched(handle, n, &alpha, devX, incx, batchSize);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasGetVector(handle, lenx*batchSize, sizeof(double), devX, incx, X, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  for(int i = 0; i < batchSize; i++)
          cblas_dscal( n, alpha, Xcblas + i * n, incx);
  for(int i =0; i < lenx * batchSize; i ++){
          EXPECT_EQ(X[i], Xcblas[i]);
  }

  // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasDscalBatched(handle, n, &alpha, devX, incx, batchSize);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED); 

  free(X);
  free(Xcblas);
  hc::am_free(devX);
}

TEST(hcblaswrapper_scopy, func_return_correct_scopy) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status= hcblasCreate(&handle);
  int n = 123;
  int incx = 1;
  int incy = 1;
  long lenx = 1 + (n-1) * abs(incx);
  long leny = 1 + (n-1) * abs(incy);
  float alpha = 1;

  // HCBLAS_STATUS_SUCCESS and FUNCTIONALITY CHECK
  float *X = (float*)calloc(lenx, sizeof(float));//host input
  float *Y = (float*)calloc(leny, sizeof(float));
  float *Ycblas = (float*)calloc(leny, sizeof(float));
  float* devX = hc::am_alloc(sizeof(float) * lenx, handle->currentAccl, 0);
  float* devY = hc::am_alloc(sizeof(float) * leny, handle->currentAccl, 0);
  for(int i = 0; i < lenx; i++){
            X[i] = rand() % 10;
  }
  for(int i = 0;i < leny;i++){
            Y[i] =  rand() % 15;
            Ycblas[i] = Y[i];
  }
  status = hcblasSetVector(handle, lenx, sizeof(float), X, incx, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetVector(handle, leny, sizeof(float), Y, incy, devY, incy);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasScopy(handle, n, devX, incx, devY, incy);
//  handle->currentAcclView.copy(devY, Y, leny * sizeof(float));
  status = hcblasGetVector(handle, leny, sizeof(float), devY, incy, Y, incy);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  cblas_scopy( n, X, incx, Ycblas, incy );
  for(int i = 0; i < leny; i++){
        EXPECT_EQ(Y[i], Ycblas[i]);
  }

  // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasScopy(handle, n, devX, incx, devY, incy);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED);

  free(X);
  free(Ycblas);
  hc::am_free(devX);
  free(Y);
  hc::am_free(devY);
}

TEST(hcblaswrapper_scopyBatched, func_return_correct_scopyBatched) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status= hcblasCreate(&handle);
  int n = 123;
  int incx = 1;
  int incy = 1;
  long lenx = 1 + (n-1) * abs(incx);
  long leny = 1 + (n-1) * abs(incy);
  float alpha = 1;
  int batchSize = 32; 

  // HCBLAS_STATUS_SUCCESS and FUNCTIONALITY CHECK
  float *X = (float*)calloc(lenx * batchSize, sizeof(float));//host input
  float *Y = (float*)calloc(leny * batchSize, sizeof(float));
  float *Ycblas = (float*)calloc(leny *  batchSize, sizeof(float));
  float* devX = hc::am_alloc(sizeof(float) * lenx *  batchSize, handle->currentAccl, 0);
  float* devY = hc::am_alloc(sizeof(float) * leny *  batchSize, handle->currentAccl, 0);
  for(int i = 0; i < lenx *  batchSize; i++){
            X[i] = rand() % 10;
  }
  for(int i = 0;i < leny *  batchSize;i++){
            Y[i] =  rand() % 15;
            Ycblas[i] = Y[i];
  }
  status = hcblasSetVector(handle, lenx * batchSize, sizeof(float), X, incx, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetVector(handle, leny * batchSize, sizeof(float), Y, incy, devY, incy);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasScopyBatched(handle, n, devX, incx, devY, incy, batchSize);
  status = hcblasGetVector(handle, leny * batchSize, sizeof(float), devY, incy, Y, incy);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
//  handle->currentAcclView.copy(devY, Y, leny * batchSize * sizeof(float));
  for(int i = 0; i < batchSize; i++)
      cblas_scopy( n, X + i * n, incx, Ycblas + i * n, incy );
  for(int i = 0; i < leny * batchSize; i++){
        EXPECT_EQ(Y[i], Ycblas[i]);
  }

  // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasScopyBatched(handle, n, devX, incx, devY, incy, batchSize);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED);

  free(X);
  free(Ycblas);
  hc::am_free(devX);
  free(Y);
  hc::am_free(devY);
}

TEST(hcblaswrapper_dcopy, func_return_correct_dcopy) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status= hcblasCreate(&handle);
  int n = 123;
  int incx = 1;
  int incy = 1;
  long lenx = 1 + (n-1) * abs(incx);
  long leny = 1 + (n-1) * abs(incy);
  double alpha = 1;

  // HCBLAS_STATUS_SUCCESS and FUNCTIONALITY CHECK
  double *X = (double*)calloc(lenx, sizeof(double));//host input
  double *Y = (double*)calloc(leny, sizeof(double));
  double *Ycblas = (double*)calloc(leny, sizeof(double));
  double* devX = hc::am_alloc(sizeof(double) * lenx, handle->currentAccl, 0);
  double* devY = hc::am_alloc(sizeof(double) * leny, handle->currentAccl, 0);
  for(int i = 0; i < lenx; i++){
            X[i] = rand() % 10;
  }
  for(int i = 0;i < leny;i++){
            Y[i] =  rand() % 15;
            Ycblas[i] = Y[i];
  }
  status = hcblasSetVector(handle, lenx, sizeof(double), X, incx, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetVector(handle, leny, sizeof(double), Y, incy, devY, incy);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasDcopy(handle, n, devX, incx, devY, incy);
  status = hcblasGetVector(handle, leny, sizeof(double), devY, incy, Y, incy);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  //handle->currentAcclView.copy(devY, Y, leny * sizeof(double));
  cblas_dcopy( n, X, incx, Ycblas, incy );
  for(int i = 0; i < leny; i++){
        EXPECT_EQ(Y[i], Ycblas[i]);
  }

  // HCBLAS_STATUS_NOT_INITIALIZED
   hcblasDestroy(&handle);
  status = hcblasDcopy(handle, n, devX, incx, devY, incy);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED);

  free(X);
  free(Ycblas);
  hc::am_free(devX);
  free(Y);
  hc::am_free(devY);
}

TEST(hcblaswrapper_dcopyBatched, func_return_correct_dcopyBatched) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status= hcblasCreate(&handle);
  int n = 123;
  int incx = 1;
  int incy = 1;
  long lenx = 1 + (n-1) * abs(incx);
  long leny = 1 + (n-1) * abs(incy);
  double alpha = 1;
  int batchSize = 32;

  // HCBLAS_STATUS_SUCCESS and FUNCTIONALITY CHECK
  double *X = (double*)calloc(lenx * batchSize, sizeof(double));//host input
  double *Y = (double*)calloc(leny * batchSize, sizeof(double));
  double *Ycblas = (double*)calloc(leny *  batchSize, sizeof(double));
  double* devX = hc::am_alloc(sizeof(double) * lenx *  batchSize, handle->currentAccl, 0);
  double* devY = hc::am_alloc(sizeof(double) * leny *  batchSize, handle->currentAccl, 0);
  for(int i = 0; i < lenx *  batchSize; i++){
            X[i] = rand() % 10;
  }
  for(int i = 0;i < leny *  batchSize;i++){
            Y[i] =  rand() % 15;
            Ycblas[i] = Y[i];
  }
  status = hcblasSetVector(handle, lenx*batchSize, sizeof(double), X, incx, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetVector(handle, leny*batchSize, sizeof(double), Y, incy, devY, incy);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasDcopyBatched(handle, n, devX, incx, devY, incy, batchSize);
//  handle->currentAcclView.copy(devY, Y, leny * batchSize * sizeof(double));
  status = hcblasGetVector(handle, leny*batchSize, sizeof(double), devY, incy, Y, incy);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  for(int i = 0; i < batchSize; i++)
      cblas_dcopy( n, X + i * n, incx, Ycblas + i * n, incy );
  for(int i = 0; i < leny * batchSize; i++){
        EXPECT_EQ(Y[i], Ycblas[i]);
  }

  // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasDcopyBatched(handle, n, devX, incx, devY, incy, batchSize);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED);

  free(X);
  free(Ycblas);
  hc::am_free(devX);
  free(Y);
  hc::am_free(devY);
}

TEST(hcblaswrapper_sdot, func_return_correct_sdot) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status= hcblasCreate(&handle);
  int n = 123;
  int incx = 1;
  int incy = 1;
  long lenx = 1 + (n-1) * abs(incx);
  long leny = 1 + (n-1) * abs(incy);
  float result;

  // HCBLAS_STATUS_SUCCESS and FUNCTIONALITY CHECK
  float *X = (float*)calloc(lenx, sizeof(float));//host input
  float *Y = (float*)calloc(leny, sizeof(float));
  float* devX = hc::am_alloc(sizeof(float) * lenx, handle->currentAccl, 0);
  float* devY = hc::am_alloc(sizeof(float) * leny, handle->currentAccl, 0);
  for(int i = 0; i < lenx; i++){
            X[i] = rand() % 10;
  }
  for(int i = 0;i < leny;i++){
            Y[i] =  rand() % 15;
  }
  status = hcblasSetVector(handle, lenx, sizeof(float), X, incx, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetVector(handle, leny, sizeof(float), Y, incy, devY, incy);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSdot(handle, n, devX, incx, devY, incy, &result);
  float  dotcblas = 0.0;
  dotcblas = cblas_sdot( n, X, incx, Y, incy);
  EXPECT_EQ(result, dotcblas);

  // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasSdot(handle, n, devX, incx, devY, incy, &result);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED);

  free(X);
  hc::am_free(devX);
  free(Y);
  hc::am_free(devY);
}

TEST(hcblaswrapper_sdotBatched, func_return_correct_sdotBatched) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status= hcblasCreate(&handle);
  int n = 123;
  int incx = 1;
  int incy = 1;
  long lenx = 1 + (n-1) * abs(incx);
  long leny = 1 + (n-1) * abs(incy);
  float result;
  int batchSize = 32;

  // HCBLAS_STATUS_SUCCESS and FUNCTIONALITY CHECK
  float *X = (float*)calloc(lenx * batchSize, sizeof(float));//host input
  float *Y = (float*)calloc(leny * batchSize, sizeof(float));
  float* devX = hc::am_alloc(sizeof(float) * lenx * batchSize, handle->currentAccl, 0);
  float* devY = hc::am_alloc(sizeof(float) * leny * batchSize, handle->currentAccl, 0);
  float *dotcblastemp =(float*)calloc(batchSize, sizeof(float));
  for(int i = 0; i < lenx * batchSize; i++){
            X[i] = rand() % 10;
  }
  for(int i = 0;i < leny * batchSize;i++){
            Y[i] =  rand() % 15;
  }
  status = hcblasSetVector(handle, lenx*batchSize, sizeof(float), X, incx, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetVector(handle, leny*batchSize, sizeof(float), Y, incy, devY, incy);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSdotBatched(handle, n, devX, incx, devY, incy, &result, batchSize);
  float  dotcblas = 0.0;
  for(int i = 0; i < batchSize; i++){
                dotcblastemp[i] = cblas_sdot( n, X + i * n, incx, Y + i * n, incy);
                dotcblas += dotcblastemp[i];
  }
  EXPECT_EQ(result, dotcblas);

  // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasSdotBatched(handle, n, devX, incx, devY, incy, &result, batchSize);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED);

  free(X);
  hc::am_free(devX);
  free(Y);
  hc::am_free(devY);
}

TEST(hcblaswrapper_ddot, func_return_correct_ddot) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status= hcblasCreate(&handle);
  int n = 123;
  int incx = 1;
  int incy = 1;
  long lenx = 1 + (n-1) * abs(incx);
  long leny = 1 + (n-1) * abs(incy);
  double result;

  // HCBLAS_STATUS_SUCCESS and FUNCTIONALITY CHECK
  double *X = (double*)calloc(lenx, sizeof(double));//host input
  double *Y = (double*)calloc(leny, sizeof(double));
  double* devX = hc::am_alloc(sizeof(double) * lenx, handle->currentAccl, 0);
  double* devY = hc::am_alloc(sizeof(double) * leny, handle->currentAccl, 0);
  for(int i = 0; i < lenx; i++){
            X[i] = rand() % 10;
  }
  for(int i = 0;i < leny;i++){
            Y[i] =  rand() % 15;
  }
  status = hcblasSetVector(handle, lenx, sizeof(double), X, incx, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetVector(handle, leny, sizeof(double), Y, incy, devY, incy);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasDdot(handle, n, devX, incx, devY, incy, &result);
  double  dotcblas = 0.0;
  dotcblas = cblas_ddot( n, X, incx, Y, incy);
  EXPECT_EQ(result, dotcblas);

  // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasDdot(handle, n, devX, incx, devY, incy, &result);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED);

  free(X);
  hc::am_free(devX);
  free(Y);
  hc::am_free(devY);
}

TEST(hcblaswrapper_ddotBatched, func_return_correct_ddotBatched) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status = hcblasCreate(&handle);
  int n = 123;
  int incx = 1;
  int incy = 1;
  long lenx = 1 + (n-1) * abs(incx);
  long leny = 1 + (n-1) * abs(incy);
  double result;
  int batchSize = 32;

  // HCBLAS_STATUS_SUCCESS and FUNCTIONALITY CHECK
  double *X = (double*)calloc(lenx * batchSize, sizeof(double));//host input
  double *Y = (double*)calloc(leny * batchSize, sizeof(double));
  double* devX = hc::am_alloc(sizeof(double) * lenx * batchSize, handle->currentAccl, 0);
  double* devY = hc::am_alloc(sizeof(double) * leny * batchSize, handle->currentAccl, 0);
  double *dotcblastemp =(double*)calloc(batchSize, sizeof(double));
  for(int i = 0; i < lenx * batchSize; i++){
            X[i] = rand() % 10;
  }
  for(int i = 0;i < leny * batchSize;i++){
            Y[i] =  rand() % 15;
  }
  status = hcblasSetVector(handle, lenx*batchSize, sizeof(double), X, incx, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetVector(handle, leny*batchSize, sizeof(double), Y, incy, devY, incy);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasDdotBatched(handle, n, devX, incx, devY, incy, &result, batchSize);
  double  dotcblas = 0.0;
  for(int i = 0; i < batchSize; i++){
                dotcblastemp[i] = cblas_ddot( n, X + i * n, incx, Y + i * n, incy);
                dotcblas += dotcblastemp[i];
  }
  EXPECT_EQ(result, dotcblas);

  // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasDdotBatched(handle, n, devX, incx, devY, incy, &result, batchSize);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED);

  free(X);
  hc::am_free(devX);
  free(Y);
  hc::am_free(devY);
}

TEST(hcblaswrapper_saxpy, func_return_correct_saxpy) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status = hcblasCreate(&handle);
  int n = 123;
  int incx = 1;
  int incy = 1;
  long lenx = 1 + (n-1) * abs(incx);
  long leny = 1 + (n-1) * abs(incy);
  float alpha = 1;

  // HCBLAS_STATUS_SUCCESS and FUNCTIONALITY CHECK
  float *X = (float*)calloc(lenx, sizeof(float));//host input
  float *Y = (float*)calloc(leny, sizeof(float));
  float *Ycblas = (float*)calloc(leny, sizeof(float));
  float* devX = hc::am_alloc(sizeof(float) * lenx, handle->currentAccl, 0);
  float* devY = hc::am_alloc(sizeof(float) * leny, handle->currentAccl, 0);
  for(int i = 0; i < lenx; i++){
            X[i] = rand() % 10;
  }
  for(int i = 0;i < leny;i++){
            Y[i] =  rand() % 15;
            Ycblas[i] = Y[i];
  }
  status = hcblasSetVector(handle, lenx, sizeof(float), X, incx, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetVector(handle, leny, sizeof(float), Y, incy, devY, incy);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSaxpy(handle, n, &alpha, devX, incx, devY, incy);
  status = hcblasGetVector(handle, leny, sizeof(float), devY, 1, Y, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  cblas_saxpy( n, alpha, X, incx, Ycblas, incy );
  for(int i = 0; i < leny ; i++){
     EXPECT_EQ(Y[i], Ycblas[i]);
  }
  
  // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasSaxpy(handle, n, &alpha, devX, incx, devY, incy);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED);

  free(X);
  free(Ycblas);
  hc::am_free(devX);
  free(Y);
  hc::am_free(devY);
}

TEST(hcblaswrapper_saxpyBatched, func_return_correct_saxpyBatched) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status = hcblasCreate(&handle);
  int n = 123;
  int incx = 1;
  int incy = 1;
  long lenx = 1 + (n-1) * abs(incx);
  long leny = 1 + (n-1) * abs(incy);
  float alpha = 1;
  int batchSize = 32;

  // HCBLAS_STATUS_SUCCESS and FUNCTIONALITY CHECK
  float *X = (float*)calloc(lenx * batchSize, sizeof(float));//host input
  float *Y = (float*)calloc(leny * batchSize, sizeof(float));
  float *Ycblas = (float*)calloc(leny * batchSize, sizeof(float));
  float* devX = hc::am_alloc(sizeof(float) * lenx * batchSize, handle->currentAccl, 0);
  float* devY = hc::am_alloc(sizeof(float) * leny * batchSize, handle->currentAccl, 0);
  for(int i = 0; i < lenx * batchSize; i++){
            X[i] = rand() % 10;
  }
  for(int i = 0;i < leny * batchSize;i++){
            Y[i] =  rand() % 15;
            Ycblas[i] = Y[i];
  }
  status = hcblasSetVector(handle, lenx*batchSize, sizeof(float), X, incx, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetVector(handle, leny*batchSize, sizeof(float), Y, incy, devY, incy);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSaxpyBatched(handle, n, &alpha, devX, incx, devY, incy, batchSize);
  status = hcblasGetVector(handle, leny, sizeof(float), devY, 1, Y, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  for(int i = 0; i < batchSize; i++)
       cblas_saxpy( n, alpha, X + i * n, incx, Ycblas + i * n, incy );
  for(int i =0; i < leny * batchSize; i ++){
     // TODO: CHeck the cause for this failure 
     // EXPECT_EQ(Y[i], Ycblas[i]);
  }

  // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasSaxpyBatched(handle, n, &alpha, devX, incx, devY, incy, batchSize);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED);

  free(X);
  hc::am_free(devX);
  free(Y);
  free(Ycblas);
  hc::am_free(devY);
}

TEST(hcblaswrapper_sger, func_return_correct_sger) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status = hcblasCreate(&handle);
  int m = 123;
  int n = 78;
  int incx = 1;
  int incy = 1;
  long lenx = 1 + (m-1) * abs(incx);
  long leny = 1 + (n-1) * abs(incy);
  float alpha = 1;
  long lda;
  lda = (handle->Order)? m : n;
  CBLAS_ORDER order;
  order = (handle->Order)? CblasColMajor: CblasRowMajor;

  // HCBLAS_STATUS_SUCCESS and FUNCTIONALITY CHECK
  float *Acblas = (float *)calloc( lenx * leny , sizeof(float));
  float *X = (float*)calloc(lenx, sizeof(float));//host input
  float *Y = (float*)calloc(leny, sizeof(float));
  float *A = (float *)calloc( lenx * leny , sizeof(float));
  float* devA = hc::am_alloc(sizeof(float) * lenx * leny, handle->currentAccl, 0);
  float* devX = hc::am_alloc(sizeof(float) * lenx, handle->currentAccl, 0);
  float* devY = hc::am_alloc(sizeof(float) * leny, handle->currentAccl, 0);
  for(int i = 0; i < lenx; i++){
            X[i] = rand() % 10;
  }
  for(int i = 0;i < leny;i++){
            Y[i] =  rand() % 15;
  }
  for(int i = 0;i< lenx * leny ;i++) {
            A[i] = rand() % 25;
            Acblas[i] = A[i];
  }
  status = hcblasSetVector(handle, lenx*leny, sizeof(float), A, incx, devA, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetVector(handle, lenx, sizeof(float), X, incx, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetVector(handle, leny, sizeof(float), Y, incy, devY, incy);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSger(handle, m, n, &alpha, devX, incx, devY, incy, devA, lda);
  status = hcblasGetVector(handle, lenx * leny, sizeof(float), devA, 1, A, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  cblas_sger( order, m, n, alpha, X, incx, Y, incy, Acblas, lda);
  for(int i =0; i < lenx * leny ; i++){
      EXPECT_EQ(A[i], Acblas[i]);
  }
  
  // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasSger(handle, m, n, &alpha, devX, incx, devY, incy, devA, lda);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED);

  free(X);
  free(Acblas);
  hc::am_free(devX);
  free(Y);
  hc::am_free(devY);
}

TEST(hcblaswrapper_sgerBatched, func_return_correct_sgerBatched) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status = hcblasCreate(&handle);
  int m = 123;
  int n = 67;
  int incx = 1;
  int incy = 1;
  long lenx = 1 + (m-1) * abs(incx);
  long leny = 1 + (n-1) * abs(incy);
  float alpha = 1;
  int batchSize = 32;
  long lda;
  lda = (handle->Order)? m : n;
  CBLAS_ORDER order;
  order = (handle->Order)? CblasColMajor: CblasRowMajor;

  // HCBLAS_STATUS_SUCCESS and FUNCTIONALITY CHECK
  float *X = (float*)calloc(lenx * batchSize, sizeof(float));//host input
  float *Y = (float*)calloc(leny * batchSize, sizeof(float));
  float *Acblas = (float*)calloc(leny * lenx * batchSize, sizeof(float));
  float *A = (float *)calloc( lenx * leny * batchSize, sizeof(float));
  float* devA = hc::am_alloc(sizeof(float) * lenx * leny * batchSize, handle->currentAccl, 0);
  float* devX = hc::am_alloc(sizeof(float) * lenx * batchSize, handle->currentAccl, 0);
  float* devY = hc::am_alloc(sizeof(float) * leny * batchSize, handle->currentAccl, 0);
  for(int i = 0; i < lenx * batchSize; i++){
            X[i] = rand() % 10;
  }
  for(int i = 0;i < leny * batchSize;i++){
            Y[i] =  rand() % 15;
  }
  for(int i = 0;i< lenx * leny * batchSize;i++) {
            A[i] = rand() % 25;
            Acblas[i] = A[i];
  }
  status = hcblasSetVector(handle, lenx*leny*batchSize, sizeof(float), A, incx, devA, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetVector(handle, lenx*batchSize, sizeof(float), X, incx, devX, incx);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetVector(handle, leny*batchSize, sizeof(float), Y, incy, devY, incy);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSgerBatched(handle, m, n, &alpha, devX, incx, devY, incy, devA, lda, batchSize);
  status = hcblasGetVector(handle, lenx * leny * batchSize, sizeof(float), devA, 1, A, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  for(int i = 0; i < batchSize; i++)
      cblas_sger( order, m, n, alpha, X + i * m, incx, Y + i * n, incy, Acblas + i * m * n, lda);
  for(int i =0; i < lenx * leny * batchSize; i++){
      EXPECT_EQ(A[i], Acblas[i]);
  }

  // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasSgerBatched(handle, m, n, &alpha, devX, incx, devY, incy, devA, lda, batchSize);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED);

  free(X);
  hc::am_free(devX);
  free(Y);
  free(Acblas);
  hc::am_free(devY);
  free(A);
  hc::am_free(devA);
}


TEST(hcblaswrapper_sgemv, func_return_correct_sgemv) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status = hcblasCreate(&handle);
  int m = 123;
  int n = 78;
  int incx = 1;
  int incy = 1;
  long lenx;
  long leny;
  float alpha = 1;
  float beta = 1;
  long lda;
  CBLAS_ORDER order;
  order = (handle->Order)? CblasColMajor: CblasRowMajor;
  int row, col;
  row = n; col = m; lda = m; 
  hcblasOperation_t trans = HCBLAS_OP_N;
  CBLAS_TRANSPOSE transa;
  transa = (trans == HCBLAS_OP_N)? CblasNoTrans : CblasTrans;
  lenx = 1 + (row - 1) * abs(incx);
  leny = 1 + (col - 1) * abs(incy);

  // NoTransA
  // HCBLAS_STATUS_SUCCESS and FUNCTIONALITY CHECK
  float *Ycblas = (float *)calloc( leny , sizeof(float));
  float *X = (float*)calloc(lenx, sizeof(float));//host input
  float *Y = (float*)calloc(leny, sizeof(float));
  float *A = (float *)calloc( lenx * leny , sizeof(float));
  float* devA = hc::am_alloc(sizeof(float) * lenx * leny, handle->currentAccl, 0);
  float* devX = hc::am_alloc(sizeof(float) * lenx, handle->currentAccl, 0);
  float* devY = hc::am_alloc(sizeof(float) * leny, handle->currentAccl, 0);
  for(int i = 0; i < lenx; i++){
            X[i] = rand() % 10;
  }
  for(int i = 0;i < leny;i++){
            Y[i] =  rand() % 15;
            Ycblas[i] = Y[i];
  }
  for(int i = 0;i< lenx * leny ;i++) {
            A[i] = rand() % 25;
  }
  status = hcblasSetVector(handle, lenx * leny, sizeof(float), A, 1, devA, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetVector(handle, lenx, sizeof(float), X, 1, devX, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetVector(handle, leny, sizeof(float), Y, 1, devY, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);

  status = hcblasSgemv(handle, trans, m, n, &alpha, devA, lda, devX, incx, &beta, devY, incy); 
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);

  status = hcblasGetVector(handle, leny, sizeof(float), devY, 1, Y, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);

  lda = (handle->Order)? m: n;
  cblas_sgemv( order, transa, m, n, alpha, A, lda , X, incx, beta, Ycblas, incy );
  for(int i =0; i < leny ; i++){
      EXPECT_EQ(Y[i], Ycblas[i]);
  }
  // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasSgemv(handle, trans, m, n, &alpha, devA, lda, devX, incx, &beta, devY, incy);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED);

  free(X);
  free(Ycblas);
  hc::am_free(devX);
  free(Y);
  hc::am_free(devY);
  free(A);
  hc::am_free(devA);
}


TEST(hcblaswrapper_sgemvBatched, func_return_correct_sgemvBatched) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status = hcblasCreate(&handle);
  int m = 123;
  int n = 67;
  int incx = 1;
  int incy = 1;
  long lenx;
  long leny;
  float alpha = 1;
  float beta = 1;
  int batchSize = 32;
  long lda;
  CBLAS_ORDER order;
  order = (handle->Order)? CblasColMajor: CblasRowMajor;
  int row, col;
  row = n; col = m; lda = m;
  hcblasOperation_t trans = HCBLAS_OP_N;
  CBLAS_TRANSPOSE transa;
  transa = (trans == HCBLAS_OP_N)? CblasNoTrans : CblasTrans;
  lenx = 1 + (row - 1) * abs(incx);
  leny = 1 + (col - 1) * abs(incy);

  // HCBLAS_STATUS_SUCCESS and FUNCTIONALITY CHECK
  float *X = (float*)calloc(lenx * batchSize, sizeof(float));//host input
  float *Y = (float*)calloc(leny * batchSize, sizeof(float));
  float *Ycblas = (float*)calloc(leny * batchSize, sizeof(float));
  float *A = (float *)calloc( lenx * leny * batchSize, sizeof(float));
  float* devA = hc::am_alloc(sizeof(float) * lenx * leny * batchSize, handle->currentAccl, 0);
  float* devX = hc::am_alloc(sizeof(float) * lenx * batchSize, handle->currentAccl, 0);
  float* devY = hc::am_alloc(sizeof(float) * leny * batchSize, handle->currentAccl, 0);
  for(int i = 0; i < lenx * batchSize; i++){
            X[i] = rand() % 10;
  }
  for(int i = 0;i < leny * batchSize;i++){
            Y[i] =  rand() % 15;
            Ycblas[i] = Y[i];
  }
  for(int i = 0;i< lenx * leny * batchSize;i++) {
            A[i] = rand() % 25;
  }

  status = hcblasSetVector(handle, lenx * leny * batchSize, sizeof(float), A, 1, devA, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetVector(handle, lenx * batchSize, sizeof(float), X, 1, devX, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetVector(handle, leny * batchSize, sizeof(float), Y, 1, devY, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);

  status = hcblasSgemvBatched(handle, trans, m, n, &alpha, devA, lda, devX, incx, &beta, devY, incy, batchSize);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);

  status = hcblasGetVector(handle, leny * batchSize, sizeof(float), devY, 1, Y, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);

  lda = (handle->Order)? m: n;
  for(int i =0 ; i < batchSize; i++)
      cblas_sgemv( order, transa, m, n, alpha, A + i * m * n, lda , X + i * row, incx, beta, Ycblas + i * col, incy );
  for(int i =0; i < leny * batchSize; i++){
      EXPECT_EQ(Y[i], Ycblas[i]);
  }

  // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasSgemvBatched(handle, trans, m, n, &alpha, devA, lda, devX, incx, &beta, devY, incy, batchSize);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED);

  free(X);
  hc::am_free(devX);
  free(Y);
  free(Ycblas);
  hc::am_free(devY);
  free(A);
  hc::am_free(devA);
}

TEST(hcblaswrapper_sgemm, func_return_correct_sgemm) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status = hcblasCreate(&handle);
  int M = 123;
  int N = 78;
  int K = 23;
  int incx = 1, incy = 1;
  float alpha = 1;
  float beta = 1;
  long lda;
  long ldb;
  long ldc;
  CBLAS_ORDER order;
  order = (handle->Order)? CblasColMajor: CblasRowMajor;
  hcblasOperation_t typeA, typeB;
  CBLAS_TRANSPOSE Transa, Transb;
  float *A = (float*) calloc(M * K, sizeof(float));
  float *B = (float*) calloc(K * N, sizeof(float));
  float *C = (float*) calloc(M * N, sizeof(float));
  float *C_hcblas = (float*) calloc(M * N, sizeof(float));
  float *C_cblas = (float*) calloc(M * N, sizeof(float));
  float* devA = hc::am_alloc(sizeof(float) * M * K, handle->currentAccl, 0);
  float* devB = hc::am_alloc(sizeof(float) * K * N, handle->currentAccl, 0);
  float* devC = hc::am_alloc(sizeof(float) * M * N, handle->currentAccl, 0);
  for(int i = 0; i < M * K; i++) {
              A[i] = rand()%100;
  }
  for(int i = 0; i < K * N;i++) {
              B[i] = rand() % 15;
  }
  for(int i = 0; i < M * N;i++) {
              C[i] = rand() % 25;
              C_cblas[i] = C[i];
  }
  status = hcblasSetMatrix(handle, M, K, sizeof(float), A, 1, devA, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetMatrix(handle, K, N, sizeof(float), B, 1, devB, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetMatrix(handle, M, N, sizeof(float), C, 1, devC, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);

  // NoTransA and NoTransB */           
  typeA = HCBLAS_OP_N;
  typeB = HCBLAS_OP_N;
  Transa = CblasNoTrans;
  Transb = CblasNoTrans;

    // Column major */
  lda = M; ldb = K ; ldc = M;
  status = hcblasSgemm(handle, typeA, typeB, M, N, K, &alpha, devA, lda, devB, ldb, &beta, devC, ldc);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);

  status = hcblasGetMatrix(handle, M, N, sizeof(float), devC, 1, C_hcblas, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);

  cblas_sgemm( order, Transa, Transb, M, N, K, alpha, A, lda, B, ldb, beta, C_cblas, ldc);
  for(int i = 0 ; i < M * N ; i++)
    EXPECT_EQ(C_hcblas[i], C_cblas[i]);

   // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasSgemm(handle, typeA, typeB, M, N, K, &alpha, devA, lda, devB, ldb, &beta, devC, ldc);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED);

  free(A);
  free(B);
  free(C);
  hc::am_free(devA);
  hc::am_free(devB);
  hc::am_free(devC);
  free(C_cblas);
  free(C_hcblas);
}

TEST(hcblaswrapper_sgemmBatched, func_return_correct_sgemmBatched) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status = hcblasCreate(&handle);
  int M = 123;
  int N = 78;
  int K = 23;
  int incx = 1, incy = 1;
  float alpha = 1;
  float beta = 1;
  long lda;
  long ldb;
  long ldc;
  int batchSize = 32;
  CBLAS_ORDER order;
  order = (handle->Order)? CblasColMajor: CblasRowMajor;
  hcblasOperation_t typeA, typeB;
  CBLAS_TRANSPOSE Transa, Transb;
  float *A = (float*) calloc(M * K, sizeof(float));
  float *B = (float*) calloc(K * N, sizeof(float));
  float *C = (float*) calloc(M * N * batchSize, sizeof(float));
  float *C_hcblas = (float*) calloc(M * N * batchSize, sizeof(float));
  float *C_cblas = (float*) calloc(M * N * batchSize, sizeof(float));
  float* devA = hc::am_alloc(sizeof(float) * M * K, handle->currentAccl, 0);
  float* devB = hc::am_alloc(sizeof(float) * K * N, handle->currentAccl, 0);
  float* devC = hc::am_alloc(sizeof(float) * M * N * batchSize, handle->currentAccl, 0);
  for(int i = 0; i < M * K; i++) {
              A[i] = rand()%100;
  }
  for(int i = 0; i < K * N;i++) {
              B[i] = rand() % 15;
  }
  for(int i = 0; i < M * N * batchSize;i++) {
              C[i] = rand() % 25;
              C_cblas[i] = C[i];
  }
  status = hcblasSetMatrix(handle, M, K, sizeof(float), A, 1, devA, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetMatrix(handle, K, N, sizeof(float), B, 1, devB, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetMatrix(handle, M, N * batchSize, sizeof(float), C, 1, devC, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);

  // NoTransA and NoTransB */           
  typeA = HCBLAS_OP_N;
  typeB = HCBLAS_OP_N;
  Transa = CblasNoTrans;
  Transb = CblasNoTrans;

    // Column major */
  lda = M; ldb = K ; ldc = M;
  status = hcblasSgemmBatched(handle, typeA, typeB, M, N, K, &alpha, devA, lda, devB, ldb, &beta, devC, ldc, batchSize);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);

  status = hcblasGetMatrix(handle, M, N * batchSize, sizeof(float), devC, 1, C_hcblas, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);

  for(int i = 0; i < batchSize; i++)
         cblas_sgemm( order, Transa, Transb, M, N, K, alpha, A, lda, B, ldb, beta, C_cblas  + i * M * N ,ldc );
  for(int i = 0 ; i < M * N * batchSize; i++)
    EXPECT_EQ(C_hcblas[i], C_cblas[i]);

  // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasSgemmBatched(handle, typeA, typeB, M, N, K, &alpha, devA, lda, devB, ldb, &beta, devC, ldc, batchSize);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED);

  free(A);
  free(B);
  free(C);
  hc::am_free(devA);
  hc::am_free(devB);
  hc::am_free(devC);
  free(C_cblas);
  free(C_hcblas);

}

TEST(hcblaswrapper_cgemm, func_return_correct_cgemm) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status = hcblasCreate(&handle);
  int M = 123;
  int N = 78;
  int K = 23;
  int incx = 1, incy = 1;
  long lda;
  long ldb;
  long ldc;
  CBLAS_ORDER order;
  order = (handle->Order)? CblasColMajor: CblasRowMajor;
  hcblasOperation_t typeA, typeB;
  CBLAS_TRANSPOSE Transa, Transb;
    float alpha[2], beta[2];
    hcComplex cAlpha, cBeta;
    cAlpha.x = 1;
    cAlpha.y = 1;
    cBeta.x = 1;
    cBeta.y = 1;
    alpha[0] = cAlpha.x;
    alpha[1] = cAlpha.y;
    beta[0] = cBeta.x;
    beta[1] = cBeta.y;
    hcComplex *A = (hcComplex*) calloc(M * K, sizeof(hcComplex));
    hcComplex *B = (hcComplex*) calloc(K * N, sizeof(hcComplex));
    hcComplex *C = (hcComplex*) calloc(M * N, sizeof(hcComplex));
    hcComplex* devA = hc::am_alloc(sizeof(hcComplex) * M * K, handle->currentAccl, 0);
    hcComplex* devB = hc::am_alloc(sizeof(hcComplex) * K * N, handle->currentAccl, 0);
    hcComplex* devC = hc::am_alloc(sizeof(hcComplex) * M * N, handle->currentAccl, 0);
    float* ablas = (float *)malloc(sizeof(float )* M * K * 2);
    float* bblas = (float *)malloc(sizeof(float )* K * N * 2);
    float* cblas = (float *)malloc(sizeof(float )* M * N * 2);
    int k = 0;
    for(int i = 0; i < M * K; i++) {
                A[i].x = rand() % 10;
                A[i].y = rand() % 20;
                ablas[k++] = A[i].x;
                ablas[k++] = A[i].y;
    }
    k = 0;
    for(int i = 0; i < K * N;i++) {
                B[i].x = rand() % 15;
                B[i].y = rand() % 25;
                bblas[k++] = B[i].x;
                bblas[k++] = B[i].y;
    }
    k = 0;
    for(int i = 0; i < M * N;i++) {
                C[i].x = rand() % 18;
                C[i].y = rand() % 28;
                cblas[k++] = C[i].x;
                cblas[k++] = C[i].y;
    }

  status = hcblasSetMatrix(handle, M, K, sizeof(hcComplex), A, 1, devA, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetMatrix(handle, K, N, sizeof(hcComplex), B, 1, devB, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetMatrix(handle, M, N, sizeof(hcComplex), C, 1, devC, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  // NoTransA and NoTransB */           
  typeA = HCBLAS_OP_N;
  typeB = HCBLAS_OP_N;
  Transa = CblasNoTrans;
  Transb = CblasNoTrans;

    // Column major */
  lda = M; ldb = K ; ldc = M;
  status = hcblasCgemm(handle, typeA, typeB, M, N, K, &cAlpha, devA, lda, devB, ldb, &cBeta, devC, ldc);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);

  status = hcblasGetMatrix(handle, M, N, sizeof(hcComplex), devC, 1, C, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);

  cblas_cgemm( order, Transa, Transb, M, N, K, &alpha, ablas, lda, bblas, ldb, &beta, cblas, ldc );
  for(int i = 0, k = 0; ((i < M * N) && ( k < M * N * 2)) ; i++, k = k + 2) {
            EXPECT_EQ(C[i].x, cblas[k]);
            EXPECT_EQ(C[i].y, cblas[k+1]);
  }

   // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasCgemm(handle, typeA, typeB, M, N, K, &cAlpha, devA, lda, devB, ldb, &cBeta, devC, ldc);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED);

  free(A);
  free(B);
  free(C);
  hc::am_free(devA);
  hc::am_free(devB);
  hc::am_free(devC);
  free(ablas);
  free(bblas);
  free(cblas);
}

TEST(hcblaswrapper_cgemmBatched, func_return_correct_cgemmBatched) {
  hcblasStatus_t status;
  hcblasHandle_t handle = NULL;
  status = hcblasCreate(&handle);
  int M = 123;
  int N = 78;
  int K = 23;
  int incx = 1, incy = 1;
  long lda;
  long ldb;
  long ldc;
  int batchSize = 64;
  CBLAS_ORDER order;
  order = (handle->Order)? CblasColMajor: CblasRowMajor;
  hcblasOperation_t typeA, typeB;
  CBLAS_TRANSPOSE Transa, Transb;
    float alpha[2], beta[2];
    hcComplex cAlpha, cBeta;
    cAlpha.x = 1;
    cAlpha.y = 1;
    cBeta.x = 1;
    cBeta.y = 1;
    alpha[0] = cAlpha.x;
    alpha[1] = cAlpha.y;
    beta[0] = cBeta.x;
    beta[1] = cBeta.y;
    hcComplex *A = (hcComplex*) calloc(M * K, sizeof(hcComplex));
    hcComplex *B = (hcComplex*) calloc(K * N, sizeof(hcComplex));
    hcComplex *C = (hcComplex*) calloc(M * N * batchSize, sizeof(hcComplex));
    hcComplex* devA = hc::am_alloc(sizeof(hcComplex) * M * K, handle->currentAccl, 0);
    hcComplex* devB = hc::am_alloc(sizeof(hcComplex) * K * N, handle->currentAccl, 0);
    hcComplex* devC = hc::am_alloc(sizeof(hcComplex) * M * N * batchSize, handle->currentAccl, 0);
    float* ablas = (float *)malloc(sizeof(float )* M * K * 2);
    float* bblas = (float *)malloc(sizeof(float )* K * N * 2);
    float* cblas = (float *)malloc(sizeof(float )* M * N * batchSize * 2);
    int k = 0;
    for(int i = 0; i < M * K; i++) {
                A[i].x = rand() % 10;
                A[i].y = rand() % 20;
                ablas[k++] = A[i].x;
                ablas[k++] = A[i].y;
    }
    k = 0;
    for(int i = 0; i < K * N;i++) {
                B[i].x = rand() % 15;
                B[i].y = rand() % 25;
                bblas[k++] = B[i].x;
                bblas[k++] = B[i].y;
    }
    k = 0;
    for(int i = 0; i < M * N * batchSize;i++) {
                C[i].x = rand() % 18;
                C[i].y = rand() % 28;
                cblas[k++] = C[i].x;
                cblas[k++] = C[i].y;
    }

  status = hcblasSetMatrix(handle, M, K, sizeof(hcComplex), A, 1, devA, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetMatrix(handle, K, N, sizeof(hcComplex), B, 1, devB, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);
  status = hcblasSetMatrix(handle, M, N * batchSize, sizeof(hcComplex), C, 1, devC, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);

  // NoTransA and NoTransB */           
  typeA = HCBLAS_OP_N;
  typeB = HCBLAS_OP_N;
  Transa = CblasNoTrans;
  Transb = CblasNoTrans;

    // Column major */
  lda = M; ldb = K ; ldc = M;
  status = hcblasCgemmBatched(handle, typeA, typeB, M, N, K, &cAlpha, devA, lda, devB, ldb, &cBeta, devC, ldc, batchSize);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);

  status = hcblasGetMatrix(handle, M, N * batchSize, sizeof(hcComplex), devC, 1, C, 1);
  EXPECT_EQ(status, HCBLAS_STATUS_SUCCESS);

  for(int i = 0; i < batchSize;i++)
         cblas_cgemm( order, Transa, Transb, M, N, K, &alpha, ablas, lda, bblas, ldb, &beta, cblas + i * M * N * 2, ldc );
  for(int i = 0, k = 0; ((i < M * N * batchSize) && ( k < M * N * batchSize * 2)) ; i++, k = k + 2) {
            EXPECT_EQ(C[i].x, cblas[k]);
            EXPECT_EQ(C[i].y, cblas[k+1]);
  }

   // HCBLAS_STATUS_NOT_INITIALIZED
  hcblasDestroy(&handle);
  status = hcblasCgemmBatched(handle, typeA, typeB, M, N, K, &cAlpha, devA, lda, devB, ldb, &cBeta, devC, ldc, batchSize);
  EXPECT_EQ(status, HCBLAS_STATUS_NOT_INITIALIZED);

  free(A);
  free(B);
  free(C);
  hc::am_free(devA);
  hc::am_free(devB);
  hc::am_free(devC);
  free(ablas);
  free(bblas);
  free(cblas);
}

