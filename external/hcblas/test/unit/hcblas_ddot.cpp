#include "hcblaslib.h"
#include <cstdlib>
#include "gtest/gtest.h"
#include "hc_am.hpp"
#include "cblas.h"
#include "test_constants.h"

// code to check input given n size N
void func_check_ddot_with_input(long N) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
  int incX = 1;
  int incY = 1;
  long yOffset = 0;
  long xOffset = 0;
  double dothcblas;
  hcblasStatus status;
  double  dotcblas = 0.0;
  long lenx = 1 + (N - 1) * abs(incX);
  long leny = 1 + (N - 1) * abs(incY);
  double* X = (double*)calloc(lenx, sizeof(double));
  double* Y = (double*)calloc(leny, sizeof(double));
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;
  /*Implementation type I - Inputs and Outputs are HCC double array containers */
  double* devX = hc::am_alloc(sizeof(double) * lenx, acc, 0);
  double* devY = hc::am_alloc(sizeof(double) * leny, acc, 0);

  for(int i = 0; i < lenx; i++) {
    X[i] = rand() % 10;
  }

  for(int i = 0; i < leny; i++) {
    Y[i] =  rand() % 15;
  }

  accl_view.copy(X, devX, lenx * sizeof(double));
  accl_view.copy(Y, devY, leny * sizeof(double));
  /* Proper call */
  status = hc.hcblas_ddot(accl_view, N, devX, incX, xOffset, devY, incY, yOffset, dothcblas);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  dotcblas = cblas_ddot( N, X, incX, Y, incY);
  EXPECT_EQ(dothcblas, dotcblas);
  free(X);
  free(Y);
  hc::am_free(devX);
  hc::am_free(devY);
}

TEST(hcblas_ddot, return_correct_ddot_Implementation_type_1) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
  long N = 189;
  int incX = 1;
  int incY = 1;
  long yOffset = 0;
  long xOffset = 0;
  double dothcblas;
  hcblasStatus status;
  long lenx = 1 + (N - 1) * abs(incX);
  long leny = 1 + (N - 1) * abs(incY);
  double* X = (double*)calloc(lenx, sizeof(double));
  double* Y = (double*)calloc(leny, sizeof(double));
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;
  /* Implementation type I - Inputs and Outputs are HCC double array containers */
  double* devX = hc::am_alloc(sizeof(double) * lenx, acc, 0);
  double* devY = hc::am_alloc(sizeof(double) * leny, acc, 0);

  for(int i = 0; i < lenx; i++) {
    X[i] = rand() % 10;
  }

  for(int i = 0; i < leny; i++) {
    Y[i] =  rand() % 15;
  }

  accl_view.copy(X, devX, lenx * sizeof(double));
  accl_view.copy(Y, devY, leny * sizeof(double));
  /* Proper call */
  status = hc.hcblas_ddot(accl_view, N, devX, incX, xOffset, devY, incY, yOffset, dothcblas);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  /* X and Y are not properly allocated */
  double* devX1 = NULL;
  double* devY1 = NULL;
  status = hc.hcblas_ddot(accl_view, N, devX1, incX, xOffset, devY, incY, yOffset, dothcblas);
  EXPECT_EQ(status, HCBLAS_INVALID);
  status = hc.hcblas_ddot(accl_view, N, devX, incX, xOffset, devY1, incY, yOffset, dothcblas);
  EXPECT_EQ(status, HCBLAS_INVALID);
  /* N is 0 */
  N = 0;
  status = hc.hcblas_ddot(accl_view, N, devX, incX, xOffset, devY, incY, yOffset, dothcblas);
  EXPECT_EQ(status, HCBLAS_INVALID);
  /* incX is 0 */
  incX = 0;
  status = hc.hcblas_ddot(accl_view, N, devX, incX, xOffset, devY, incY, yOffset, dothcblas);
  EXPECT_EQ(status, HCBLAS_INVALID);
  /* incY is 0 */
  incX = 1;
  incY = 0;
  status = hc.hcblas_ddot(accl_view, N, devX, incX, xOffset, devY, incY, yOffset, dothcblas);
  EXPECT_EQ(status, HCBLAS_INVALID);
  free(X);
  free(Y);
  hc::am_free(devX);
  hc::am_free(devY);
}

TEST(hcblas_ddot, return_correct_ddot_Implementation_type_2) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
  long N = 189;
  int incX = 1;
  int incY = 1;
  long yOffset = 0;
  int batchSize = 128;
  long xOffset = 0;
  double dothcblas;
  hcblasStatus status;
  long X_batchOffset = N;
  long Y_batchOffset = N;
  long lenx = 1 + (N - 1) * abs(incX);
  long leny = 1 + (N - 1) * abs(incY);
  double* Xbatch = (double*)calloc(lenx * batchSize, sizeof(double));
  double* Ybatch = (double*)calloc(leny * batchSize, sizeof(double));
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;
  double* devXbatch = hc::am_alloc(sizeof(double) * lenx * batchSize, acc, 0);
  double* devYbatch = hc::am_alloc(sizeof(double) * leny * batchSize, acc, 0);

  /* Implementation type II - Inputs and Outputs are HCC double array containers with batch processing */
  for(int i = 0; i < lenx * batchSize; i++) {
    Xbatch[i] = rand() % 10;
  }

  for(int i = 0; i < leny * batchSize; i++) {
    Ybatch[i] =  rand() % 15;
  }

  accl_view.copy(Xbatch, devXbatch, lenx * batchSize * sizeof(double));
  accl_view.copy(Ybatch, devYbatch, leny * batchSize * sizeof(double));
  /* Proper call */
  status = hc.hcblas_ddot(accl_view, N, devXbatch, incX, xOffset, devYbatch, incY, yOffset, dothcblas, X_batchOffset, Y_batchOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  /* X and Y are not properly allocated */
  double* devX1 = NULL;
  double* devY1 = NULL;
  status = hc.hcblas_ddot(accl_view, N, devX1, incX, xOffset, devYbatch, incY, yOffset, dothcblas, X_batchOffset, Y_batchOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_INVALID);
  status = hc.hcblas_ddot(accl_view, N, devXbatch, incX, xOffset, devY1, incY, yOffset, dothcblas, X_batchOffset, Y_batchOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_INVALID);
  /* N is 0 */
  N = 0;
  status = hc.hcblas_ddot(accl_view, N, devXbatch, incX, xOffset, devYbatch, incY, yOffset, dothcblas, X_batchOffset, Y_batchOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_INVALID);
  /* incX is 0 */
  incX = 0;
  status = hc.hcblas_ddot(accl_view, N, devXbatch, incX, xOffset, devYbatch, incY, yOffset, dothcblas, X_batchOffset, Y_batchOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_INVALID);
  /* incY is 0 */
  incX = 1;
  incY = 0;
  status = hc.hcblas_ddot(accl_view, N, devXbatch, incX, xOffset, devYbatch, incY, yOffset, dothcblas, X_batchOffset, Y_batchOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_INVALID);
  free(Xbatch);
  free(Ybatch);
  hc::am_free(devXbatch);
  hc::am_free(devYbatch);
}
// SDOT functionality check for different sizes
// vvlarge test
TEST(hcblas_ddot, func_correct_ddot_vvlargeN_Implementation_type_1) {
  long input = gen_vvlarge();
  func_check_ddot_with_input(input);
}

// vlarge test
TEST(hcblas_ddot, func_correct_ddot_vlargeN_Implementation_type_1) {
  long input = gen_vlarge();
  func_check_ddot_with_input(input);
}

// large test
TEST(hcblas_ddot, func_correct_ddot_largeN_Implementation_type_1) {
  long input = gen_large();
  func_check_ddot_with_input(input);
}

// REGULAR test
TEST(hcblas_ddot, func_correct_ddot_regularN_Implementation_type_1) {
  long input = gen_regular();
  func_check_ddot_with_input(input);
}

// SMALL test
TEST(hcblas_ddot, func_correct_ddot_smallN_Implementation_type_1) {
  long input = gen_small();
  func_check_ddot_with_input(input);
}

// VSMALL test
TEST(hcblas_ddot, func_correct_ddot_vsmallN_Implementation_type_1) {
  long input = gen_vsmall();
  func_check_ddot_with_input(input);
}

// VV_SMALL test
TEST(hcblas_ddot, func_correct_ddot_vvsmallN_Implementation_type_1) {
  long input = gen_vsmall();
  func_check_ddot_with_input(input);
}

// Func to check batch ddot gven inut size
void func_check_ddot_batch_with_input(long N) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
  int incX = 1;
  int incY = 1;
  long yOffset = 0;
  int batchSize = 128;
  long xOffset = 0;
  double dothcblas;
  double  dotcblas = 0.0;
  double* dotcblastemp = (double*)calloc(batchSize, sizeof(double));
  hcblasStatus status;
  long X_batchOffset = N;
  long Y_batchOffset = N;
  long lenx = 1 + (N - 1) * abs(incX);
  long leny = 1 + (N - 1) * abs(incY);
  double* Xbatch = (double*)calloc(lenx * batchSize, sizeof(double));
  double* Ybatch = (double*)calloc(leny * batchSize, sizeof(double));
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;
  double* devXbatch = hc::am_alloc(sizeof(double) * lenx * batchSize, acc, 0);
  double* devYbatch = hc::am_alloc(sizeof(double) * leny * batchSize, acc, 0);

  /* Implementation type II - Inputs and Outputs are HCC double array containers with batch processing */
  for(int i = 0; i < lenx * batchSize; i++) {
    Xbatch[i] = rand() % 10;
  }

  for(int i = 0; i < leny * batchSize; i++) {
    Ybatch[i] =  rand() % 15;
  }

  accl_view.copy(Xbatch, devXbatch, lenx * batchSize * sizeof(double));
  accl_view.copy(Ybatch, devYbatch, leny * batchSize * sizeof(double));
  /* Proper call */
  status = hc.hcblas_ddot(accl_view, N, devXbatch, incX, xOffset, devYbatch, incY, yOffset, dothcblas, X_batchOffset, Y_batchOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);

  for(int i = 0; i < batchSize; i++) {
    dotcblastemp[i] = cblas_ddot( N, Xbatch + i * N, incX, Ybatch + i * N, incY);
    dotcblas += dotcblastemp[i];
  }

  EXPECT_EQ(dothcblas, dotcblas);
  free(Xbatch);
  free(Ybatch);
  hc::am_free(devXbatch);
  hc::am_free(devYbatch);
}

// SDOT batch functionality check for different sizes

// SMALL test
TEST(hcblas_ddot, func_correct_ddot_batch_smallN_Implementation_type_2) {
  long input = gen_small();
  func_check_ddot_batch_with_input(input);
}

// VSMALL test
TEST(hcblas_ddot, func_correct_ddot_batch_vsmallN_Implementation_type_2) {
  long input = gen_vsmall();
  func_check_ddot_batch_with_input(input);
}

// VV_SMALL test
TEST(hcblas_ddot, func_correct_ddot_batch_vvsmallN_Implementation_type_2) {
  long input = gen_vsmall();
  func_check_ddot_batch_with_input(input);
}

