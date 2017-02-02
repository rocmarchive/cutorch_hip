#include "hcblaslib.h"
#include <cstdlib>
#include "gtest/gtest.h"
#include "hc_am.hpp"
#include "cblas.h"
#include "test_constants.h"

// code to check input given n size N
void func_check_sdot_with_input(long N) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
  int incX = 1;
  int incY = 1;
  long yOffset = 0;
  long xOffset = 0;
  float dothcblas;
  hcblasStatus status;
  float  dotcblas = 0.0;
  long lenx = 1 + (N - 1) * abs(incX);
  long leny = 1 + (N - 1) * abs(incY);
  float* X = (float*)calloc(lenx, sizeof(float));
  float* Y = (float*)calloc(leny, sizeof(float));
  accelerator_view accl_view = hc.currentAcclView;
  accelerator acc = hc.currentAccl;
  /*Implementation type I - Inputs and Outputs are HCC float array containers */
  float* devX = hc::am_alloc(sizeof(float) * lenx, acc, 0);
  float* devY = hc::am_alloc(sizeof(float) * leny, acc, 0);

  for(int i = 0; i < lenx; i++) {
    X[i] = rand() % 10;
  }

  for(int i = 0; i < leny; i++) {
    Y[i] =  rand() % 15;
  }

  accl_view.copy(X, devX, lenx * sizeof(float));
  accl_view.copy(Y, devY, leny * sizeof(float));
  /* Proper call */
  status = hc.hcblas_sdot(accl_view, N, devX, incX, xOffset, devY, incY, yOffset, dothcblas);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  dotcblas = cblas_sdot( N, X, incX, Y, incY);
  EXPECT_EQ(dothcblas, dotcblas);
  free(X);
  free(Y);
  hc::am_free(devX);
  hc::am_free(devY);
}

TEST(hcblas_sdot, return_correct_sdot_Implementation_type_1) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
  long N = 189;
  int incX = 1;
  int incY = 1;
  long yOffset = 0;
  long xOffset = 0;
  float dothcblas;
  hcblasStatus status;
  long lenx = 1 + (N - 1) * abs(incX);
  long leny = 1 + (N - 1) * abs(incY);
  float* X = (float*)calloc(lenx, sizeof(float));
  float* Y = (float*)calloc(leny, sizeof(float));
  accelerator_view accl_view = hc.currentAcclView;
  accelerator acc = hc.currentAccl;
  /* Implementation type I - Inputs and Outputs are HCC float array containers */
  float* devX = hc::am_alloc(sizeof(float) * lenx, acc, 0);
  float* devY = hc::am_alloc(sizeof(float) * leny, acc, 0);

  for(int i = 0; i < lenx; i++) {
    X[i] = rand() % 10;
  }

  for(int i = 0; i < leny; i++) {
    Y[i] =  rand() % 15;
  }

  accl_view.copy(X, devX, lenx * sizeof(float));
  accl_view.copy(Y, devY, leny * sizeof(float));
  /* Proper call */
  status = hc.hcblas_sdot(accl_view, N, devX, incX, xOffset, devY, incY, yOffset, dothcblas);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  /* X and Y are not properly allocated */
  float* devX1 = NULL;
  float* devY1 = NULL;
  status = hc.hcblas_sdot(accl_view, N, devX1, incX, xOffset, devY, incY, yOffset, dothcblas);
  EXPECT_EQ(status, HCBLAS_INVALID);
  status = hc.hcblas_sdot(accl_view, N, devX, incX, xOffset, devY1, incY, yOffset, dothcblas);
  EXPECT_EQ(status, HCBLAS_INVALID);
  /* N is 0 */
  N = 0;
  status = hc.hcblas_sdot(accl_view, N, devX, incX, xOffset, devY, incY, yOffset, dothcblas);
  EXPECT_EQ(status, HCBLAS_INVALID);
  /* incX is 0 */
  incX = 0;
  status = hc.hcblas_sdot(accl_view, N, devX, incX, xOffset, devY, incY, yOffset, dothcblas);
  EXPECT_EQ(status, HCBLAS_INVALID);
  /* incY is 0 */
  incX = 1;
  incY = 0;
  status = hc.hcblas_sdot(accl_view, N, devX, incX, xOffset, devY, incY, yOffset, dothcblas);
  EXPECT_EQ(status, HCBLAS_INVALID);
  free(X);
  free(Y);
  hc::am_free(devX);
  hc::am_free(devY);
}

TEST(hcblas_sdot, return_correct_sdot_Implementation_type_2) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
  long N = 189;
  int incX = 1;
  int incY = 1;
  long yOffset = 0;
  int batchSize = 128;
  long xOffset = 0;
  float dothcblas;
  hcblasStatus status;
  long X_batchOffset = N;
  long Y_batchOffset = N;
  long lenx = 1 + (N - 1) * abs(incX);
  long leny = 1 + (N - 1) * abs(incY);
  float* Xbatch = (float*)calloc(lenx * batchSize, sizeof(float));
  float* Ybatch = (float*)calloc(leny * batchSize, sizeof(float));
  accelerator_view accl_view = hc.currentAcclView;
  accelerator acc = hc.currentAccl;
  float* devXbatch = hc::am_alloc(sizeof(float) * lenx * batchSize, acc, 0);
  float* devYbatch = hc::am_alloc(sizeof(float) * leny * batchSize, acc, 0);

  /* Implementation type II - Inputs and Outputs are HCC float array containers with batch processing */
  for(int i = 0; i < lenx * batchSize; i++) {
    Xbatch[i] = rand() % 10;
  }

  for(int i = 0; i < leny * batchSize; i++) {
    Ybatch[i] =  rand() % 15;
  }

  accl_view.copy(Xbatch, devXbatch, lenx * batchSize * sizeof(float));
  accl_view.copy(Ybatch, devYbatch, leny * batchSize * sizeof(float));
  /* Proper call */
  status = hc.hcblas_sdot(accl_view, N, devXbatch, incX, xOffset, devYbatch, incY, yOffset, dothcblas, X_batchOffset, Y_batchOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  /* X and Y are not properly allocated */
  float* devX1 = NULL;
  float* devY1 = NULL;
  status = hc.hcblas_sdot(accl_view, N, devX1, incX, xOffset, devYbatch, incY, yOffset, dothcblas, X_batchOffset, Y_batchOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_INVALID);
  status = hc.hcblas_sdot(accl_view, N, devXbatch, incX, xOffset, devY1, incY, yOffset, dothcblas, X_batchOffset, Y_batchOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_INVALID);
  /* N is 0 */
  N = 0;
  status = hc.hcblas_sdot(accl_view, N, devXbatch, incX, xOffset, devYbatch, incY, yOffset, dothcblas, X_batchOffset, Y_batchOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_INVALID);
  /* incX is 0 */
  incX = 0;
  status = hc.hcblas_sdot(accl_view, N, devXbatch, incX, xOffset, devYbatch, incY, yOffset, dothcblas, X_batchOffset, Y_batchOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_INVALID);
  /* incY is 0 */
  incX = 1;
  incY = 0;
  status = hc.hcblas_sdot(accl_view, N, devXbatch, incX, xOffset, devYbatch, incY, yOffset, dothcblas, X_batchOffset, Y_batchOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_INVALID);
  free(Xbatch);
  free(Ybatch);
  hc::am_free(devXbatch);
  hc::am_free(devYbatch);
}
// SDOT functionality check for different sizes
// vvlarge test
TEST(hcblas_sdot, func_correct_sdot_vvlargeN_Implementation_type_1) {
  long input = gen_vvlarge();
  func_check_sdot_with_input(input);
}

// vlarge test
TEST(hcblas_sdot, func_correct_sdot_vlargeN_Implementation_type_1) {
  long input = gen_vlarge();
  func_check_sdot_with_input(input);
}

// large test
TEST(hcblas_sdot, func_correct_sdot_largeN_Implementation_type_1) {
  long input = gen_large();
  func_check_sdot_with_input(input);
}

// REGULAR test
TEST(hcblas_sdot, func_correct_sdot_regularN_Implementation_type_1) {
  long input = gen_regular();
  func_check_sdot_with_input(input);
}

// SMALL test
TEST(hcblas_sdot, func_correct_sdot_smallN_Implementation_type_1) {
  long input = gen_small();
  func_check_sdot_with_input(input);
}

// VSMALL test
TEST(hcblas_sdot, func_correct_sdot_vsmallN_Implementation_type_1) {
  long input = gen_vsmall();
  func_check_sdot_with_input(input);
}

// VV_SMALL test
TEST(hcblas_sdot, func_correct_sdot_vvsmallN_Implementation_type_1) {
  long input = gen_vsmall();
  func_check_sdot_with_input(input);
}

// Func to check batch sdot gven inut size
void func_check_sdot_batch_with_input(long N) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
  int incX = 1;
  int incY = 1;
  long yOffset = 0;
  int batchSize = 128;
  long xOffset = 0;
  float dothcblas;
  float  dotcblas = 0.0;
  float* dotcblastemp = (float*)calloc(batchSize, sizeof(float));
  hcblasStatus status;
  long X_batchOffset = N;
  long Y_batchOffset = N;
  long lenx = 1 + (N - 1) * abs(incX);
  long leny = 1 + (N - 1) * abs(incY);
  float* Xbatch = (float*)calloc(lenx * batchSize, sizeof(float));
  float* Ybatch = (float*)calloc(leny * batchSize, sizeof(float));
  accelerator_view accl_view = hc.currentAcclView;
  accelerator acc = hc.currentAccl;
  float* devXbatch = hc::am_alloc(sizeof(float) * lenx * batchSize, acc, 0);
  float* devYbatch = hc::am_alloc(sizeof(float) * leny * batchSize, acc, 0);

  /* Implementation type II - Inputs and Outputs are HCC float array containers with batch processing */
  for(int i = 0; i < lenx * batchSize; i++) {
    Xbatch[i] = rand() % 10;
  }

  for(int i = 0; i < leny * batchSize; i++) {
    Ybatch[i] =  rand() % 15;
  }

  accl_view.copy(Xbatch, devXbatch, lenx * batchSize * sizeof(float));
  accl_view.copy(Ybatch, devYbatch, leny * batchSize * sizeof(float));
  /* Proper call */
  status = hc.hcblas_sdot(accl_view, N, devXbatch, incX, xOffset, devYbatch, incY, yOffset, dothcblas, X_batchOffset, Y_batchOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);

  for(int i = 0; i < batchSize; i++) {
    dotcblastemp[i] = cblas_sdot( N, Xbatch + i * N, incX, Ybatch + i * N, incY);
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
TEST(hcblas_sdot, func_correct_sdot_batch_smallN_Implementation_type_2) {
  long input = gen_small();
  func_check_sdot_batch_with_input(input);
}

// VSMALL test
TEST(hcblas_sdot, func_correct_sdot_batch_vsmallN_Implementation_type_2) {
  long input = gen_vsmall();
  func_check_sdot_batch_with_input(input);
}

// VV_SMALL test
TEST(hcblas_sdot, func_correct_sdot_batch_vvsmallN_Implementation_type_2) {
  long input = gen_vsmall();
  func_check_sdot_batch_with_input(input);
}

