#include "hcblaslib.h"
#include <cstdlib> 
#include "gtest/gtest.h"
#include "hc_am.hpp"
#include "cblas.h"

TEST(hcblas_sasum, return_correct_sasum_Implementation_type_1) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
   int N = 119;
   int incX = 1;
   long xOffset = 0;
   float asumhcblas;
   hcblasStatus status; 
   long lenx = 1 + (N-1) * abs(incX);
   float *X = (float*)calloc(lenx, sizeof(float));
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;
/* Implementation type I - Inputs and Outputs are HCC device pointers */
   float* devX = hc::am_alloc(sizeof(float) * lenx, acc, 0);
   for(int i = 0; i < lenx; i++){
            X[i] = rand() % 10;
   }
   accl_view.copy(X, devX, lenx * sizeof(float));
   /* Proper call */
   status = hc.hcblas_sasum(accl_view, N, devX, incX, xOffset, &asumhcblas);
   EXPECT_EQ(status, HCBLAS_SUCCEEDS);
   /* X not properly allocated */
   float *devX1 = NULL;
   status = hc.hcblas_sasum(accl_view, N, devX1, incX, xOffset, &asumhcblas);
   EXPECT_EQ(status, HCBLAS_INVALID);
   /* N is 0 */
   N = 0;
   status = hc.hcblas_sasum(accl_view, N, devX, incX, xOffset, &asumhcblas);
   EXPECT_EQ(status, HCBLAS_INVALID);
   /* incX is 0 */
   incX = 0;
   status = hc.hcblas_sasum(accl_view, N, devX, incX, xOffset, &asumhcblas);
   EXPECT_EQ(status, HCBLAS_INVALID);  
   free(X);
   hc::am_free(devX);
}

TEST(hcblas_sasum, func_correct_sasum_Implementation_type_1){ 
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
   int N = 119;
   int incX = 1;
   long xOffset = 0;
   float asumhcblas;
   float asumcblas = 0.0;
   hcblasStatus status;
   long lenx = 1 + (N-1) * abs(incX);
   float *X = (float*)calloc(lenx, sizeof(float));
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;
/* Implementation type I - Inputs and Outputs are HCC device pointers */
   float* devX = hc::am_alloc(sizeof(float) * lenx, acc, 0);
   for(int i = 0; i < lenx; i++){
            X[i] = rand() % 10;
   }
   accl_view.copy(X, devX, lenx * sizeof(float));
   /* Proper call */
   status = hc.hcblas_sasum(accl_view, N, devX, incX, xOffset, &asumhcblas);
   EXPECT_EQ(status, HCBLAS_SUCCEEDS);
   asumcblas = cblas_sasum( N, X, incX);
   EXPECT_EQ(asumhcblas, asumcblas);
   free(X);
   hc::am_free(devX);
}

TEST(hcblas_sasum, return_correct_sasum_Implementation_type_2) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
   int N = 119;
   int incX = 1;
   int batchSize = 128;
   long xOffset = 0;
   float asumhcblas;
   hcblasStatus status;
   long X_batchOffset = N; 
   long lenx = 1 + (N-1) * abs(incX);
   float *Xbatch = (float*)calloc(lenx * batchSize, sizeof(float));
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;
   float* devXbatch = hc::am_alloc(sizeof(float) * lenx * batchSize, acc, 0); 
/* Implementation type II - Inputs and Outputs are HCC device pointers with batch processing */
   for(int i = 0;i < lenx * batchSize;i++){
            Xbatch[i] = rand() % 10;
   }
   accl_view.copy(Xbatch, devXbatch, lenx * batchSize * sizeof(float));
   /* Proper call */
   status= hc.hcblas_sasum(accl_view, N, devXbatch, incX, xOffset, &asumhcblas, X_batchOffset, batchSize);
   EXPECT_EQ(status, HCBLAS_SUCCEEDS);
   /* X is not properly allocated */
   float *devX1 = NULL;
   status= hc.hcblas_sasum(accl_view, N, devX1, incX, xOffset, &asumhcblas, X_batchOffset, batchSize);
   EXPECT_EQ(status, HCBLAS_INVALID);
   /* N is 0 */
   N = 0;
   status= hc.hcblas_sasum(accl_view, N, devXbatch, incX, xOffset, &asumhcblas, X_batchOffset, batchSize);
   EXPECT_EQ(status, HCBLAS_INVALID);
   /* incX is 0 */
   incX = 0;
   status= hc.hcblas_sasum(accl_view, N, devXbatch, incX, xOffset, &asumhcblas, X_batchOffset, batchSize);
   EXPECT_EQ(status, HCBLAS_INVALID);
   free(Xbatch);
   hc::am_free(devXbatch);
}

TEST(hcblas_sasum, func_correct_sasum_Implementation_type_2) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
   int N = 119;
   int incX = 1;
   int batchSize = 128;
   long xOffset = 0;
   float asumhcblas;
   float asumcblas = 0.0;
   float *asumcblastemp = (float*)calloc(batchSize, sizeof(float));
   hcblasStatus status;
   long X_batchOffset = N;
   long lenx = 1 + (N-1) * abs(incX);
   float *Xbatch = (float*)calloc(lenx * batchSize, sizeof(float));
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;
   float* devXbatch = hc::am_alloc(sizeof(float) * lenx * batchSize, acc, 0);
/* Implementation type II - Inputs and Outputs are HCC float array containers with batch processing */
   for(int i = 0;i < lenx * batchSize;i++){
            Xbatch[i] = rand() % 10;
   }
   accl_view.copy(Xbatch, devXbatch, lenx * batchSize * sizeof(float));
   /* Proper call */
   status= hc.hcblas_sasum(accl_view, N, devXbatch, incX, xOffset, &asumhcblas, X_batchOffset, batchSize);
   EXPECT_EQ(status, HCBLAS_SUCCEEDS);
   for(int i = 0; i < batchSize; i++) {
                asumcblastemp[i] = cblas_sasum( N, Xbatch + i * N, incX);
                asumcblas += asumcblastemp[i];
   }
   EXPECT_EQ(asumhcblas, asumcblas);
   free(Xbatch);
   hc::am_free(devXbatch);
} 

