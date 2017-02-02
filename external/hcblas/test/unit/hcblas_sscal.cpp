#include "hcblaslib.h"
#include <cstdlib> 
#include "gtest/gtest.h"
#include "hc_am.hpp"
#include "cblas.h"

TEST(hcblas_sscal, return_correct_sscal_Implementation_type_1) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
   int N = 23;
   int incX = 1;
   long xOffset = 0;
   float alpha = 1;
   hcblasStatus status; 
   long lenx = 1 + (N-1) * abs(incX);
   float *X = (float*)calloc(lenx, sizeof(float));//host input
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;
/* Implementation type I - Inputs and Outputs are HCC device pointers */
   float* devX = hc::am_alloc(sizeof(float) * lenx, acc, 0);
   for(int i = 0; i < lenx; i++){
            X[i] = rand() % 10;
   }
   float* devX1 = NULL;
   accl_view.copy(X, devX, lenx * sizeof(float));
   /* devX1 is NULL */
   status = hc.hcblas_sscal(accl_view, N, alpha, devX1, incX, xOffset);
   EXPECT_EQ(status, HCBLAS_INVALID);
   /* alpha is some scalar */
   status = hc.hcblas_sscal(accl_view, N, alpha, devX, incX, xOffset);
   EXPECT_EQ(status, HCBLAS_SUCCEEDS);
   /* N is 0 */
   N = 0;
   status = hc.hcblas_sscal(accl_view, N, alpha, devX, incX, xOffset);
   EXPECT_EQ(status, HCBLAS_INVALID);
   /* incX is 0 */
   incX = 0;
   status = hc.hcblas_sscal(accl_view, N, alpha, devX, incX, xOffset);
   EXPECT_EQ(status, HCBLAS_INVALID); 
}

TEST(hcblas_sscal, function_correct_sscal_Implementation_type_1) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
   int N = 23;
   int incX = 1;
   long xOffset = 0;
   float alpha = 1;
   hcblasStatus status;
   long lenx = 1 + (N-1) * abs(incX);
   float *X = (float*)calloc(lenx, sizeof(float));//host input
   float *Xcblas = (float*)calloc(lenx, sizeof(float));
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;
 /* Implementation type I - Inputs and Outputs are HCC device pointers */
   float* devX = hc::am_alloc(sizeof(float) * lenx, acc, 0);
   for(int i = 0; i < lenx; i++){
           X[i] = rand() % 10;
	   Xcblas[i] = X[i];
   }
   accl_view.copy(X, devX, lenx * sizeof(float));
   status = hc.hcblas_sscal(accl_view, N, alpha, devX, incX, xOffset);
   EXPECT_EQ(status, HCBLAS_SUCCEEDS);
   accl_view.copy(devX, X, lenx * sizeof(float));
   cblas_sscal( N, alpha, Xcblas, incX );
   for(int i = 0; i < lenx ; i++){
	   EXPECT_EQ(X[i], Xcblas[i]);
   }
/* alpha is 0 */
   alpha = 0;
   accl_view.copy(X, devX, lenx * sizeof(float));
   status = hc.hcblas_sscal(accl_view, N, alpha, devX, incX, xOffset);
   EXPECT_EQ(status, HCBLAS_SUCCEEDS);
   accl_view.copy(devX, X, lenx * sizeof(float));
   cblas_sscal( N, alpha, Xcblas, incX );
   for(int i = 0; i < lenx ; i++){
           EXPECT_EQ(X[i], Xcblas[i]);
   }
}

TEST(hcblas_sscal, return_correct_sscal_Implementation_type_2) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
   int N = 19;
   int incX = 1;
   int batchSize = 32;
   long xOffset = 0;
   float alpha = 1;
   hcblasStatus status;
   long X_batchOffset = N; 
   long lenx = 1 + (N-1) * abs(incX);
   float *Xbatch = (float*)calloc(lenx * batchSize, sizeof(float));//host input
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;
   float* devXbatch = hc::am_alloc(sizeof(float) * lenx * batchSize, acc, 0);
   float* devX1batch = NULL; 
/* Implementation type II - Inputs and Outputs are HCC device pointers with batch processing */
   for(int i = 0;i < lenx * batchSize;i++){
            Xbatch[i] = rand() % 10;
   }
   accl_view.copy(Xbatch, devXbatch, lenx * batchSize * sizeof(float));
   /* x1 is NULL */
   status= hc.hcblas_sscal(accl_view, N, alpha, devX1batch, incX, xOffset, X_batchOffset, batchSize);
   EXPECT_EQ(status, HCBLAS_INVALID);
   /* alpha is some scalar */
   status= hc.hcblas_sscal(accl_view, N, alpha, devXbatch, incX, xOffset, X_batchOffset, batchSize);
   EXPECT_EQ(status, HCBLAS_SUCCEEDS);
   /* N is 0 */
   N = 0;
   status= hc.hcblas_sscal(accl_view, N, alpha, devXbatch, incX, xOffset, X_batchOffset, batchSize);
   EXPECT_EQ(status, HCBLAS_INVALID);
   /* incX is 0 */
   incX = 0;
   status= hc.hcblas_sscal(accl_view, N, alpha, devXbatch, incX, xOffset, X_batchOffset, batchSize);
   EXPECT_EQ(status, HCBLAS_INVALID);
}

TEST(hcblas_sscal, function_correct_sscal_Implementation_type_2) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
   int N = 19;
   int incX = 1;
   int batchSize = 32;
   long xOffset = 0;
   float alpha = 1;
   hcblasStatus status;
   long X_batchOffset = N;
   long lenx = 1 + (N-1) * abs(incX);
   float *Xbatch = (float*)calloc(lenx * batchSize, sizeof(float));//host input
   float *Xcblasbatch = (float*)calloc(lenx * batchSize, sizeof(float));
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;
   float* devXbatch = hc::am_alloc(sizeof(float) * lenx * batchSize, acc, 0);
   /* Implementation type II - Inputs and Outputs are HCC device pointers with batch processing */
   for(int i = 0;i < lenx * batchSize;i++){
               Xbatch[i] = rand() % 10;
	       Xcblasbatch[i] =  Xbatch[i];
   }
   accl_view.copy(Xbatch, devXbatch, lenx * batchSize * sizeof(float));
   status= hc.hcblas_sscal(accl_view, N, alpha, devXbatch, incX, xOffset, X_batchOffset, batchSize);
   EXPECT_EQ(status, HCBLAS_SUCCEEDS);
   accl_view.copy(devXbatch, Xbatch, lenx * batchSize * sizeof(float));
   for(int i = 0; i < batchSize; i++)
        cblas_sscal( N, alpha, Xcblasbatch + i * N, incX);
   for(int i =0; i < lenx * batchSize; i ++){
	EXPECT_EQ(Xbatch[i], Xcblasbatch[i]);
   }
   /* alpha is 0 */
   alpha = 0;
   accl_view.copy(Xbatch, devXbatch, lenx * batchSize * sizeof(float));
   status= hc.hcblas_sscal(accl_view, N, alpha, devXbatch, incX, xOffset, X_batchOffset, batchSize);
   EXPECT_EQ(status, HCBLAS_SUCCEEDS);
   accl_view.copy(devXbatch, Xbatch, lenx * batchSize * sizeof(float));
   for(int i = 0; i < batchSize; i++)
        cblas_sscal( N, alpha, Xcblasbatch + i * N, incX);
   for(int i =0; i < lenx * batchSize; i ++){
        EXPECT_EQ(Xbatch[i], Xcblasbatch[i]);
   }
}
