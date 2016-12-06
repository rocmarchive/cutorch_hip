#include "hcblaslib.h"
#include <cstdlib> 
#include "gtest/gtest.h"
#include "hc_am.hpp"
#include "cblas.h"

TEST(hcblas_daxpy, return_correct_daxpy_Implementation_type_1) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
   int N = 279;
   int incX = 1;
   int incY = 1;
   long yOffset = 0;
   long xOffset = 0;
   hcblasStatus status; 
   double alpha = 1;
   long lenx = 1 + (N-1) * abs(incX);
   long leny = 1 + (N-1) * abs(incY);
   double *X = (double*)calloc(lenx, sizeof(double));
   double *Y = (double*)calloc(leny, sizeof(double));
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;
/* Implementation type I - Inputs and Outputs are HCC device pointers */
   double* devX = hc::am_alloc(sizeof(double) * lenx, acc, 0);
   double* devY = hc::am_alloc(sizeof(double) * leny, acc, 0);
   for(int i = 0; i < lenx; i++){
            X[i] = rand() % 10;
   }
   for(int i = 0; i < leny; i++){
            Y[i] =  rand() % 15;
   }
   accl_view.copy(X, devX, lenx * sizeof(double));
   accl_view.copy(Y, devY, leny * sizeof(double));
   /* Proper call */
   status = hc.hcblas_daxpy(accl_view, N, alpha, devX, incX, devY, incY , xOffset, yOffset);
   EXPECT_EQ(status, HCBLAS_SUCCEEDS);
   /* X and Y are not properly allocated */
   double *devX1 = NULL;
   double *devY1 = NULL;
   status = hc.hcblas_daxpy(accl_view, N, alpha, devX1, incX, devY, incY , xOffset, yOffset);
   EXPECT_EQ(status, HCBLAS_INVALID);
   status = hc.hcblas_daxpy(accl_view, N, alpha, devX, incX, devY1, incY , xOffset, yOffset);
   EXPECT_EQ(status, HCBLAS_INVALID);
   /* alpha is 0 */
   alpha = 0;
   status = hc.hcblas_daxpy(accl_view, N, alpha, devX, incX, devY, incY , xOffset, yOffset);
   EXPECT_EQ(status, HCBLAS_SUCCEEDS);
   /* N is 0 */
   N = 0;
   status = hc.hcblas_daxpy(accl_view, N, alpha, devX, incX, devY, incY , xOffset, yOffset);
   EXPECT_EQ(status, HCBLAS_INVALID);
   /* incX is 0 */
   incX = 0;
   status = hc.hcblas_daxpy(accl_view, N, alpha, devX, incX, devY, incY , xOffset, yOffset);
   EXPECT_EQ(status, HCBLAS_INVALID); 
   /* incY is 0 */
   incX = 1; incY = 0;
   status = hc.hcblas_daxpy(accl_view, N, alpha, devX, incX, devY, incY , xOffset, yOffset);
   EXPECT_EQ(status, HCBLAS_INVALID); 
   free(X);
   free(Y);
   hc::am_free(devX);
   hc::am_free(devY);
}

TEST(hcblas_daxpy, func_correct_daxpy_Implementation_type_1) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
   int N = 279;
   int incX = 1;
   int incY = 1;
   long yOffset = 0;
   long xOffset = 0;
   hcblasStatus status; 
   double alpha = 1;
   long lenx = 1 + (N-1) * abs(incX);
   long leny = 1 + (N-1) * abs(incY);
   double *X = (double*)calloc(lenx, sizeof(double));
   double *Y = (double*)calloc(leny, sizeof(double));
   double *Ycblas = (double*)calloc(N, sizeof(double));
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;
/* Implementation type I - Inputs and Outputs are HCC device pointers */
   double* devX = hc::am_alloc(sizeof(double) * lenx, acc, 0);
   double* devY = hc::am_alloc(sizeof(double) * leny, acc, 0);
   for(int i = 0; i < lenx; i++){
            X[i] = rand() % 10;
   }
   for(int i = 0; i < leny; i++){
            Y[i] =  rand() % 15;
            Ycblas[i] = Y[i];
   }
   accl_view.copy(X, devX, lenx * sizeof(double));
   accl_view.copy(Y, devY, leny * sizeof(double));
   /* Proper call */
   status = hc.hcblas_daxpy(accl_view, N, alpha, devX, incX, devY, incY , xOffset, yOffset);
   EXPECT_EQ(status, HCBLAS_SUCCEEDS);
   accl_view.copy(devY, Y, leny * sizeof(double));
   cblas_daxpy( N, alpha, X, incX, Ycblas, incY );
   for(int i = 0; i < leny ; i++)
      EXPECT_EQ(Y[i], Ycblas[i]);
   free(X);
   free(Y);
   free(Ycblas);
   hc::am_free(devX);
   hc::am_free(devY);
}

TEST(hcblas_daxpy, return_correct_daxpy_Implementation_type_2) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
   int N = 279;
   int incX = 1;
   int incY = 1;
   long yOffset = 0;
   int batchSize = 128;
   long xOffset = 0;
   double alpha = 1;
   hcblasStatus status;
   long X_batchOffset = N; 
   long Y_batchOffset = N;
   long lenx = 1 + (N-1) * abs(incX);
   long leny = 1 + (N-1) * abs(incY);
   double *Xbatch = (double*)calloc(lenx * batchSize, sizeof(double));
   double *Ybatch = (double*)calloc(leny * batchSize, sizeof(double));
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;
   double* devXbatch = hc::am_alloc(sizeof(double) * lenx * batchSize, acc, 0);
   double* devYbatch = hc::am_alloc(sizeof(double) * leny * batchSize, acc, 0);   
/* Implementation type II - Inputs and Outputs are HCC double array containers with batch processing */
   for(int i = 0;i < lenx * batchSize;i++){
            Xbatch[i] = rand() % 10;
   }
   for(int i = 0;i < leny * batchSize;i++){
            Ybatch[i] =  rand() % 15;
   }
   accl_view.copy(Xbatch, devXbatch, lenx * batchSize * sizeof(double));
   accl_view.copy(Ybatch, devYbatch, leny * batchSize * sizeof(double));
   /* Proper call */
   status= hc.hcblas_daxpy(accl_view, N, alpha, devXbatch, incX, X_batchOffset, devYbatch, incY, Y_batchOffset, xOffset, yOffset, batchSize);
   EXPECT_EQ(status, HCBLAS_SUCCEEDS);
   /* alpha is 0*/
   alpha = 0;
   status= hc.hcblas_daxpy(accl_view, N, alpha, devXbatch, incX, X_batchOffset, devYbatch, incY, Y_batchOffset, xOffset, yOffset, batchSize);
   EXPECT_EQ(status, HCBLAS_SUCCEEDS);
   /* X and Y are not properly allocated */
   double *devX1 = NULL;
   double *devY1 = NULL;
   status= hc.hcblas_daxpy(accl_view, N, alpha, devX1, incX, X_batchOffset, devYbatch, incY, Y_batchOffset, xOffset, yOffset, batchSize);
   EXPECT_EQ(status, HCBLAS_INVALID);
   status= hc.hcblas_daxpy(accl_view, N, alpha, devXbatch, incX, X_batchOffset, devY1, incY, Y_batchOffset, xOffset, yOffset, batchSize);
   EXPECT_EQ(status, HCBLAS_INVALID);
   /* N is 0 */
   N = 0;
   status= hc.hcblas_daxpy(accl_view, N, alpha, devXbatch, incX, X_batchOffset, devYbatch, incY, Y_batchOffset, xOffset, yOffset, batchSize);
   EXPECT_EQ(status, HCBLAS_INVALID);
   /* incX is 0 */
   incX = 0;
   status= hc.hcblas_daxpy(accl_view, N, alpha, devXbatch, incX, X_batchOffset, devYbatch, incY, Y_batchOffset, xOffset, yOffset, batchSize);
   EXPECT_EQ(status, HCBLAS_INVALID);
   /* incY is 0 */
   incX = 1; incY = 0;
   status= hc.hcblas_daxpy(accl_view, N, alpha, devXbatch, incX, X_batchOffset, devYbatch, incY, Y_batchOffset, xOffset, yOffset, batchSize);
   EXPECT_EQ(status, HCBLAS_INVALID);
   free(Xbatch);
   free(Ybatch);
   hc::am_free(devXbatch);
   hc::am_free(devYbatch);
}

TEST(hcblas_daxpy, func_correct_daxpy_Implementation_type_2) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
   int N = 279;
   int incX = 1;
   int incY = 1;
   long yOffset = 0;
   int batchSize = 128;
   long xOffset = 0;
   double alpha = 1;
   hcblasStatus status;
   long X_batchOffset = N;
   long Y_batchOffset = N;
   long lenx = 1 + (N-1) * abs(incX);
   long leny = 1 + (N-1) * abs(incY);
   double *Xbatch = (double*)calloc(lenx * batchSize, sizeof(double));
   double *Ybatch = (double*)calloc(leny * batchSize, sizeof(double));
   double *Ycblasbatch = (double*)calloc(N * batchSize, sizeof(double));
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;
   double* devXbatch = hc::am_alloc(sizeof(double) * lenx * batchSize, acc, 0);
   double* devYbatch = hc::am_alloc(sizeof(double) * leny * batchSize, acc, 0);
/* Implementation type II - Inputs and Outputs are HCC double array containers with batch processing */
   for(int i = 0;i < lenx * batchSize;i++){
            Xbatch[i] = rand() % 10;
   }
   for(int i = 0;i < leny * batchSize;i++){
            Ybatch[i] =  rand() % 15;
            Ycblasbatch[i] = Ybatch[i];
   }
   accl_view.copy(Xbatch, devXbatch, lenx * batchSize * sizeof(double));
   accl_view.copy(Ybatch, devYbatch, leny * batchSize * sizeof(double));
   /* Proper call */
   status= hc.hcblas_daxpy(accl_view, N, alpha, devXbatch, incX, X_batchOffset, devYbatch, incY, Y_batchOffset, xOffset, yOffset, batchSize);
   EXPECT_EQ(status, HCBLAS_SUCCEEDS);
   accl_view.copy(devYbatch, Ybatch, leny * batchSize * sizeof(double));
   for(int i = 0; i < batchSize; i++)
          cblas_daxpy( N, alpha, Xbatch + i * N, incX, Ycblasbatch + i * N, incY );
   for(int i =0; i < leny * batchSize; i ++)
         EXPECT_EQ( Ybatch[i], Ycblasbatch[i]);
   free(Xbatch);
   free(Ybatch);
   free(Ycblasbatch);
   hc::am_free(devXbatch);
   hc::am_free(devYbatch);
}
