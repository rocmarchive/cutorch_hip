#include "hcblaslib.h"
#include <cstdlib> 
#include "gtest/gtest.h"
#include "hc_am.hpp"
#include "cblas.h"

TEST(hcblas_sger, return_correct_sger_Implementation_type_1) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
    int M = 179;
    int N = 19;
    float alpha = 1;
    long lda;
    int incX = 1;
    int incY = 1;
    long xOffset = 0;
    long yOffset = 0;
    long aOffset = 0;
    long lenx,  leny;
    hcblasStatus status;
    hcblasOrder hcOrder = ColMajor;
    lda = (hcOrder)? M : N;
    lenx =  1 + (M-1) * abs(incX);
    leny =  1 + (N-1) * abs(incY);
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;

/* Implementation type I - Inputs and Outputs are HCC device pointers */
    float *x = (float*)calloc( lenx , sizeof(float));
    float *y = (float*)calloc( leny , sizeof(float));
    float *A = (float *)calloc( lenx * leny , sizeof(float));
    float* devA = hc::am_alloc(sizeof(float) * lenx * leny, acc, 0);
    float* devX = hc::am_alloc(sizeof(float) * lenx, acc, 0);
    float* devY = hc::am_alloc(sizeof(float) * leny, acc, 0);
    for(int i = 0; i < lenx; i++) {
                x[i] = rand() % 10;
    }
    for(int i = 0; i < leny; i++) {
                y[i] = rand() % 15;
    }
    for(int i = 0; i< lenx * leny; i++) {
                A[i] = rand() % 25;
    }
    accl_view.copy(A, devA, lenx * leny * sizeof(float));
    accl_view.copy(x, devX, lenx * sizeof(float));
    accl_view.copy(y, devY, leny * sizeof(float));
    /* Proper call with column major */
    status = hc.hcblas_sger(accl_view, hcOrder, M , N , alpha, devX, xOffset, incX, devY, yOffset, incY, devA, aOffset, lda );
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    
    hcOrder = RowMajor;
    lda = (hcOrder)? M : N;
    /* Proper call with row major */
    status = hc.hcblas_sger(accl_view, hcOrder, M , N , alpha, devX, xOffset, incX, devY, yOffset, incY, devA, aOffset, lda );
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    /* alpha is 0 */
    status = hc.hcblas_sger(accl_view, hcOrder, M , N , 0, devX, xOffset, incX, devY, yOffset, incY, devA, aOffset, lda );
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    /* x, y, A are not allocated properly*/
    float *devA1 = NULL;
    float *devX1 = NULL;
    float *devY1 = NULL;
    status = hc.hcblas_sger(accl_view, hcOrder, M , N , alpha, devX1, xOffset, incX, devY, yOffset, incY, devA, aOffset, lda );
    EXPECT_EQ(status, HCBLAS_INVALID);
    status = hc.hcblas_sger(accl_view, hcOrder, M , N , alpha, devX, xOffset, incX, devY1, yOffset, incY, devA, aOffset, lda );
    EXPECT_EQ(status, HCBLAS_INVALID);
    status = hc.hcblas_sger(accl_view, hcOrder, M , N , alpha, devX, xOffset, incX, devY, yOffset, incY, devA1, aOffset, lda );
    EXPECT_EQ(status, HCBLAS_INVALID);
    /* M is 0 */
    status = hc.hcblas_sger(accl_view, hcOrder, 0 , N , alpha, devX, xOffset, incX, devY, yOffset, incY, devA, aOffset, lda );
    EXPECT_EQ(status, HCBLAS_INVALID);
    /* N is 0 */
    status = hc.hcblas_sger(accl_view, hcOrder, M , 0 , alpha, devX, xOffset, incX, devY, yOffset, incY, devA, aOffset, lda );
    EXPECT_EQ(status, HCBLAS_INVALID);
    /* incx is 0 */
    status = hc.hcblas_sger(accl_view, hcOrder, M , N , alpha, devX, xOffset, 0 , devY, yOffset, incY, devA, aOffset, lda );
    EXPECT_EQ(status, HCBLAS_INVALID);
    /* incy is 0 */
    status = hc.hcblas_sger(accl_view, hcOrder, M , N , alpha, devX, xOffset, incX, devY, yOffset, 0 , devA, aOffset, lda );
    EXPECT_EQ(status, HCBLAS_INVALID);
    free(x);
    free(y);
    free(A);
    hc::am_free(devA);
    hc::am_free(devX);
    hc::am_free(devY);
}

TEST(hcblas_sger, func_correct_sger_Implementation_type_1) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
    int M = 179;
    int N = 19;
    float alpha = 1;
    long lda;
    int incX = 1;
    int incY = 1;
    long xOffset = 0;
    long yOffset = 0;
    long aOffset = 0;
    long lenx,  leny;
    hcblasStatus status;
    hcblasOrder hcOrder = ColMajor;
    lda = (hcOrder)? M : N;
    lenx =  1 + (M-1) * abs(incX);
    leny =  1 + (N-1) * abs(incY);
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;
    float *Acblas = (float *)calloc( lenx * leny , sizeof(float));
/* Implementation type I - Inputs and Outputs are HCC device pointers */
    float *x = (float*)calloc( lenx , sizeof(float));
    float *y = (float*)calloc( leny , sizeof(float));
    float *A = (float *)calloc( lenx * leny , sizeof(float));
    float* devA = hc::am_alloc(sizeof(float) * lenx * leny, acc, 0);
    float* devX = hc::am_alloc(sizeof(float) * lenx, acc, 0);
    float* devY = hc::am_alloc(sizeof(float) * leny, acc, 0);
    for(int i = 0; i < lenx; i++) {
                x[i] = rand() % 10;
    }
    for(int i = 0; i < leny; i++) {
                y[i] = rand() % 15;
    }
    for(int i = 0; i< lenx * leny; i++) {
                A[i] = rand() % 25;
                Acblas[i] = A[i];
    }
    accl_view.copy(A, devA, lenx * leny * sizeof(float));
    accl_view.copy(x, devX, lenx * sizeof(float));
    accl_view.copy(y, devY, leny * sizeof(float));
    /* Proper call with column major */
    status = hc.hcblas_sger(accl_view, hcOrder, M , N , alpha, devX, xOffset, incX, devY, yOffset, incY, devA, aOffset, lda );
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devA, A, lenx * leny * sizeof(float));
    cblas_sger( CblasColMajor, M, N, alpha, x, incX, y, incY, Acblas, lda);
    for(int i =0; i < lenx * leny ; i++)
         EXPECT_EQ(A[i], Acblas[i]);

    hcOrder = RowMajor;
    lda = (hcOrder)? M : N;
    /* Proper call with row major */
    status = hc.hcblas_sger(accl_view, hcOrder, M , N , alpha, devX, xOffset, incX, devY, yOffset, incY, devA, aOffset, lda );
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devA, A, lenx * leny * sizeof(float));
    cblas_sger( CblasRowMajor, M, N, alpha, x, incX, y, incY, Acblas, lda);
    for(int i =0; i < lenx * leny ; i++)
         EXPECT_EQ(A[i], Acblas[i]);
    free(x);
    free(y);
    free(A);
    free(Acblas);
    hc::am_free(devA);
    hc::am_free(devX);
    hc::am_free(devY);
}

TEST(hcblas_sger, return_correct_sger_Implementation_type_2) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
    int M = 179;
    int N = 19;
    float alpha = 1;
    long lda;
    int incX = 1;
    int incY = 1;
    long xOffset = 0;
    long yOffset = 0;
    long aOffset = 0;
    long X_batchOffset = M;
    long Y_batchOffset = N;
    long A_batchOffset = M * N;
    int batchSize = 128;
    long lenx,  leny;
    hcblasStatus status;
    hcblasOrder hcOrder = ColMajor;
    lda = (hcOrder)? M : N;
    lenx =  1 + (M-1) * abs(incX);
    leny =  1 + (N-1) * abs(incY);
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;

/* Implementation type II - Inputs and Outputs are HCC device pointers with batch processing */

    float *xbatch = (float*)calloc( lenx * batchSize, sizeof(float));
    float *ybatch = (float*)calloc( leny * batchSize, sizeof(float));
    float *Abatch = (float *)calloc( lenx * leny * batchSize, sizeof(float));
    float* devXbatch = hc::am_alloc(sizeof(float) * lenx * batchSize, acc, 0);
    float* devYbatch = hc::am_alloc(sizeof(float) * leny * batchSize, acc, 0);
    float* devAbatch = hc::am_alloc(sizeof(float) * lenx * leny * batchSize, acc, 0);
    for(int i = 0; i < lenx * batchSize; i++){
                xbatch[i] = rand() % 10;
    }
    for(int i = 0; i < leny * batchSize; i++){
                ybatch[i] = rand() % 15;
    }
    for(int i = 0; i< lenx * leny * batchSize; i++){
                Abatch[i] = rand() % 25;
    }
    accl_view.copy(xbatch, devXbatch, lenx * batchSize * sizeof(float));
    accl_view.copy(ybatch, devYbatch, leny * batchSize * sizeof(float));
    accl_view.copy(Abatch, devAbatch, lenx * leny * batchSize * sizeof(float));
    /* Proper call with column major */
    status = hc.hcblas_sger(accl_view, hcOrder, M , N , alpha, devXbatch, xOffset, X_batchOffset, incX, devYbatch, yOffset, Y_batchOffset, incY, devAbatch, aOffset, A_batchOffset, lda, batchSize );
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    hcOrder = RowMajor;
    lda = (hcOrder)? M : N;
    /* Proper call with row major */
    status = hc.hcblas_sger(accl_view, hcOrder, M , N , alpha, devXbatch, xOffset, X_batchOffset, incX, devYbatch, yOffset, Y_batchOffset, incY, devAbatch, aOffset, A_batchOffset, lda, batchSize );
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    /* alpha is 0 */
    status = hc.hcblas_sger(accl_view, hcOrder, M , N , 0, devXbatch, xOffset, X_batchOffset, incX, devYbatch, yOffset, Y_batchOffset, incY, devAbatch, aOffset, A_batchOffset, lda, batchSize );
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    /* x, y, A are not allocated properly*/
    float *devA1 = NULL;
    float *devX1 = NULL;
    float *devY1 = NULL;
    status = hc.hcblas_sger(accl_view, hcOrder, M , N , alpha, devX1, xOffset, X_batchOffset, incX, devYbatch, yOffset, Y_batchOffset, incY, devAbatch, aOffset, A_batchOffset, lda, batchSize );
    EXPECT_EQ(status, HCBLAS_INVALID);
    status = hc.hcblas_sger(accl_view, hcOrder, M , N , alpha, devXbatch, xOffset, X_batchOffset, incX, devY1, yOffset, Y_batchOffset, incY, devAbatch, aOffset, A_batchOffset, lda, batchSize );
    EXPECT_EQ(status, HCBLAS_INVALID);
    status = hc.hcblas_sger(accl_view, hcOrder, M , N , alpha, devXbatch, xOffset, X_batchOffset, incX, devYbatch, yOffset, Y_batchOffset, incY, devA1, aOffset, A_batchOffset, lda, batchSize );
    EXPECT_EQ(status, HCBLAS_INVALID);
    /* M is 0 */
    status = hc.hcblas_sger(accl_view, hcOrder, 0 , N , alpha, devXbatch, xOffset, X_batchOffset, incX, devYbatch, yOffset, Y_batchOffset, incY, devAbatch, aOffset, A_batchOffset, lda, batchSize );
    EXPECT_EQ(status, HCBLAS_INVALID);
    /* N is 0 */
    status = hc.hcblas_sger(accl_view, hcOrder, M , 0 , alpha, devXbatch, xOffset, X_batchOffset, incX, devYbatch, yOffset, Y_batchOffset, incY, devAbatch, aOffset, A_batchOffset, lda, batchSize );
    EXPECT_EQ(status, HCBLAS_INVALID);
    /* incx is 0 */
    status = hc.hcblas_sger(accl_view, hcOrder, M , N , alpha, devXbatch, xOffset, X_batchOffset, 0, devYbatch, yOffset, Y_batchOffset, incY, devAbatch, aOffset, A_batchOffset, lda, batchSize );
    EXPECT_EQ(status, HCBLAS_INVALID);
    /* incy is 0 */
    status = hc.hcblas_sger(accl_view, hcOrder, M , N , alpha, devXbatch, xOffset, X_batchOffset, incX, devYbatch, yOffset, Y_batchOffset, 0, devAbatch, aOffset, A_batchOffset, lda, batchSize );
    EXPECT_EQ(status, HCBLAS_INVALID);
    free(xbatch);
    free(ybatch);
    free(Abatch);
    hc::am_free(devAbatch);
    hc::am_free(devXbatch);
    hc::am_free(devYbatch);
}

TEST(hcblas_sger, func_correct_sger_Implementation_type_2) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
    int M = 179;
    int N = 19;
    float alpha = 1;
    long lda;
    int incX = 1;
    int incY = 1;
    long xOffset = 0;
    long yOffset = 0;
    long aOffset = 0;
    long X_batchOffset = M;
    long Y_batchOffset = N;
    long A_batchOffset = M * N;
    int batchSize = 128;
    long lenx,  leny;
    hcblasStatus status;
    hcblasOrder hcOrder = ColMajor;
    lda = (hcOrder)? M : N;
    lenx =  1 + (M-1) * abs(incX);
    leny =  1 + (N-1) * abs(incY);
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;

/* Implementation type II - Inputs and Outputs are HCC device pointers with batch processing */
    float *Acblasbatch = (float *)calloc( lenx * leny * batchSize, sizeof(float));
    float *xbatch = (float*)calloc( lenx * batchSize, sizeof(float));
    float *ybatch = (float*)calloc( leny * batchSize, sizeof(float));
    float *Abatch = (float *)calloc( lenx * leny * batchSize, sizeof(float));
    float* devXbatch = hc::am_alloc(sizeof(float) * lenx * batchSize, acc, 0);
    float* devYbatch = hc::am_alloc(sizeof(float) * leny * batchSize, acc, 0);
    float* devAbatch = hc::am_alloc(sizeof(float) * lenx * leny * batchSize, acc, 0);
    for(int i = 0; i < lenx * batchSize; i++){
                xbatch[i] = rand() % 10;
    }
    for(int i = 0; i < leny * batchSize; i++){
                ybatch[i] = rand() % 15;
    }
    for(int i = 0; i< lenx * leny * batchSize; i++){
                Abatch[i] = rand() % 25;
                Acblasbatch[i] = Abatch[i];
    }
    accl_view.copy(xbatch, devXbatch, lenx * batchSize * sizeof(float));
    accl_view.copy(ybatch, devYbatch, leny * batchSize * sizeof(float));
    accl_view.copy(Abatch, devAbatch, lenx * leny * batchSize * sizeof(float));
    /* Proper call with column major */
    status = hc.hcblas_sger(accl_view, hcOrder, M , N , alpha, devXbatch, xOffset, X_batchOffset, incX, devYbatch, yOffset, Y_batchOffset, incY, devAbatch, aOffset, A_batchOffset, lda, batchSize );
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devAbatch, Abatch, lenx * leny * batchSize * sizeof(float));
    for(int i = 0; i < batchSize; i++)
        cblas_sger( CblasColMajor, M, N, alpha, xbatch + i * M, incX, ybatch + i * N, incY, Acblasbatch + i * M * N, lda);
    for(int i =0; i < lenx * leny * batchSize; i++)
        EXPECT_EQ(Abatch[i], Acblasbatch[i]);

    hcOrder = RowMajor;
    lda = (hcOrder)? M : N;
    /* Proper call with row major */
    status = hc.hcblas_sger(accl_view, hcOrder, M , N , alpha, devXbatch, xOffset, X_batchOffset, incX, devYbatch, yOffset, Y_batchOffset, incY, devAbatch, aOffset, A_batchOffset, lda, batchSize );
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devAbatch, Abatch, lenx * leny * batchSize * sizeof(float));
    for(int i = 0; i < batchSize; i++)
        cblas_sger( CblasRowMajor, M, N, alpha, xbatch + i * M, incX, ybatch + i * N, incY, Acblasbatch + i * M * N, lda);
    for(int i =0; i < lenx * leny * batchSize; i++)
        EXPECT_EQ(Abatch[i], Acblasbatch[i]);
    free(xbatch);
    free(ybatch);
    free(Abatch);
    free(Acblasbatch);
    hc::am_free(devAbatch);
    hc::am_free(devXbatch);
    hc::am_free(devYbatch);
}
