#include "hcblaslib.h"
#include <cstdlib> 
#include "hc_short_vector.hpp"
#include <unistd.h>
#include "gtest/gtest.h"
#include "hc_am.hpp"
#include "cblas.h"

TEST(hcblas_sgemv, return_correct_sgemv_Implementation_type_1) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
    int M = 179;
    int N = 19;
    int isTransA = 1;
    int row, col;
    float alpha = 1;
    float beta = 1;
    long lda;
    int incX = 1;
    int incY = 1;
    long xOffset = 0;
    long yOffset = 0;
    long aOffset = 0;
    int batchSize = 128;
    long lenx,  leny;
    hcblasStatus status;
    hcblasTranspose typeA;
    hcblasOrder hcOrder = ColMajor;
    row = M; col = N; lda = N; typeA = Trans;
    lenx = 1 + (row - 1) * abs(incX);
    leny = 1 + (col - 1) * abs(incY);
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;

/* Implementation type I - Inputs and Outputs are HCC float array containers */
    float *x = (float*)calloc( lenx , sizeof(float));
    float *y = (float*)calloc( leny , sizeof(float));
    float *A = (float *)calloc( lenx * leny , sizeof(float));
    float* devA = hc::am_alloc(sizeof(float) * lenx * leny, acc, 0);
    float* devX = hc::am_alloc(sizeof(float) * lenx, acc, 0);
    float* devY = hc::am_alloc(sizeof(float) * leny, acc, 0);
    for(int i = 0; i < lenx; i++) {
           x[i] = rand() % 10;
    }
    for(int i = 0; i< lenx * leny; i++) {
           A[i] = rand() % 25;
    }
    for(int i = 0; i < leny; i++) {
           y[i] = rand() % 15;
    }
    accl_view.copy(A, devA, lenx * leny * sizeof(float));
    accl_view.copy(x, devX, lenx * sizeof(float));
    accl_view.copy(y, devY, leny * sizeof(float));
    /* Proper call with column major */
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devA, aOffset, lda, devX, xOffset, incX, beta, devY, yOffset, incY);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    /*Proper call with row major */
    status = hc.hcblas_sgemv(accl_view, RowMajor, typeA, M, N, alpha, devA, aOffset, lda, devX, xOffset, incX, beta, devY, yOffset, incY);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    /* x, y, A are not properly allocated */
    float *devAn = NULL;
    float *devXn = NULL;
    float *devYn = NULL;
    status =  hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devAn, aOffset, lda, devX, xOffset, incX, beta, devY, yOffset, incY);
    EXPECT_EQ(status, HCBLAS_INVALID);
    status =  hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devA, aOffset, lda, devXn, xOffset, incX, beta, devY, yOffset, incY);
    EXPECT_EQ(status, HCBLAS_INVALID);
    status =  hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devA, aOffset, lda, devX, xOffset, incX, beta, devYn, yOffset, incY);
    EXPECT_EQ(status, HCBLAS_INVALID);
    /* M is 0 */
    status =  hc.hcblas_sgemv(accl_view, hcOrder, typeA, 0, N, alpha, devA, aOffset, lda, devX, xOffset, incX, beta, devY, yOffset, incY);
    EXPECT_EQ(status, HCBLAS_INVALID);
    /* N is 0 */
    status =hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, 0, alpha, devA, aOffset, lda, devX, xOffset, incX, beta, devY, yOffset, incY);
    EXPECT_EQ(status, HCBLAS_INVALID);
    /* incx is 0 */
    status =hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devA, aOffset, lda, devX, xOffset, 0, beta, devY, yOffset, incY);
    EXPECT_EQ(status, HCBLAS_INVALID);
    /* incy is 0 */
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devA, aOffset, lda, devX, xOffset, incX, beta, devY, yOffset, 0);
    EXPECT_EQ(status, HCBLAS_INVALID);
    /* alpha and beta is 0 */
    alpha = 0; beta = 0;
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devA, aOffset, lda, devX, xOffset, incX, beta, devY, yOffset, incY);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    /* alpha is 0 and beta is 1*/
    beta = 1;
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devA, aOffset, lda, devX, xOffset, incX, beta, devY, yOffset, incY);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    
/* NoTransA */
    hcOrder = ColMajor;
    row = N; col = M; lda = M; typeA = NoTrans;
    lenx = 1 + (row - 1) * abs(incX);
    leny = 1 + (col - 1) * abs(incY);
    float *x1 = (float*)calloc( lenx , sizeof(float));
    float *y1 = (float*)calloc( leny , sizeof(float));
    float *A1 = (float *)calloc( lenx * leny , sizeof(float));
    float* devA1 = hc::am_alloc(sizeof(float) * lenx * leny, acc, 0);
    float* devX1 = hc::am_alloc(sizeof(float) * lenx, acc, 0);
    float* devY1 = hc::am_alloc(sizeof(float) * leny, acc, 0);
    for(int i = 0; i < lenx; i++) {
           x1[i] = rand() % 10;
    }
    for(int i = 0; i< lenx * leny; i++) {
           A1[i] = rand() % 25;
    }
    for(int i = 0; i < leny; i++) {
           y1[i] = rand() % 15;
    }
    accl_view.copy(A1, devA1, lenx * leny * sizeof(float));
    accl_view.copy(x1, devX1, lenx * sizeof(float));
    accl_view.copy(y1, devY1, leny * sizeof(float));
    /* Proper call with column major */
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devA1, aOffset, lda, devX1, xOffset, incX, beta, devY1, yOffset, incY);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    /*Proper call with row major */
    status = hc.hcblas_sgemv(accl_view, RowMajor, typeA, M, N, alpha, devA1, aOffset, lda, devX1, xOffset, incX, beta, devY1, yOffset, incY);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    free(x);
    free(y);
    free(A);
    hc::am_free(devA);
    hc::am_free(devX);
    hc::am_free(devY);
    free(x1);
    free(y1);
    free(A1);
    hc::am_free(devA1);
    hc::am_free(devX1);
    hc::am_free(devY1);
}

TEST(hcblas_sgemv, func_correct_sgemv_Implementation_type_1) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
    int M = 179;
    int N = 19;
    int isTransA = 1;
    int row, col;
    float alpha = 1;
    float beta = 1;
    long lda;
    int incX = 1;
    int incY = 1;
    long xOffset = 0;
    long yOffset = 0;
    long aOffset = 0;
    int batchSize = 128;
    long lenx,  leny;
    hcblasStatus status;
    hcblasTranspose typeA;
    hcblasOrder hcOrder = ColMajor;
    row = M; col = N; lda = N; typeA = Trans;
    lenx = 1 + (row - 1) * abs(incX);
    leny = 1 + (col - 1) * abs(incY);
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;
    CBLAS_TRANSPOSE transa;
    transa = CblasTrans; 
/* Implementation type I - Inputs and Outputs are HCC float array containers */
    float *x = (float*)calloc( lenx , sizeof(float));
    float *y = (float*)calloc( leny , sizeof(float));
    float *A = (float *)calloc( lenx * leny , sizeof(float));
    float *ycblas = (float *)calloc( leny , sizeof(float));
    float* devA = hc::am_alloc(sizeof(float) * lenx * leny, acc, 0);
    float* devX = hc::am_alloc(sizeof(float) * lenx, acc, 0);
    float* devY = hc::am_alloc(sizeof(float) * leny, acc, 0);
    for(int i = 0; i < lenx; i++) {
           x[i] = rand() % 10;
    }
    for(int i = 0; i< lenx * leny; i++) {
           A[i] = rand() % 25;
    }
    for(int i = 0; i < leny; i++) {
           y[i] = rand() % 15;
           ycblas[i] = y[i];
    }
    accl_view.copy(A, devA, lenx * leny * sizeof(float));
    accl_view.copy(x, devX, lenx * sizeof(float));
    accl_view.copy(y, devY, leny * sizeof(float));
    /* Proper call with column major */
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devA, aOffset, lda, devX, xOffset, incX, beta, devY, yOffset, incY);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devY, y, leny * sizeof(float));
    lda = (hcOrder)? M: N;
    cblas_sgemv( CblasColMajor, transa, M, N, alpha, A, lda , x, incX, beta, ycblas, incY );
    for(int i =0; i < leny; i++)
        EXPECT_EQ(y[i], ycblas[i]);

    /* alpha and beta is 0 */
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, 0, devA, aOffset, lda, devX, xOffset, incX, 0, devY, yOffset, incY);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devY, y, leny * sizeof(float));
    lda = (hcOrder)? M: N;
    cblas_sgemv( CblasColMajor, transa, M, N, 0, A, lda , x, incX, 0, ycblas, incY );
    for(int i =0; i < leny; i++)
        EXPECT_EQ(y[i], ycblas[i]);
    /* alpha is 0 and beta is 1*/
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, 0, devA, aOffset, lda, devX, xOffset, incX, beta, devY, yOffset, incY);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devY, y, leny * sizeof(float));
    lda = (hcOrder)? M: N;
    cblas_sgemv( CblasColMajor, transa, M, N, 0, A, lda , x, incX, beta, ycblas, incY );
    for(int i =0; i < leny; i++)
        EXPECT_EQ(y[i], ycblas[i]);

    /*Proper call with row major */
    hcOrder = RowMajor;
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devA, aOffset, lda, devX, xOffset, incX, beta, devY, yOffset, incY);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devY, y, leny * sizeof(float));
    lda = (hcOrder)? M: N;
    cblas_sgemv( CblasRowMajor, transa, M, N, alpha, A, lda , x, incX, beta, ycblas, incY );
    for(int i =0; i < leny; i++)
        EXPECT_EQ(y[i], ycblas[i]);

/* NoTransA */
    transa = CblasNoTrans;
    hcOrder = ColMajor;
    row = N; col = M; lda = M; typeA = NoTrans;
    lenx = 1 + (row - 1) * abs(incX);
    leny = 1 + (col - 1) * abs(incY);
    float *x1 = (float*)calloc( lenx , sizeof(float));
    float *y1 = (float*)calloc( leny , sizeof(float));
    float *A1 = (float *)calloc( lenx * leny , sizeof(float));
    float *ycblas1 = (float *)calloc( leny , sizeof(float));
    float* devA1 = hc::am_alloc(sizeof(float) * lenx * leny, acc, 0);
    float* devX1 = hc::am_alloc(sizeof(float) * lenx, acc, 0);
    float* devY1 = hc::am_alloc(sizeof(float) * leny, acc, 0);
    for(int i = 0; i < lenx; i++) {
           x1[i] = rand() % 10;
    }
    for(int i = 0; i< lenx * leny; i++) {
           A1[i] = rand() % 25;
    }
    for(int i = 0; i < leny; i++) {
           y1[i] = rand() % 15;
           ycblas1[i] = y1[i];
    }
    accl_view.copy(A1, devA1, lenx * leny * sizeof(float));
    accl_view.copy(x1, devX1, lenx * sizeof(float));
    accl_view.copy(y1, devY1, leny * sizeof(float));
    /* Proper call with column major */
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devA1, aOffset, lda, devX1, xOffset, incX, beta, devY1, yOffset, incY);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devY1, y1, leny * sizeof(float));
    lda = (hcOrder)? M: N;
    cblas_sgemv( CblasColMajor, transa, M, N, alpha, A1, lda , x1, incX, beta, ycblas1, incY );
    for(int i =0; i < leny; i++)
        EXPECT_EQ(y1[i], ycblas1[i]);

    /*Proper call with row major */
    hcOrder = RowMajor;
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devA1, aOffset, lda, devX1, xOffset, incX, beta, devY1, yOffset, incY);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devY1, y1, leny * sizeof(float));
    lda = (hcOrder)? M: N;
    cblas_sgemv( CblasRowMajor, transa, M, N, alpha, A1, lda , x1, incX, beta, ycblas1, incY );
    for(int i =0; i < leny; i++)
        EXPECT_EQ(y1[i], ycblas1[i]);
    free(x);
    free(y);
    free(A);
    free(ycblas);
    free(ycblas1);
    hc::am_free(devA);
    hc::am_free(devX);
    hc::am_free(devY);
    free(x1);
    free(y1);
    free(A1);
    hc::am_free(devA1);
    hc::am_free(devX1);
    hc::am_free(devY1);
}

 
TEST(hcblas_sgemv, return_correct_sgemv_Implementation_type_2) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
    int M = 179;
    int N = 19;
    int isTransA = 1;
    int row, col;
    float alpha = 1;
    float beta = 1;
    long lda;
    int incX = 1;
    int incY = 1;
    long xOffset = 0;
    long yOffset = 0;
    long aOffset = 0;
    int batchSize = 32;
    long lenx,  leny;
    hcblasStatus status;
    hcblasTranspose typeA;
    hcblasOrder hcOrder = ColMajor;
    row = M; col = N; lda = N; typeA = Trans;
    lenx = 1 + (row - 1) * abs(incX);
    leny = 1 + (col - 1) * abs(incY);
    long X_batchOffset = row;
    long Y_batchOffset = col;
    long A_batchOffset = row * col;
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;

/* Implementation type II - Inputs and Outputs are HCC device pointers with batch processing */

    float *xbatch = (float*)calloc( lenx * batchSize, sizeof(float));
    float *ybatch = (float*)calloc( leny * batchSize, sizeof(float));
    float *Abatch = (float *)calloc( lenx * leny * batchSize, sizeof(float));
    float* devXbatch = hc::am_alloc(sizeof(float) * lenx * batchSize, acc, 0);
    float* devYbatch = hc::am_alloc(sizeof(float) * leny * batchSize, acc, 0);
    float* devAbatch = hc::am_alloc(sizeof(float) * lenx * leny * batchSize, acc, 0);
    for(int i = 0; i < lenx * batchSize; i++) {
            xbatch[i] = rand() % 10;
    }
    for(int i = 0; i< lenx * leny * batchSize; i++) {
            Abatch[i] = rand() % 25;
    }
    for(int i = 0; i < leny * batchSize; i++) {
            ybatch[i] = rand() % 15;
    }
    accl_view.copy(xbatch, devXbatch, lenx * batchSize * sizeof(float));
    accl_view.copy(ybatch, devYbatch, leny * batchSize * sizeof(float));
    accl_view.copy(Abatch, devAbatch, lenx * leny * batchSize * sizeof(float));
    /* Proper call with column major */
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devAbatch, aOffset, A_batchOffset, lda, devXbatch, xOffset, X_batchOffset, incX, beta, devYbatch, yOffset, Y_batchOffset, incY, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    /*Proper call with row major */
    status = hc.hcblas_sgemv(accl_view, RowMajor, typeA, M, N, alpha, devAbatch, aOffset, A_batchOffset, lda, devXbatch, xOffset, X_batchOffset, incX, beta, devYbatch, yOffset, Y_batchOffset, incY, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    /* x, y, A are not properly allocated */
    float *devAn = NULL;
    float *devXn = NULL;
    float *devYn = NULL;
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devAn, aOffset, A_batchOffset, lda, devXbatch, xOffset, X_batchOffset, incX, beta, devYbatch, yOffset, Y_batchOffset, incY, batchSize);
    EXPECT_EQ(status, HCBLAS_INVALID);
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devAbatch, aOffset, A_batchOffset, lda, devXn, xOffset, X_batchOffset, incX, beta, devYbatch, yOffset, Y_batchOffset, incY, batchSize);
    EXPECT_EQ(status, HCBLAS_INVALID);
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devAbatch, aOffset, A_batchOffset, lda, devXbatch, xOffset, X_batchOffset, incX, beta, devYn, yOffset, Y_batchOffset, incY, batchSize);
    EXPECT_EQ(status, HCBLAS_INVALID);
    /* M is 0 */
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, 0, N, alpha, devAbatch, aOffset, A_batchOffset, lda, devXbatch, xOffset, X_batchOffset, incX, beta, devYbatch, yOffset, Y_batchOffset, incY, batchSize);
    EXPECT_EQ(status, HCBLAS_INVALID);
    /* N is 0 */
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, 0, alpha, devAbatch, aOffset, A_batchOffset, lda, devXbatch, xOffset, X_batchOffset, incX, beta, devYbatch, yOffset, Y_batchOffset, incY, batchSize);
    EXPECT_EQ(status, HCBLAS_INVALID);
    /* incx is 0 */
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devAbatch, aOffset, A_batchOffset, lda, devXbatch, xOffset, X_batchOffset, 0, beta, devYbatch, yOffset, Y_batchOffset, incY, batchSize);
    EXPECT_EQ(status, HCBLAS_INVALID);
    /* incy is 0 */
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devAbatch, aOffset, A_batchOffset, lda, devXbatch, xOffset, X_batchOffset, incX, beta, devYbatch, yOffset, Y_batchOffset, 0, batchSize);
    EXPECT_EQ(status, HCBLAS_INVALID);
    /* alpha and beta is 0 */
    alpha = 0; beta = 0;
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devAbatch, aOffset, A_batchOffset, lda, devXbatch, xOffset, X_batchOffset, incX, beta, devYbatch, yOffset, Y_batchOffset, incY, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    /* alpha is 0 and beta is 1*/
    beta = 1;
    status =  hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devAbatch, aOffset, A_batchOffset, lda, devXbatch, xOffset, X_batchOffset, incX, beta, devYbatch, yOffset, Y_batchOffset, incY, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);  
    
/* NoTransA */
    hcOrder = ColMajor;
    row = N; col = M; lda = M; typeA = NoTrans;
    lenx = 1 + (row - 1) * abs(incX);
    leny = 1 + (col - 1) * abs(incY);
    float *xbatch1 = (float*)calloc( lenx * batchSize, sizeof(float));
    float *ybatch1 = (float*)calloc( leny * batchSize, sizeof(float));
    float *Abatch1 = (float *)calloc( lenx * leny * batchSize, sizeof(float));
    float* devXbatch1 = hc::am_alloc(sizeof(float) * lenx * batchSize, acc, 0);
    float* devYbatch1 = hc::am_alloc(sizeof(float) * leny * batchSize, acc, 0);
    float* devAbatch1 = hc::am_alloc(sizeof(float) * lenx * leny * batchSize, acc, 0);
    X_batchOffset = row;
    Y_batchOffset = col;
    A_batchOffset = row * col;
    for(int i = 0; i < lenx * batchSize; i++) {
            xbatch1[i] = rand() % 10;
    }
    for(int i = 0; i< lenx * leny * batchSize; i++) {
            Abatch1[i] = rand() % 25;
    }
    for(int i = 0; i < leny * batchSize; i++) {
            ybatch1[i] = rand() % 15;
    }
    accl_view.copy(xbatch1, devXbatch1, lenx * batchSize * sizeof(float));
    accl_view.copy(ybatch1, devYbatch1, leny * batchSize * sizeof(float));
    accl_view.copy(Abatch1, devAbatch1, lenx * leny * batchSize * sizeof(float));

   /* Proper call with column major */
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devAbatch1, aOffset, A_batchOffset, lda, devXbatch1, xOffset, X_batchOffset, incX, beta, devYbatch1, yOffset, Y_batchOffset, incY, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    /*Proper call with row major */
    status = hc.hcblas_sgemv(accl_view, RowMajor, typeA, M, N, alpha, devAbatch1, aOffset, A_batchOffset, lda, devXbatch1, xOffset, X_batchOffset, incX, beta, devYbatch1, yOffset, Y_batchOffset, incY, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    free(xbatch);
    free(ybatch);
    free(Abatch);
    hc::am_free(devAbatch);
    hc::am_free(devXbatch);
    hc::am_free(devYbatch);
    free(xbatch1);
    free(ybatch1);
    free(Abatch1);
    hc::am_free(devAbatch1);
    hc::am_free(devXbatch1);
    hc::am_free(devYbatch1);
}

TEST(hcblas_sgemv, func_correct_sgemv_Implementation_type_2) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
    int M = 179;
    int N = 19;
    int isTransA = 1;
    int row, col;
    float alpha = 1;
    float beta = 1;
    long lda;
    int incX = 1;
    int incY = 1;
    long xOffset = 0;
    long yOffset = 0;
    long aOffset = 0;
    int batchSize = 32;
    long lenx,  leny;
    hcblasStatus status;
    hcblasTranspose typeA;
    hcblasOrder hcOrder = ColMajor;
    row = M; col = N; lda = N; typeA = Trans;
    lenx = 1 + (row - 1) * abs(incX);
    leny = 1 + (col - 1) * abs(incY);
    long X_batchOffset = row;
    long Y_batchOffset = col;
    long A_batchOffset = row * col;
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;
    CBLAS_TRANSPOSE transa;
    transa = CblasTrans;
/* Implementation type II - Inputs and Outputs are HCC device pointers with batch processing */
    float *ycblasbatch = (float *)calloc( leny * batchSize, sizeof(float));
    float *xbatch = (float*)calloc( lenx * batchSize, sizeof(float));
    float *ybatch = (float*)calloc( leny * batchSize, sizeof(float));
    float *Abatch = (float *)calloc( lenx * leny * batchSize, sizeof(float));
    float* devXbatch = hc::am_alloc(sizeof(float) * lenx * batchSize, acc, 0);
    float* devYbatch = hc::am_alloc(sizeof(float) * leny * batchSize, acc, 0);
    float* devAbatch = hc::am_alloc(sizeof(float) * lenx * leny * batchSize, acc, 0);
    for(int i = 0; i < lenx * batchSize; i++) {
            xbatch[i] = rand() % 10;
    }
    for(int i = 0; i< lenx * leny * batchSize; i++) {
            Abatch[i] = rand() % 25;
    }
    for(int i = 0; i < leny * batchSize; i++) {
            ybatch[i] = rand() % 15;
            ycblasbatch[i] = ybatch[i];
    }
    accl_view.copy(xbatch, devXbatch, lenx * batchSize * sizeof(float));
    accl_view.copy(ybatch, devYbatch, leny * batchSize * sizeof(float));
    accl_view.copy(Abatch, devAbatch, lenx * leny * batchSize * sizeof(float));
    /* Proper call with column major */
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devAbatch, aOffset, A_batchOffset, lda, devXbatch, xOffset, X_batchOffset, incX, beta, devYbatch, yOffset, Y_batchOffset, incY, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devYbatch, ybatch, leny * batchSize * sizeof(float));
    lda = (hcOrder)? M : N;
    for(int i =0 ; i < batchSize; i++)
         cblas_sgemv( CblasColMajor, transa, M, N, alpha, Abatch + i * M * N, lda , xbatch + i * row, incX, beta, ycblasbatch + i * col, incY );
    for(int i =0; i < leny * batchSize; i++)
         EXPECT_EQ(ybatch[i], ycblasbatch[i]);

    /* alpha and beta is 0 */
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, 0, devAbatch, aOffset, A_batchOffset, lda, devXbatch, xOffset, X_batchOffset, incX, 0, devYbatch, yOffset, Y_batchOffset, incY, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devYbatch, ybatch, leny * batchSize * sizeof(float));
    lda = (hcOrder)? M : N;
    for(int i =0 ; i < batchSize; i++)
         cblas_sgemv( CblasColMajor, transa, M, N, 0, Abatch + i * M * N, lda , xbatch + i * row, incX, 0, ycblasbatch + i * col, incY );
    for(int i =0; i < leny * batchSize; i++)
         EXPECT_EQ(ybatch[i], ycblasbatch[i]);

    /* alpha is 0 and beta is 1*/
    status =  hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, 0, devAbatch, aOffset, A_batchOffset, lda, devXbatch, xOffset, X_batchOffset, incX, beta, devYbatch, yOffset, Y_batchOffset, incY, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devYbatch, ybatch, leny * batchSize * sizeof(float));
    lda = (hcOrder)? M : N;
    for(int i =0 ; i < batchSize; i++)
         cblas_sgemv( CblasColMajor, transa, M, N, 0, Abatch + i * M * N, lda , xbatch + i * row, incX, beta, ycblasbatch + i * col, incY );
    for(int i =0; i < leny * batchSize; i++)
         EXPECT_EQ(ybatch[i], ycblasbatch[i]);

    /*Proper call with row major */
    hcOrder = RowMajor;
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devAbatch, aOffset, A_batchOffset, lda, devXbatch, xOffset, X_batchOffset, incX, beta, devYbatch, yOffset, Y_batchOffset, incY, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devYbatch, ybatch, leny * batchSize * sizeof(float));
    lda = (hcOrder)? M : N;
    for(int i =0 ; i < batchSize; i++)
         cblas_sgemv( CblasRowMajor, transa, M, N, alpha, Abatch + i * M * N, lda , xbatch + i * row, incX, beta, ycblasbatch + i * col, incY );
    for(int i =0; i < leny * batchSize; i++)
         EXPECT_EQ(ybatch[i], ycblasbatch[i]);

/* NoTransA */
    transa = CblasNoTrans;
    hcOrder = ColMajor;
    row = N; col = M; lda = M; typeA = NoTrans;
    lenx = 1 + (row - 1) * abs(incX);
    leny = 1 + (col - 1) * abs(incY);
    float *ycblasbatch1 = (float *)calloc( leny * batchSize, sizeof(float));
    float *xbatch1 = (float*)calloc( lenx * batchSize, sizeof(float));
    float *ybatch1 = (float*)calloc( leny * batchSize, sizeof(float));
    float *Abatch1 = (float *)calloc( lenx * leny * batchSize, sizeof(float));
    float* devXbatch1 = hc::am_alloc(sizeof(float) * lenx * batchSize, acc, 0);
    float* devYbatch1 = hc::am_alloc(sizeof(float) * leny * batchSize, acc, 0);
    float* devAbatch1 = hc::am_alloc(sizeof(float) * lenx * leny * batchSize, acc, 0);
    X_batchOffset = row;
    Y_batchOffset = col;
    A_batchOffset = row * col;
    for(int i = 0; i < lenx * batchSize; i++) {
            xbatch1[i] = rand() % 10;
    }
    for(int i = 0; i< lenx * leny * batchSize; i++) {
            Abatch1[i] = rand() % 25;
    }
    for(int i = 0; i < leny * batchSize; i++) {
            ybatch1[i] = rand() % 15;
            ycblasbatch1[i] = ybatch1[i];
    }
    accl_view.copy(xbatch1, devXbatch1, lenx * batchSize * sizeof(float));
    accl_view.copy(ybatch1, devYbatch1, leny * batchSize * sizeof(float));
    accl_view.copy(Abatch1, devAbatch1, lenx * leny * batchSize * sizeof(float));

   /* Proper call with column major */
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devAbatch1, aOffset, A_batchOffset, lda, devXbatch1, xOffset, X_batchOffset, incX, beta, devYbatch1, yOffset, Y_batchOffset, incY, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devYbatch1, ybatch1, leny * batchSize * sizeof(float));
    lda = (hcOrder)? M : N;
    for(int i =0 ; i < batchSize; i++)
         cblas_sgemv( CblasColMajor, transa, M, N, alpha, Abatch1 + i * M * N, lda , xbatch1 + i * row, incX, beta, ycblasbatch1 + i * col, incY );
    for(int i =0; i < leny * batchSize; i++)
         EXPECT_EQ(ybatch1[i], ycblasbatch1[i]);

    /*Proper call with row major */
    hcOrder = RowMajor;
    status = hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devAbatch1, aOffset, A_batchOffset, lda, devXbatch1, xOffset, X_batchOffset, incX, beta, devYbatch1, yOffset, Y_batchOffset, incY, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devYbatch1, ybatch1, leny * batchSize * sizeof(float));
    lda = (hcOrder)? M : N;
    for(int i =0 ; i < batchSize; i++)
         cblas_sgemv( CblasRowMajor, transa, M, N, alpha, Abatch1 + i * M * N, lda , xbatch1 + i * row, incX, beta, ycblasbatch1 + i * col, incY );
    for(int i =0; i < leny * batchSize; i++)
         EXPECT_EQ(ybatch1[i], ycblasbatch1[i]);
    free(xbatch);
    free(ybatch);
    free(Abatch);
    free(ycblasbatch);
    free(ycblasbatch1);
    hc::am_free(devAbatch);
    hc::am_free(devXbatch);
    hc::am_free(devYbatch);
    free(xbatch1);
    free(ybatch1);
    free(Abatch1);
    hc::am_free(devAbatch1);
    hc::am_free(devXbatch1);
    hc::am_free(devYbatch1);
}

