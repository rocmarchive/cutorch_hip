#include "hcblaslib.h"
#include <cstdlib>
#include "gtest/gtest.h"
#include "hc_am.hpp"
#include "cblas.h"
#include "hc_short_vector.hpp"

using namespace hc::short_vector;

TEST(hcblas_cgemm, return_correct_cgemm_Implementation_type_1) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
    int M = 189, N = 9, K = 19;
    long lda, ldb, ldc;
    int incX = 1, incY = 1;
    long aOffset = 0;
    long bOffset = 0;
    long cOffset = 0;
    hcblasOrder hcOrder;
    hcblasTranspose typeA, typeB;
    hcblasStatus status;
    accelerator_view accl_view = hc.currentAcclView;
    accelerator acc = hc.currentAccl;
    
// Implementation type I - Inputs and Outputs are HCC device pointers */

    float_2 cAlpha, cBeta;
    cAlpha.x = 1;
    cAlpha.y = 1;
    cBeta.x = 1;
    cBeta.y = 1;
    float_2 *A = (float_2*) calloc(M * K, sizeof(float_2));
    float_2 *B = (float_2*) calloc(K * N, sizeof(float_2));
    float_2 *C = (float_2*) calloc(M * N, sizeof(float_2));
    float_2* devA = hc::am_alloc(sizeof(float_2) * M * K, acc, 0);
    float_2* devB = hc::am_alloc(sizeof(float_2) * K * N, acc, 0);
    float_2* devC = hc::am_alloc(sizeof(float_2) * M * N, acc, 0);
    for(int i = 0; i < M * K; i++) {
                A[i].x = rand() % 10;
                A[i].y = rand() % 20;
    }
    for(int i = 0; i < K * N;i++) {
                B[i].x = rand() % 15;
                B[i].y = rand() % 25;
    }
    for(int i = 0; i < M * N;i++) {
                C[i].x = rand() % 18;
                C[i].y = rand() % 28;
                C[i] = rand() % 25;
    }
    accl_view.copy(A, devA, M * K * sizeof(float_2));
    accl_view.copy(B, devB, K * N * sizeof(float_2));
    accl_view.copy(C, devC, M * N * sizeof(float_2));
// NoTransA and NoTransB */           
    typeA = NoTrans;
    typeB = NoTrans;
    // Column major */
    lda = M; ldb = K ; ldc = M;
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    
    // Row Major */
    lda = K; ldb = N ; ldc = N;      
    status = hc.hcblas_cgemm(accl_view, RowMajor, typeA, typeB, M, N, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    
// NoTransA TransB */  
    typeA = NoTrans;
    typeB = Trans;
    // Column major */
    lda = M; ldb = N ; ldc = M;
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    
    // Row Major */ 
    lda = K; ldb = K ; ldc = N;
    status = hc.hcblas_cgemm(accl_view, RowMajor, typeA, typeB, M, N, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    
// TransA NoTransB */
    typeA = Trans;
    typeB = NoTrans;
    // Column major */
    lda = K; ldb = K ; ldc = M;
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    
    // Row Major */ 
    lda = M; ldb = N ; ldc = N;
    status = hc.hcblas_cgemm(accl_view, RowMajor, typeA, typeB, M, N, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);  
    
// TransA TransB */
    typeA = Trans;
    typeB = Trans;
    // Column major */
    lda = K; ldb = N ; ldc = M;
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    
    // Row Major */ 
    lda = M; ldb = K ; ldc = N;
    status = hc.hcblas_cgemm(accl_view, RowMajor, typeA, typeB, M, N, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);  
    
    typeA = NoTrans;
    typeB = NoTrans;
    lda = M; ldb = K ; ldc = M;
    hcOrder = ColMajor;
    float_2 *devA1 = NULL;
    float_2 *devB1 = NULL;
    float_2 *devC1 = NULL;
    /* A, B, C device pointers are not allocated properly */
    status = hc.hcblas_cgemm(accl_view, hcOrder, typeA, typeB, M, N, K, cAlpha, devA1, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_INVALID);
    status = hc.hcblas_cgemm(accl_view, hcOrder, typeA, typeB, M, N, K, cAlpha, devA, aOffset, lda, devB1, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_INVALID);
    status = hc.hcblas_cgemm(accl_view, hcOrder, typeA, typeB, M, N, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC1, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_INVALID);
    // M is 0 */ 
    status = hc.hcblas_cgemm(accl_view, hcOrder, typeA, typeB, 0, N, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_INVALID);
    // N is 0 */
    status = hc.hcblas_cgemm(accl_view, hcOrder, typeA, typeB, M, 0, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_INVALID);
    // K is 0 */
    status = hc.hcblas_cgemm(accl_view, hcOrder, typeA, typeB, M, N, 0, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_INVALID);
    free(A);
    free(B);
    free(C);
    hc::am_free(devA);
    hc::am_free(devB);
    hc::am_free(devC);
}


TEST(hcblas_cgemm, func_correct_cgemm_Implementation_type_1) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
    int M = 189, N = 9, K = 19;
    long lda, ldb, ldc;
    int incX = 1, incY = 1;
    long aOffset = 0;
    long bOffset = 0;
    long cOffset = 0;
    hcblasOrder hcOrder;
    hcblasTranspose typeA, typeB;
    hcblasStatus status;
    accelerator_view accl_view = hc.currentAcclView;
    accelerator acc = hc.currentAccl;
    CBLAS_TRANSPOSE Transa, Transb;
    float alpha[2], beta[2];
// Implementation type I - Inputs and Outputs are HCC device pointers */
    float_2 cAlpha, cBeta;
    cAlpha.x = 1;
    cAlpha.y = 1;
    cBeta.x = 1;
    cBeta.y = 1;
    alpha[0] = cAlpha.x;
    alpha[1] = cAlpha.y;
    beta[0] = cBeta.x;
    beta[1] = cBeta.y;
    float_2 *A = (float_2*) calloc(M * K, sizeof(float_2));
    float_2 *B = (float_2*) calloc(K * N, sizeof(float_2));
    float_2 *C = (float_2*) calloc(M * N, sizeof(float_2));
    float_2* devA = hc::am_alloc(sizeof(float_2) * M * K, acc, 0);
    float_2* devB = hc::am_alloc(sizeof(float_2) * K * N, acc, 0);
    float_2* devC = hc::am_alloc(sizeof(float_2) * M * N, acc, 0);
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
    accl_view.copy(A, devA, M * K * sizeof(float_2));
    accl_view.copy(B, devB, K * N * sizeof(float_2));
    accl_view.copy(C, devC, M * N * sizeof(float_2));

// NoTransA and NoTransB */           
    typeA = NoTrans;
    typeB = NoTrans;
    Transa = CblasNoTrans;
    Transb = CblasNoTrans;

    // Column major */
    lda = M; ldb = K ; ldc = M;
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devC, C, M * N * sizeof(float_2));
    cblas_cgemm( CblasColMajor, Transa, Transb, M, N, K, &alpha, ablas, lda, bblas, ldb, &beta, cblas, ldc );
    for(int i = 0, k = 0; ((i < M * N) && ( k < M * N * 2)) ; i++, k = k + 2) {
            EXPECT_EQ(C[i].x, cblas[k]);
            EXPECT_EQ(C[i].y, cblas[k+1]);
    }

    // Row Major */
    lda = K; ldb = N ; ldc = N;     
    status = hc.hcblas_cgemm(accl_view, RowMajor, typeA, typeB, M, N, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devC, C, M * N * sizeof(float_2));
    cblas_cgemm( CblasRowMajor, Transa, Transb, M, N, K, &alpha, ablas, lda, bblas, ldb, &beta, cblas, ldc );
    for(int i = 0,k = 0; ((i < M * N) && ( k < M * N * 2)) ; i++, k = k + 2) {
            EXPECT_EQ(C[i].x, cblas[k]);
            EXPECT_EQ(C[i].y, cblas[k+1]);
    } 
    
// NoTransA TransB */  
    typeA = NoTrans;
    typeB = Trans;
    Transa = CblasNoTrans;
    Transb = CblasTrans;

    // Column major */
    lda = M; ldb = N ; ldc = M;
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devC, C, M * N * sizeof(float_2));
    cblas_cgemm( CblasColMajor, Transa, Transb, M, N, K, &alpha, ablas, lda, bblas, ldb, &beta, cblas, ldc );
    for(int i = 0,k = 0; ((i < M * N) && ( k < M * N * 2)) ; i++, k = k + 2) {
            EXPECT_EQ(C[i].x, cblas[k]);
            EXPECT_EQ(C[i].y, cblas[k+1]);
    }
    
    /* alpha and beta are zeroes */
    /* alpha = 0*/
    cAlpha.x = 0;
    cAlpha.y = 0;
    cBeta.x = 1;
    cBeta.y = 1;
    alpha[0] = cAlpha.x;
    alpha[1] = cAlpha.y;
    beta[0] = cBeta.x;
    beta[1] = cBeta.y;
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devC, C, M * N * sizeof(float_2));
    cblas_cgemm( CblasColMajor, Transa, Transb, M, N, K, &alpha, ablas, lda, bblas, ldb, &beta, cblas, ldc );
    for(int i = 0,k = 0; ((i < M * N) && ( k < M * N * 2)) ; i++, k = k + 2) {
            EXPECT_EQ(C[i].x, cblas[k]);
            EXPECT_EQ(C[i].y, cblas[k+1]);
    }
    /* alpha = 0, beta = 0*/
    cAlpha.x = 0;
    cAlpha.y = 0;
    cBeta.x = 0;
    cBeta.y = 0;
    alpha[0] = cAlpha.x;
    alpha[1] = cAlpha.y;
    beta[0] = cBeta.x;
    beta[1] = cBeta.y;
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devC, C, M * N * sizeof(float_2));
    cblas_cgemm( CblasColMajor, Transa, Transb, M, N, K, &alpha, ablas, lda, bblas, ldb, &beta, cblas, ldc );
    for(int i = 0,k = 0; ((i < M * N) && ( k < M * N * 2)) ; i++, k = k + 2) {
            EXPECT_EQ(C[i].x, cblas[k]);
            EXPECT_EQ(C[i].y, cblas[k+1]);
    }

    // Row Major */
    cAlpha.x = 1;
    cAlpha.y = 1;
    cBeta.x = 1;
    cBeta.y = 1;
    alpha[0] = cAlpha.x;
    alpha[1] = cAlpha.y;
    beta[0] = cBeta.x;
    beta[1] = cBeta.y; 
    lda = K; ldb = K ; ldc = N;
    status = hc.hcblas_cgemm(accl_view, RowMajor, typeA, typeB, M, N, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devC, C, M * N * sizeof(float_2));
    cblas_cgemm( CblasRowMajor, Transa, Transb, M, N, K, &alpha, ablas, lda, bblas, ldb, &beta, cblas, ldc );
    for(int i = 0,k = 0; ((i < M * N) && ( k < M * N * 2)) ; i++, k = k + 2) {
            EXPECT_EQ(C[i].x, cblas[k]);
            EXPECT_EQ(C[i].y, cblas[k+1]);
    }

    /* alpha and beta are zeroes */
    /* alpha = 0*/
    cAlpha.x = 0;
    cAlpha.y = 0;
    cBeta.x = 1;
    cBeta.y = 1;
    alpha[0] = cAlpha.x;
    alpha[1] = cAlpha.y;
    beta[0] = cBeta.x;
    beta[1] = cBeta.y;
    status = hc.hcblas_cgemm(accl_view, RowMajor, typeA, typeB, M, N, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devC, C, M * N * sizeof(float_2));
    cblas_cgemm( CblasRowMajor, Transa, Transb, M, N, K, &alpha, ablas, lda, bblas, ldb, &beta, cblas, ldc );
    for(int i = 0,k = 0; ((i < M * N) && ( k < M * N * 2)) ; i++, k = k + 2) {
            EXPECT_EQ(C[i].x, cblas[k]);
            EXPECT_EQ(C[i].y, cblas[k+1]);
    }

    /* alpha = 0, beta = 0*/
    cAlpha.x = 0;
    cAlpha.y = 0;
    cBeta.x = 0;
    cBeta.y = 0;
    alpha[0] = cAlpha.x;
    alpha[1] = cAlpha.y;
    beta[0] = cBeta.x;
    beta[1] = cBeta.y;
    status = hc.hcblas_cgemm(accl_view, RowMajor, typeA, typeB, M, N, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devC, C, M * N * sizeof(float_2));
    cblas_cgemm( CblasRowMajor, Transa, Transb, M, N, K, &alpha, ablas, lda, bblas, ldb, &beta, cblas, ldc );
    for(int i = 0,k = 0; ((i < M * N) && ( k < M * N * 2)) ; i++, k = k + 2) {
            EXPECT_EQ(C[i].x, cblas[k]);
            EXPECT_EQ(C[i].y, cblas[k+1]);
    }

// TransA NoTransB */
    typeA = Trans;
    typeB = NoTrans;
    Transa = CblasTrans;
    Transb = CblasNoTrans;
    cAlpha.x = 1;
    cAlpha.y = 1;
    cBeta.x = 1;
    cBeta.y = 1;
    alpha[0] = cAlpha.x;
    alpha[1] = cAlpha.y;
    beta[0] = cBeta.x;
    beta[1] = cBeta.y;

    // Column major */
    lda = K; ldb = K ; ldc = M;
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devC, C, M * N * sizeof(float_2));
    cblas_cgemm( CblasColMajor, Transa, Transb, M, N, K, &alpha, ablas, lda, bblas, ldb, &beta, cblas, ldc );
    for(int i = 0,k = 0; ((i < M * N) && ( k < M * N * 2)) ; i++, k = k + 2) {
            EXPECT_EQ(C[i].x, cblas[k]);
            EXPECT_EQ(C[i].y, cblas[k+1]);
    }

    // Row Major */ 
    lda = M; ldb = N ; ldc = N;
    status = hc.hcblas_cgemm(accl_view, RowMajor, typeA, typeB, M, N, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devC, C, M * N * sizeof(float_2));
    cblas_cgemm( CblasRowMajor, Transa, Transb, M, N, K, &alpha, ablas, lda, bblas, ldb, &beta, cblas, ldc );
    for(int i = 0,k = 0; ((i < M * N) && ( k < M * N * 2)) ; i++, k = k + 2) {
            EXPECT_EQ(C[i].x, cblas[k]);
            EXPECT_EQ(C[i].y, cblas[k+1]);
    }

// TransA TransB */
    typeA = Trans;
    typeB = Trans;
    Transa = CblasTrans;
    Transb = CblasTrans;

    // Column major */
    lda = K; ldb = N ; ldc = M;
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devC, C, M * N * sizeof(float_2));
    cblas_cgemm( CblasColMajor, Transa, Transb, M, N, K, &alpha, ablas, lda, bblas, ldb, &beta, cblas, ldc );
    for(int i = 0,k = 0; ((i < M * N) && ( k < M * N * 2)) ; i++, k = k + 2) {
            EXPECT_EQ(C[i].x, cblas[k]);
            EXPECT_EQ(C[i].y, cblas[k+1]);
    }

    
    // Row Major */ 
    lda = M; ldb = K ; ldc = N;
    status = hc.hcblas_cgemm(accl_view, RowMajor, typeA, typeB, M, N, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devC, C, M * N * sizeof(float_2));
    cblas_cgemm( CblasRowMajor, Transa, Transb, M, N, K, &alpha, ablas, lda, bblas, ldb, &beta, cblas, ldc );
    for(int i = 0,k = 0; ((i < M * N) && ( k < M * N * 2)) ; i++, k = k + 2) {
            EXPECT_EQ(C[i].x, cblas[k]);
            EXPECT_EQ(C[i].y, cblas[k+1]);
    }

    free(A);
    free(B);
    free(C);
    free(cblas);
    free(ablas);
    free(bblas);
    hc::am_free(devA);
    hc::am_free(devB);
    hc::am_free(devC);
}


TEST(hcblas_cgemm, return_correct_cgemm_Implementation_type_2) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
    int M = 189, N = 9, K = 19;
    long lda, ldb, ldc;
    int incX = 1, incY = 1;
    long aOffset = 0;
    long bOffset = 0;
    long cOffset = 0;
    long A_batchOffset = 0;
    long B_batchOffset = 0;
    long C_batchOffset = M * N;
    int batchSize = 64;
    hcblasOrder hcOrder;
    hcblasTranspose typeA, typeB;
    hcblasStatus status;
    accelerator_view accl_view = hc.currentAcclView;
    accelerator acc = hc.currentAccl;
    float_2 cAlpha, cBeta;
    cAlpha.x = 1;
    cAlpha.y = 1;
    cBeta.x = 1;
    cBeta.y = 1;
 
   // Implementation type II - Inputs and Outputs are HCC device pointers with batch processing 
        
   float_2 *Abatch = (float_2*) calloc(M * K, sizeof(float_2));
   float_2 *Bbatch = (float_2*) calloc(K * N, sizeof(float_2));
   float_2 *Cbatch = (float_2*) calloc(M * N * batchSize, sizeof(float_2));          
   float_2* devAbatch = hc::am_alloc(sizeof(float_2) * M * K, acc, 0);
   float_2* devBbatch = hc::am_alloc(sizeof(float_2) * K * N, acc, 0);
   float_2* devCbatch = hc::am_alloc(sizeof(float_2) * M * N * batchSize, acc, 0);

   for(int i = 0; i < M * K; i++) {
             Abatch[i].x = rand() % 10;
             Abatch[i].y = rand() % 20;
   }
   for(int i = 0; i < K * N;i++) {
             Bbatch[i].x = rand() % 15;
             Bbatch[i].y = rand() % 25;
   }
   for(int i = 0; i < M * N * batchSize;i++) {
             Cbatch[i].x = rand() % 18;
             Cbatch[i].y = rand() % 28;
   } 

   accl_view.copy(Abatch, devAbatch, M * K * sizeof(float_2));
   accl_view.copy(Bbatch, devBbatch, K * N * sizeof(float_2));
   accl_view.copy(Cbatch, devCbatch, M * N * batchSize * sizeof(float_2));

   // NoTransA and NoTransB            
    typeA = NoTrans;
    typeB = NoTrans;
    // Column major 
    lda = M; ldb = K ; ldc = M;
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    
    // Row Major 
    lda = K; ldb = N ; ldc = N;   
    status = hc.hcblas_cgemm(accl_view, RowMajor, typeA, typeB, M, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);   
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);

    // NoTransA TransB   
    typeA = NoTrans;
    typeB = Trans;
    // Column major 
    lda = M; ldb = N ; ldc = M;
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    
    // Row Major  
    lda = K; ldb = K ; ldc = N;
    status = hc.hcblas_cgemm(accl_view, RowMajor, typeA, typeB, M, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);

    // TransA NoTransB 
    typeA = Trans;
    typeB = NoTrans;
    // Column major
    lda = K; ldb = K ; ldc = M;
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    
    // Row Major 
    lda = M; ldb = N ; ldc = N;
    status = hc.hcblas_cgemm(accl_view, RowMajor, typeA, typeB, M, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);  

    // TransA TransB 
    typeA = Trans;
    typeB = Trans;
    // Column major 
    lda = K; ldb = N ; ldc = M;
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    
    // Row Major 
    lda = M; ldb = K ; ldc = N;
    status = hc.hcblas_cgemm(accl_view, RowMajor, typeA, typeB, M, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);

    typeA = NoTrans;
    typeB = NoTrans;
    lda = M; ldb = K ; ldc = M;
    float_2 *devA1 = NULL;
    float_2 *devB1 = NULL;
    float_2 *devC1 = NULL;
    /* A, B, C device pointers are not allocated properly */
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, K, cAlpha, devA1, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    EXPECT_EQ(status, HCBLAS_INVALID); 
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devB1, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    EXPECT_EQ(status, HCBLAS_INVALID);
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devC1, cOffset, C_batchOffset, ldc, batchSize);
    EXPECT_EQ(status, HCBLAS_INVALID);
    // M is 0
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, 0, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    EXPECT_EQ(status, HCBLAS_INVALID);
    // N is 0
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, 0, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    EXPECT_EQ(status, HCBLAS_INVALID);
    // K is 0
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, 0, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    EXPECT_EQ(status, HCBLAS_INVALID);            
    free(Abatch);
    free(Bbatch);
    free(Cbatch);
    hc::am_free(devAbatch);
    hc::am_free(devBbatch);
    hc::am_free(devCbatch);
}   
 
 
TEST(hcblas_cgemm, func_correct_cgemm_Implementation_type_2) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
   int M = 189, N = 9, K = 19;
   long lda, ldb, ldc;
   int incX = 1, incY = 1;
   long aOffset = 0;
   long bOffset = 0;
   long cOffset = 0;
   long A_batchOffset = 0;
   long B_batchOffset = 0;
   long C_batchOffset = M * N;
   int batchSize = 64;
   hcblasOrder hcOrder;
   hcblasTranspose typeA, typeB;
   hcblasStatus status;
   accelerator_view accl_view = hc.currentAcclView;
   accelerator acc = hc.currentAccl;
   CBLAS_TRANSPOSE Transa, Transb;
   // Implementation type II - Inputs and Outputs are HCC device pointers with batch processing 
   float alpha[2], beta[2];
   float_2 cAlpha, cBeta;     
   float_2 *Abatch = (float_2*) calloc(M * K, sizeof(float_2));
   float_2 *Bbatch = (float_2*) calloc(K * N, sizeof(float_2));
   float_2 *Cbatch = (float_2*) calloc(M * N * batchSize, sizeof(float_2));
   float_2* devAbatch = hc::am_alloc(sizeof(float_2) * M * K, acc, 0);
   float_2* devBbatch = hc::am_alloc(sizeof(float_2) * K * N, acc, 0);
   float_2* devCbatch = hc::am_alloc(sizeof(float_2) * M * N * batchSize, acc, 0);
   float* abatch = (float *)malloc(sizeof(float )* M * K * 2);
   float* bbatch = (float *)malloc(sizeof(float )* K * N * 2);
   float* cbatch = (float *)malloc(sizeof(float )* M * N * 2 * batchSize);   
   int k = 0;
   for (int i = 0;i < M * K; i++) {
        Abatch[i].x = rand() % 10;
        Abatch[i].y = rand() % 20;
        abatch[k++] = Abatch[i].x;
        abatch[k++] = Abatch[i].y;
    }
    k = 0;
    for (int i = 0;i < K * N; i++) {
        Bbatch[i].x = rand() % 15;
        Bbatch[i].y = rand() % 25;
        bbatch[k++] = Bbatch[i].x;
        bbatch[k++] = Bbatch[i].y;
    }
    k = 0;
    for (int i = 0;i < M * N * batchSize; i++) {
        Cbatch[i].x = rand() % 18;
        Cbatch[i].y = rand() % 28;
        cbatch[k++] = Cbatch[i].x ;
        cbatch[k++] = Cbatch[i].y;
    }
    accl_view.copy(Abatch,devAbatch, M * K * sizeof(float_2));
    accl_view.copy(Bbatch, devBbatch, K * N * sizeof(float_2));
    accl_view.copy(Cbatch, devCbatch, M * N * batchSize * sizeof(float_2));
    cAlpha.x = 1;
    cAlpha.y = 1;
    cBeta.x = 1;
    cBeta.y = 1;
    alpha[0] = cAlpha.x;
    alpha[1] = cAlpha.y;
    beta[0] = cBeta.x;
    beta[1] = cBeta.y;

   // NoTransA and NoTransB            
    typeA = NoTrans;
    typeB = NoTrans;
    Transa = CblasNoTrans;
    Transb = CblasNoTrans;

    // Column major 
    lda = M; ldb = K ; ldc = M;
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    accl_view.copy(devCbatch, Cbatch,  M * N * batchSize * sizeof(float_2));
    for(int i = 0; i < batchSize;i++)
         cblas_cgemm( CblasColMajor, Transa, Transb, M, N, K, &alpha, abatch, lda, bbatch, ldb, &beta, cbatch + i * M * N * 2, ldc );
    for(int i = 0,k = 0; ((i < M * N * batchSize)&&( k < M * N * 2 * batchSize)); i++, k = k + 2){
         EXPECT_EQ(Cbatch[i].x, cbatch[k]);
         EXPECT_EQ(Cbatch[i].y, cbatch[k+1]);
    }

    /* alpha = 0 */
    lda = M; ldb = K ; ldc = M;
    cAlpha.x = 0;
    cAlpha.y = 0;
    cBeta.x = 1;
    cBeta.y = 1;
    alpha[0] = cAlpha.x;
    alpha[1] = cAlpha.y;
    beta[0] = cBeta.x;
    beta[1] = cBeta.y;
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    accl_view.copy(devCbatch, Cbatch,  M * N * batchSize * sizeof(float_2));
    for(int i = 0; i < batchSize;i++)
         cblas_cgemm( CblasColMajor, Transa, Transb, M, N, K, &alpha, abatch, lda, bbatch, ldb, &beta, cbatch + i * M * N * 2, ldc );
    for(int i = 0,k = 0; ((i < M * N * batchSize)&&( k < M * N * 2 * batchSize)); i++, k = k + 2){
         EXPECT_EQ(Cbatch[i].x, cbatch[k]);
         EXPECT_EQ(Cbatch[i].y, cbatch[k+1]);
    }
    /* alpha = 0, beta = 0*/
    lda = M; ldb = K ; ldc = M;
    cAlpha.x = 0;
    cAlpha.y = 0;
    cBeta.x = 0;
    cBeta.y = 0;
    alpha[0] = cAlpha.x;
    alpha[1] = cAlpha.y;
    beta[0] = cBeta.x;
    beta[1] = cBeta.y;
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    accl_view.copy(devCbatch, Cbatch,  M * N * batchSize * sizeof(float_2));
    for(int i = 0; i < batchSize;i++)
         cblas_cgemm( CblasColMajor, Transa, Transb, M, N, K, &alpha, abatch, lda, bbatch, ldb, &beta, cbatch + i * M * N * 2, ldc );
    for(int i = 0,k = 0; ((i < M * N * batchSize)&&( k < M * N * 2 * batchSize)); i++, k = k + 2){
         EXPECT_EQ(Cbatch[i].x, cbatch[k]);
         EXPECT_EQ(Cbatch[i].y, cbatch[k+1]);
    }
    
    // Row Major 
    lda = K; ldb = N ; ldc = N;     
    cAlpha.x = 1;
    cAlpha.y = 1;
    cBeta.x = 1;
    cBeta.y = 1;
    alpha[0] = cAlpha.x;
    alpha[1] = cAlpha.y;
    beta[0] = cBeta.x;
    beta[1] = cBeta.y;
    status = hc.hcblas_cgemm(accl_view, RowMajor, typeA, typeB, M, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    accl_view.copy(devCbatch, Cbatch,  M * N * batchSize * sizeof(float_2));
    for(int i = 0; i < batchSize;i++)
         cblas_cgemm( CblasRowMajor, Transa, Transb, M, N, K, &alpha, abatch, lda, bbatch, ldb, &beta, cbatch + i * M * N * 2, ldc );
    for(int i = 0,k = 0; ((i < M * N * batchSize)&&( k < M * N * 2 * batchSize)); i++, k = k + 2){
         EXPECT_EQ(Cbatch[i].x, cbatch[k]);
         EXPECT_EQ(Cbatch[i].y, cbatch[k+1]);
    }
 
    /* alpha = 0 */
    lda = K; ldb = N ; ldc = N;    
    cAlpha.x = 0;
    cAlpha.y = 0;
    cBeta.x = 1;
    cBeta.y = 1;
    alpha[0] = cAlpha.x;
    alpha[1] = cAlpha.y;
    beta[0] = cBeta.x;
    beta[1] = cBeta.y; 
    status = hc.hcblas_cgemm(accl_view, RowMajor, typeA, typeB, M, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    accl_view.copy(devCbatch, Cbatch,  M * N * batchSize * sizeof(float_2));
    for(int i = 0; i < batchSize;i++)
         cblas_cgemm( CblasRowMajor, Transa, Transb, M, N, K, &alpha, abatch, lda, bbatch, ldb, &beta, cbatch + i * M * N * 2, ldc );
    for(int i = 0,k = 0; ((i < M * N * batchSize)&&( k < M * N * 2 * batchSize)); i++, k = k + 2){
         EXPECT_EQ(Cbatch[i].x, cbatch[k]);
         EXPECT_EQ(Cbatch[i].y, cbatch[k+1]);
    } 

    /* alpha = 0, beta = 0 */
    lda = K; ldb = N ; ldc = N;     
    cAlpha.x = 0;
    cAlpha.y = 0;
    cBeta.x = 0;
    cBeta.y = 0;
    alpha[0] = cAlpha.x;
    alpha[1] = cAlpha.y;
    beta[0] = cBeta.x;
    beta[1] = cBeta.y;
    status = hc.hcblas_cgemm(accl_view, RowMajor, typeA, typeB, M, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    accl_view.copy(devCbatch, Cbatch,  M * N * batchSize * sizeof(float_2));
    for(int i = 0; i < batchSize;i++)
         cblas_cgemm( CblasRowMajor, Transa, Transb, M, N, K, &alpha, abatch, lda, bbatch, ldb, &beta, cbatch + i * M * N * 2, ldc );
    for(int i = 0,k = 0; ((i < M * N * batchSize)&&( k < M * N * 2 * batchSize)); i++, k = k + 2){
         EXPECT_EQ(Cbatch[i].x, cbatch[k]);
         EXPECT_EQ(Cbatch[i].y, cbatch[k+1]);
    } 

    // NoTransA TransB   
    typeA = NoTrans;
    typeB = Trans;
    Transa = CblasNoTrans;
    Transb = CblasTrans;

    // Column major 
    cAlpha.x = 1;
    cAlpha.y = 1;
    cBeta.x = 1;
    cBeta.y = 1;
    alpha[0] = cAlpha.x;
    alpha[1] = cAlpha.y;
    beta[0] = cBeta.x;
    beta[1] = cBeta.y;
    lda = M; ldb = N ; ldc = M;
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    accl_view.copy(devCbatch, Cbatch,  M * N * batchSize * sizeof(float_2));
    for(int i = 0; i < batchSize;i++)
         cblas_cgemm( CblasColMajor, Transa, Transb, M, N, K, &alpha, abatch, lda, bbatch, ldb, &beta, cbatch + i * M * N * 2, ldc );
    for(int i = 0,k = 0; ((i < M * N * batchSize)&&( k < M * N * 2 * batchSize)); i++, k = k + 2){
         EXPECT_EQ(Cbatch[i].x, cbatch[k]);
         EXPECT_EQ(Cbatch[i].y, cbatch[k+1]);
    }
    
    // Row Major  
    lda = K; ldb = K ; ldc = N;
    status = hc.hcblas_cgemm(accl_view, RowMajor, typeA, typeB, M, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    accl_view.copy(devCbatch, Cbatch,  M * N * batchSize * sizeof(float_2));
    for(int i = 0; i < batchSize;i++)
         cblas_cgemm( CblasRowMajor, Transa, Transb, M, N, K, &alpha, abatch, lda, bbatch, ldb, &beta, cbatch + i * M * N * 2, ldc );
    for(int i = 0,k = 0; ((i < M * N * batchSize)&&( k < M * N * 2 * batchSize)); i++, k = k + 2){
         EXPECT_EQ(Cbatch[i].x, cbatch[k]);
         EXPECT_EQ(Cbatch[i].y, cbatch[k+1]);
    }

    // TransA NoTransB 
    typeA = Trans;
    typeB = NoTrans;
    Transa = CblasTrans;
    Transb = CblasNoTrans;

    // Column major
    lda = K; ldb = K ; ldc = M;
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    accl_view.copy(devCbatch, Cbatch,  M * N * batchSize * sizeof(float_2));
    for(int i = 0; i < batchSize;i++)
         cblas_cgemm( CblasColMajor, Transa, Transb, M, N, K, &alpha, abatch, lda, bbatch, ldb, &beta, cbatch + i * M * N * 2, ldc );
    for(int i = 0,k = 0; ((i < M * N * batchSize)&&( k < M * N * 2 * batchSize)); i++, k = k + 2){
         EXPECT_EQ(Cbatch[i].x, cbatch[k]);
         EXPECT_EQ(Cbatch[i].y, cbatch[k+1]);
    }
    
    // Row Major 
    lda = M; ldb = N ; ldc = N;
    status = hc.hcblas_cgemm(accl_view, RowMajor, typeA, typeB, M, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    accl_view.copy(devCbatch, Cbatch,  M * N * batchSize * sizeof(float_2));
    for(int i = 0; i < batchSize;i++)
         cblas_cgemm( CblasRowMajor, Transa, Transb, M, N, K, &alpha, abatch, lda, bbatch, ldb, &beta, cbatch + i * M * N * 2, ldc );
    for(int i = 0,k = 0; ((i < M * N * batchSize)&&( k < M * N * 2 * batchSize)); i++, k = k + 2){
         EXPECT_EQ(Cbatch[i].x, cbatch[k]);
         EXPECT_EQ(Cbatch[i].y, cbatch[k+1]);
    }

    // TransA TransB 
    typeA = Trans;
    typeB = Trans;
    Transa = CblasTrans;
    Transb = CblasTrans;

    // Column major 
    lda = K; ldb = N ; ldc = M;
    status = hc.hcblas_cgemm(accl_view, ColMajor, typeA, typeB, M, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    accl_view.copy(devCbatch, Cbatch,  M * N * batchSize * sizeof(float_2));
    for(int i = 0; i < batchSize;i++)
         cblas_cgemm( CblasColMajor, Transa, Transb, M, N, K, &alpha, abatch, lda, bbatch, ldb, &beta, cbatch + i * M * N * 2, ldc );
    for(int i = 0,k = 0; ((i < M * N * batchSize)&&( k < M * N * 2 * batchSize)); i++, k = k + 2){
         EXPECT_EQ(Cbatch[i].x, cbatch[k]);
         EXPECT_EQ(Cbatch[i].y, cbatch[k+1]);
    }
    
    // Row Major 
    lda = M; ldb = K ; ldc = N;
    status = hc.hcblas_cgemm(accl_view, RowMajor, typeA, typeB, M, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
    accl_view.copy(devCbatch, Cbatch,  M * N * batchSize * sizeof(float_2));
    for(int i = 0; i < batchSize;i++)
         cblas_cgemm( CblasRowMajor, Transa, Transb, M, N, K, &alpha, abatch, lda, bbatch, ldb, &beta, cbatch + i * M * N * 2, ldc );
    for(int i = 0,k = 0; ((i < M * N * batchSize)&&( k < M * N * 2 * batchSize)); i++, k = k + 2){
         EXPECT_EQ(Cbatch[i].x, cbatch[k]);
         EXPECT_EQ(Cbatch[i].y, cbatch[k+1]);
    }
    free(Abatch);
    free(Bbatch);
    free(Cbatch);
    free(abatch);
    free(bbatch);
    free(cbatch);
    hc::am_free(devAbatch);
    hc::am_free(devBbatch);
    hc::am_free(devCbatch);

}

