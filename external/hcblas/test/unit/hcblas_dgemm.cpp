#include "hcblaslib.h"
#include <cstdlib>
#include "gtest/gtest.h"
#include "hc_am.hpp"
#include "cblas.h"

TEST(hcblas_dgemm, return_correct_dgemm_Implementation_type_1) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
    int M = 189, N = 9, K = 19;
    double alpha = 1, beta = 1;
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

    double *A = (double*) calloc(M * K, sizeof(double));
    double *B = (double*) calloc(K * N, sizeof(double));
    double *C = (double*) calloc(M * N, sizeof(double));
    double* devA = hc::am_alloc(sizeof(double) * M * K, acc, 0);
    double* devB = hc::am_alloc(sizeof(double) * K * N, acc, 0);
    double* devC = hc::am_alloc(sizeof(double) * M * N, acc, 0);
    for(int i = 0; i < M * K; i++) {
                A[i] = rand()%100;
    }
    for(int i = 0; i < K * N;i++) {
                B[i] = rand() % 15;
    }
    for(int i = 0; i < M * N;i++) {
                C[i] = rand() % 25;
    }
    accl_view.copy(A, devA, M * K * sizeof(double));
    accl_view.copy(B, devB, K * N * sizeof(double));
    accl_view.copy(C, devC, M * N * sizeof(double));
// NoTransA and NoTransB */           
    typeA = NoTrans;
    typeB = NoTrans;
    // Column major */
    lda = M; ldb = K ; ldc = M;
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    
    // Row Major */
    lda = K; ldb = N ; ldc = N;      
    status = hc.hcblas_dgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    
// NoTransA TransB */  
    typeA = NoTrans;
    typeB = Trans;
    // Column major */
    lda = M; ldb = N ; ldc = M;
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    
    // Row Major */ 
    lda = K; ldb = K ; ldc = N;
    status = hc.hcblas_dgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    
// TransA NoTransB */
    typeA = Trans;
    typeB = NoTrans;
    // Column major */
    lda = K; ldb = K ; ldc = M;
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    
    // Row Major */ 
    lda = M; ldb = N ; ldc = N;
    status = hc.hcblas_dgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);  
    
// TransA TransB */
    typeA = Trans;
    typeB = Trans;
    // Column major */
    lda = K; ldb = N ; ldc = M;
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    
    // Row Major */ 
    lda = M; ldb = K ; ldc = N;
    status = hc.hcblas_dgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset); 
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);  
    
    typeA = NoTrans;
    typeB = NoTrans;
    lda = M; ldb = K ; ldc = M;
    hcOrder = ColMajor;
    double *devA1 = NULL;
    double *devB1 = NULL;
    double *devC1 = NULL;
    /* A, B, C device pointers are not allocated properly */
    status = hc.hcblas_dgemm(accl_view, hcOrder, typeA, typeB, M, N, K, alpha, devA1, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_INVALID);
    status = hc.hcblas_dgemm(accl_view, hcOrder, typeA, typeB, M, N, K, alpha, devA, lda, devB1, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_INVALID);
    status = hc.hcblas_dgemm(accl_view, hcOrder, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC1, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_INVALID);
    // M is 0 */ 
    status = hc.hcblas_dgemm(accl_view, hcOrder, typeA, typeB, 0, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_INVALID);
    // N is 0 */
    status = hc.hcblas_dgemm(accl_view, hcOrder, typeA, typeB, M, 0, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_INVALID);
    // K is 0 */
    status = hc.hcblas_dgemm(accl_view, hcOrder, typeA, typeB, M, N, 0, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_INVALID);
    free(A);
    free(B);
    free(C);
    hc::am_free(devA);
    hc::am_free(devB);
    hc::am_free(devC);
}

TEST(hcblas_dgemm, func_correct_dgemm_Implementation_type_1) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
    int M = 189, N = 9, K = 19;
    double alpha = 1, beta = 1;
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
// Implementation type I - Inputs and Outputs are HCC device pointers */

    double *A = (double*) calloc(M * K, sizeof(double));
    double *B = (double*) calloc(K * N, sizeof(double));
    double *C = (double*) calloc(M * N, sizeof(double));
    double *C_hcblas = (double*) calloc(M * N, sizeof(double));
    double *C_cblas = (double*) calloc(M * N, sizeof(double));
    double* devA = hc::am_alloc(sizeof(double) * M * K, acc, 0);
    double* devB = hc::am_alloc(sizeof(double) * K * N, acc, 0);
    double* devC = hc::am_alloc(sizeof(double) * M * N, acc, 0);
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
    accl_view.copy(A, devA, M * K * sizeof(double));
    accl_view.copy(B, devB, K * N * sizeof(double));
    accl_view.copy(C, devC, M * N * sizeof(double));
// NoTransA and NoTransB */           
    typeA = NoTrans;
    typeB = NoTrans;
    Transa = CblasNoTrans;
    Transb = CblasNoTrans;

    // Column major */
    lda = M; ldb = K ; ldc = M;
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devC, C_hcblas,  M * N * sizeof(double));
    cblas_dgemm( CblasColMajor, Transa, Transb, M, N, K, alpha, A, lda, B, ldb, beta, C_cblas, ldc);
    for(int i = 0 ; i < M * N ; i++)  
      EXPECT_EQ(C_hcblas[i], C_cblas[i]);

    // Row Major */
    lda = K; ldb = N ; ldc = N;      
    status = hc.hcblas_dgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devC, C_hcblas,  M * N * sizeof(double));
    cblas_dgemm( CblasRowMajor, Transa, Transb, M, N, K, alpha, A, lda, B, ldb, beta, C_cblas, ldc);
    for(int i = 0 ; i < M * N ; i++)  
      EXPECT_EQ(C_hcblas[i], C_cblas[i]);
    
// NoTransA TransB */  
    typeA = NoTrans;
    typeB = Trans;
    Transa = CblasNoTrans;
    Transb = CblasTrans;

    // Column major */
    lda = M; ldb = N ; ldc = M;
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devC, C_hcblas, M * N * sizeof(double));
    cblas_dgemm( CblasColMajor, Transa, Transb, M, N, K, alpha, A, lda, B, ldb, beta, C_cblas, ldc);
    for(int i = 0 ; i < M * N ; i++)  
      EXPECT_EQ(C_hcblas[i], C_cblas[i]);
    
    /* alpha and beta are zeroes */
    /* alpha = 0*/
    lda = M; ldb = N ; ldc = M;
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, K, 0, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devC, C_hcblas, M * N * sizeof(double));
    cblas_dgemm( CblasColMajor, Transa, Transb, M, N, K, 0, A, lda, B, ldb, beta, C_cblas, ldc);
    for(int i = 0 ; i < M * N ; i++)  
      EXPECT_EQ(C_hcblas[i], C_cblas[i]);
    /* alpha = 0, beta = 0*/
    lda = M; ldb = N ; ldc = M;
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, K, 0, devA, lda, devB, ldb, 0, devC, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devC, C_hcblas, M * N * sizeof(double));
    cblas_dgemm( CblasColMajor, Transa, Transb, M, N, K, 0, A, lda, B, ldb, 0, C_cblas, ldc);
    for(int i = 0 ; i < M * N ; i++)  
      EXPECT_EQ(C_hcblas[i], C_cblas[i]);

    // Row Major */ 
    lda = K; ldb = K ; ldc = N;
    status = hc.hcblas_dgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devC, C_hcblas, M * N * sizeof(double));
    cblas_dgemm(CblasRowMajor, Transa, Transb, M, N, K, alpha, A, lda, B, ldb, beta, C_cblas, ldc);
    for(int i = 0 ; i < M * N ; i++)  
      EXPECT_EQ(C_hcblas[i], C_cblas[i]);

    /* alpha and beta are zeroes */
    /* alpha = 0*/
    lda = K; ldb = K ; ldc = N;
    status = hc.hcblas_dgemm(accl_view, RowMajor, typeA, typeB, M, N, K, 0, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devC, C_hcblas, M * N * sizeof(double));
    cblas_dgemm(CblasRowMajor, Transa, Transb, M, N, K, 0, A, lda, B, ldb, beta, C_cblas, ldc);
    for(int i = 0 ; i < M * N ; i++)  
      EXPECT_EQ(C_hcblas[i], C_cblas[i]);
    /* alpha = 0, beta = 0*/
    lda = K; ldb = K ; ldc = N;
    status = hc.hcblas_dgemm(accl_view, RowMajor, typeA, typeB, M, N, K, 0, devA, lda, devB, ldb, 0, devC, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devC, C_hcblas, M * N * sizeof(double));
    cblas_dgemm(CblasRowMajor, Transa, Transb, M, N, K, 0, A, lda, B, ldb, 0, C_cblas, ldc);
    for(int i = 0 ; i < M * N ; i++)  
      EXPECT_EQ(C_hcblas[i], C_cblas[i]);

// TransA NoTransB */
    typeA = Trans;
    typeB = NoTrans;
    Transa = CblasTrans;
    Transb = CblasNoTrans;

    // Column major */
    lda = K; ldb = K ; ldc = M;
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devC, C_hcblas, M * N * sizeof(double));
    cblas_dgemm( CblasColMajor, Transa, Transb, M, N, K, alpha, A, lda, B, ldb, beta, C_cblas, ldc);
    for(int i = 0 ; i < M * N ; i++)  
      EXPECT_EQ(C_hcblas[i], C_cblas[i]);

    // Row Major */ 
    lda = M; ldb = N ; ldc = N;
    status = hc.hcblas_dgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);  
    accl_view.copy(devC, C_hcblas, M * N * sizeof(double));
    cblas_dgemm( CblasRowMajor, Transa, Transb, M, N, K, alpha, A, lda, B, ldb, beta, C_cblas, ldc);
    for(int i = 0 ; i < M * N ; i++)  
      EXPECT_EQ(C_hcblas[i], C_cblas[i]);    

// TransA TransB */
    typeA = Trans;
    typeB = Trans;
    Transa = CblasTrans;
    Transb = CblasTrans;

    // Column major */
    lda = K; ldb = N ; ldc = M;
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devC, C_hcblas, M * N * sizeof(double));
    cblas_dgemm( CblasColMajor, Transa, Transb, M, N, K, alpha, A, lda, B, ldb, beta, C_cblas, ldc);
    for(int i = 0 ; i < M * N ; i++)  
      EXPECT_EQ(C_hcblas[i], C_cblas[i]);
    
    // Row Major */ 
    lda = M; ldb = K ; ldc = N;
    status = hc.hcblas_dgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset); 
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);  
    accl_view.copy(devC, C_hcblas, M * N * sizeof(double));
    cblas_dgemm( CblasRowMajor, Transa, Transb, M, N, K, alpha, A, lda, B, ldb, beta, C_cblas, ldc);
    for(int i = 0 ; i < M * N ; i++)  
      EXPECT_EQ(C_hcblas[i], C_cblas[i]);
    free(A);
    free(B);
    free(C);
    free(C_cblas);
    free(C_hcblas);
    hc::am_free(devA);
    hc::am_free(devB);
    hc::am_free(devC);
}

TEST(hcblas_dgemm, return_correct_dgemm_Implementation_type_2) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
    int M = 189, N = 9, K = 19;
    double alpha = 1, beta = 1;
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
    
   // Implementation type II - Inputs and Outputs are HCC double array containers with batch processing 
        
   double *Abatch = (double*) calloc(M * K, sizeof(double));
   double *Bbatch = (double*) calloc(K * N, sizeof(double));
   double *Cbatch = (double*) calloc(M * N * batchSize, sizeof(double));          
   double* devAbatch = hc::am_alloc(sizeof(double) * M * K, acc, 0);
   double* devBbatch = hc::am_alloc(sizeof(double) * K * N, acc, 0);
   double* devCbatch = hc::am_alloc(sizeof(double) * M * N * batchSize, acc, 0);

   for(int i = 0; i < M * K; i++) {
                Abatch[i] = rand()%100;
   }
   for(int i = 0; i < K * N;i++) {
                Bbatch[i] = rand() % 15;
   }
   for(int i = 0; i < M * N * batchSize;i++) {
                Cbatch[i] = rand() % 25;
   } 

   accl_view.copy(Abatch, devAbatch, M * K * sizeof(double));
   accl_view.copy(Bbatch, devBbatch, K * N * sizeof(double));
   accl_view.copy(Cbatch, devCbatch, M * N * batchSize * sizeof(double));

   // NoTransA and NoTransB            
    typeA = NoTrans;
    typeB = NoTrans;
    // Column major 
    lda = M; ldb = K ; ldc = M;
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    
    // Row Major 
    lda = K; ldb = N ; ldc = N;      
    status = hc.hcblas_dgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);

    // NoTransA TransB   
    typeA = NoTrans;
    typeB = Trans;
    // Column major 
    lda = M; ldb = N ; ldc = M;
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    
    // Row Major  
    lda = K; ldb = K ; ldc = N;
    status = hc.hcblas_dgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);

    // TransA NoTransB 
    typeA = Trans;
    typeB = NoTrans;
    // Column major
    lda = K; ldb = K ; ldc = M;
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    
    // Row Major 
    lda = M; ldb = N ; ldc = N;
    status = hc.hcblas_dgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);  

    // TransA TransB 
    typeA = Trans;
    typeB = Trans;
    // Column major 
    lda = K; ldb = N ; ldc = M;
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    
    // Row Major 
    lda = M; ldb = K ; ldc = N;
    status = hc.hcblas_dgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);

    typeA = NoTrans;
    typeB = NoTrans;
    lda = M; ldb = K ; ldc = M;
    double *devA1 = NULL;
    double *devB1 = NULL;
    double *devC1 = NULL;
    /* A, B, C device pointers are not allocated properly */
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devA1, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_INVALID); 
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devB1, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_INVALID);
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devC1, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_INVALID);
    // M is 0
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, 0, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_INVALID);
    // N is 0
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, 0, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_INVALID);
    // K is 0
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, 0, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_INVALID);            
    free(Abatch);
    free(Bbatch);
    free(Cbatch);
    hc::am_free(devAbatch);
    hc::am_free(devBbatch);
    hc::am_free(devCbatch);
}   
  
TEST(hcblas_dgemm, func_correct_dgemm_Implementation_type_2) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
    int M = 189, N = 9, K = 19;
    double alpha = 1, beta = 1;
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
   // Implementation type II - Inputs and Outputs are HCC double array containers with batch processing 
        
   double *Abatch = (double*) calloc(M * K, sizeof(double));
   double *Bbatch = (double*) calloc(K * N, sizeof(double));
   double *Cbatch = (double*) calloc(M * N * batchSize, sizeof(double));          
   double *CCblasbatch = (double*) calloc(M * N * batchSize, sizeof(double));
   double *Chcblasbatch = (double*) calloc(M * N * batchSize, sizeof(double));
   double* devAbatch = hc::am_alloc(sizeof(double) * M * K, acc, 0);
   double* devBbatch = hc::am_alloc(sizeof(double) * K * N, acc, 0);
   double* devCbatch = hc::am_alloc(sizeof(double) * M * N * batchSize, acc, 0);

   for(int i = 0; i < M * K; i++) {
                Abatch[i] = rand()%100;
   }
   for(int i = 0; i < K * N;i++) {
                Bbatch[i] = rand() % 15;
   }
   for(int i = 0; i < M * N * batchSize;i++) {
                Cbatch[i] = rand() % 25;
                CCblasbatch[i] = Cbatch[i];
   } 

   accl_view.copy(Abatch, devAbatch, M * K * sizeof(double));
   accl_view.copy(Bbatch, devBbatch, K * N * sizeof(double));
   accl_view.copy(Cbatch, devCbatch, M * N * batchSize * sizeof(double));

   // NoTransA and NoTransB            
    typeA = NoTrans;
    typeB = NoTrans;
    Transa = CblasNoTrans;
    Transb = CblasNoTrans;

    // Column major 
    lda = M; ldb = K ; ldc = M;
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devCbatch, Chcblasbatch, M * N * batchSize * sizeof(double));
    for(int i = 0; i < batchSize; i++)
        cblas_dgemm( CblasColMajor, Transa, Transb, M, N, K, alpha, Abatch, lda, Bbatch, ldb, beta, CCblasbatch  + i * M * N ,ldc );
    for(int i = 0 ; i < M * N * batchSize; i++) 
        EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);

    /* alpha = 0 */
    lda = M; ldb = K ; ldc = M;
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, K, 0, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devCbatch, Chcblasbatch, M * N * batchSize * sizeof(double));
    for(int i = 0; i < batchSize; i++)
        cblas_dgemm( CblasColMajor, Transa, Transb, M, N, K, 0, Abatch, lda, Bbatch, ldb, beta, CCblasbatch  + i * M * N ,ldc );
    for(int i = 0 ; i < M * N * batchSize; i++) 
        EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);
    /* alpha = 0, beta = 0*/
    lda = M; ldb = K ; ldc = M;
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, K, 0, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, 0, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devCbatch, Chcblasbatch, M * N * batchSize * sizeof(double));
    for(int i = 0; i < batchSize; i++)
        cblas_dgemm( CblasColMajor, Transa, Transb, M, N, K, 0, Abatch, lda, Bbatch, ldb, 0, CCblasbatch  + i * M * N ,ldc );
    for(int i = 0 ; i < M * N * batchSize; i++) 
        EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);
    
    // Row Major 
    lda = K; ldb = N ; ldc = N;      
    status = hc.hcblas_dgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devCbatch, Chcblasbatch, M * N * batchSize * sizeof(double));
    for(int i = 0; i < batchSize; i++)
        cblas_dgemm( CblasRowMajor, Transa, Transb, M, N, K, alpha, Abatch, lda, Bbatch, ldb, beta, CCblasbatch  + i * M * N ,ldc );
    for(int i = 0 ; i < M * N * batchSize; i++) 
        EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);
    
    /* alpha = 0 */
    lda = K; ldb = N ; ldc = N;      
    status = hc.hcblas_dgemm(accl_view, RowMajor, typeA, typeB, M, N, K, 0, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devCbatch, Chcblasbatch, M * N * batchSize * sizeof(double));
    for(int i = 0; i < batchSize; i++)
        cblas_dgemm( CblasRowMajor, Transa, Transb, M, N, K, 0, Abatch, lda, Bbatch, ldb, beta, CCblasbatch  + i * M * N ,ldc );
    for(int i = 0 ; i < M * N * batchSize; i++) 
        EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);
    /* alpha = 0, beta = 0 */
    lda = K; ldb = N ; ldc = N;      
    status = hc.hcblas_dgemm(accl_view, RowMajor, typeA, typeB, M, N, K, 0, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, 0, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devCbatch, Chcblasbatch, M * N * batchSize * sizeof(double));
    for(int i = 0; i < batchSize; i++)
        cblas_dgemm( CblasRowMajor, Transa, Transb, M, N, K, 0, Abatch, lda, Bbatch, ldb, 0, CCblasbatch  + i * M * N ,ldc );
    for(int i = 0 ; i < M * N * batchSize; i++)
        EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);

    // NoTransA TransB   
    typeA = NoTrans;
    typeB = Trans;
    Transa = CblasNoTrans;
    Transb = CblasTrans;

    // Column major 
    lda = M; ldb = N ; ldc = M;
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devCbatch, Chcblasbatch, M * N * batchSize * sizeof(double));
    for(int i = 0; i < batchSize; i++)
        cblas_dgemm( CblasColMajor, Transa, Transb, M, N, K, alpha, Abatch, lda, Bbatch, ldb, beta, CCblasbatch  + i * M * N ,ldc );
    for(int i = 0 ; i < M * N * batchSize; i++) 
        EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);
    
    // Row Major  
    lda = K; ldb = K ; ldc = N;
    status = hc.hcblas_dgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devCbatch, Chcblasbatch, M * N * batchSize * sizeof(double));
    for(int i = 0; i < batchSize; i++)
        cblas_dgemm( CblasRowMajor, Transa, Transb, M, N, K, alpha, Abatch, lda, Bbatch, ldb, beta, CCblasbatch  + i * M * N ,ldc );
    for(int i = 0 ; i < M * N * batchSize; i++)
        EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);

    // TransA NoTransB 
    typeA = Trans;
    typeB = NoTrans;
    Transa = CblasTrans;
    Transb = CblasNoTrans;

    // Column major
    lda = K; ldb = K ; ldc = M;
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devCbatch, Chcblasbatch, M * N * batchSize * sizeof(double));
    for(int i = 0; i < batchSize; i++)
        cblas_dgemm( CblasColMajor, Transa, Transb, M, N, K, alpha, Abatch, lda, Bbatch, ldb, beta, CCblasbatch  + i * M * N ,ldc );
    for(int i = 0 ; i < M * N * batchSize; i++) 
        EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);
    
    // Row Major 
    lda = M; ldb = N ; ldc = N;
    status = hc.hcblas_dgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);  
    accl_view.copy(devCbatch, Chcblasbatch, M * N * batchSize * sizeof(double));
    for(int i = 0; i < batchSize; i++)
        cblas_dgemm( CblasRowMajor, Transa, Transb, M, N, K, alpha, Abatch, lda, Bbatch, ldb, beta, CCblasbatch  + i * M * N ,ldc );
    for(int i = 0 ; i < M * N * batchSize; i++) 
        EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);

    // TransA TransB 
    typeA = Trans;
    typeB = Trans;
    Transa = CblasTrans;
    Transb = CblasTrans;

    // Column major 
    lda = K; ldb = N ; ldc = M;
    status = hc.hcblas_dgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devCbatch, Chcblasbatch, M * N * batchSize * sizeof(double));
    for(int i = 0; i < batchSize; i++)
        cblas_dgemm( CblasColMajor, Transa, Transb, M, N, K, alpha, Abatch, lda, Bbatch, ldb, beta, CCblasbatch  + i * M * N ,ldc );
    for(int i = 0 ; i < M * N * batchSize; i++)
        EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);
    
    // Row Major 
    lda = M; ldb = K ; ldc = N;
    status = hc.hcblas_dgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
    EXPECT_EQ(status, HCBLAS_SUCCEEDS);
    accl_view.copy(devCbatch, Chcblasbatch, M * N * batchSize * sizeof(double));
    for(int i = 0; i < batchSize; i++)
        cblas_dgemm( CblasRowMajor, Transa, Transb, M, N, K, alpha, Abatch, lda, Bbatch, ldb, beta, CCblasbatch  + i * M * N ,ldc );
    for(int i = 0 ; i < M * N * batchSize; i++) 
        EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);
    free(Abatch);
    free(Bbatch);
    free(Cbatch);
    free(CCblasbatch);
    free(Chcblasbatch);
    hc::am_free(devAbatch);
    hc::am_free(devBbatch);
    hc::am_free(devCbatch);

}

