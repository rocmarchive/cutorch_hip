#include "hcblaslib.h"
#include <cstdlib>
#include "helper_functions.h"
#include "gtest/gtest.h"
#include "hc_am.hpp"
#include "cblas.h"
#include "test_constants.h"

TEST(hcblas_sgemm, return_correct_sgemm_Implementation_type_1) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
  int M = 189, N = 9, K = 19;
  float alpha = 1, beta = 1;
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
  // Implementation type I - Inputs and Outputs are HCC device pointers
  float* A = (float*) calloc(M * K, sizeof(float));
  float* B = (float*) calloc(K * N, sizeof(float));
  float* C = (float*) calloc(M * N, sizeof(float));
  float* devA = hc::am_alloc(sizeof(float) * M * K, acc, 0);
  float* devB = hc::am_alloc(sizeof(float) * K * N, acc, 0);
  float* devC = hc::am_alloc(sizeof(float) * M * N, acc, 0);

  for(int i = 0; i < M * K; i++) {
    A[i] = rand() % 100;
  }

  for(int i = 0; i < K * N; i++) {
    B[i] = rand() % 15;
  }

  for(int i = 0; i < M * N; i++) {
    C[i] = rand() % 25;
  }

  accl_view.copy(A, devA, M * K * sizeof(float));
  accl_view.copy(B, devB, K * N * sizeof(float));
  accl_view.copy(C, devC, M * N * sizeof(float));
  // NoTransA and NoTransB
  typeA = NoTrans;
  typeB = NoTrans;
  // Column major
  lda = M;
  ldb = K ;
  ldc = M;
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  // Row Major
  lda = K;
  ldb = N ;
  ldc = N;
  status = hc.hcblas_sgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  // NoTransA TransB
  typeA = NoTrans;
  typeB = Trans;
  // Column major
  lda = M;
  ldb = N ;
  ldc = M;
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  // Row Major
  lda = K;
  ldb = K ;
  ldc = N;
  status = hc.hcblas_sgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  // TransA NoTransB
  typeA = Trans;
  typeB = NoTrans;
  // Column major
  lda = K;
  ldb = K ;
  ldc = M;
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  // Row Major
  lda = M;
  ldb = N ;
  ldc = N;
  status = hc.hcblas_sgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  // TransA TransB
  typeA = Trans;
  typeB = Trans;
  // Column major
  lda = K;
  ldb = N ;
  ldc = M;
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  // Row Major
  lda = M;
  ldb = K ;
  ldc = N;
  status = hc.hcblas_sgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  typeA = NoTrans;
  typeB = NoTrans;
  lda = M;
  ldb = K ;
  ldc = M;
  hcOrder = ColMajor;
  float* devA1 = NULL;
  float* devB1 = NULL;
  float* devC1 = NULL;
  //A, B, C device pointers are not allocated properly
  status = hc.hcblas_sgemm(accl_view, hcOrder, typeA, typeB, M, N, K, alpha, devA1, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_INVALID);
  status = hc.hcblas_sgemm(accl_view, hcOrder, typeA, typeB, M, N, K, alpha, devA, lda, devB1, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_INVALID);
  status = hc.hcblas_sgemm(accl_view, hcOrder, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC1, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_INVALID);
  // M is 0
  status = hc.hcblas_sgemm(accl_view, hcOrder, typeA, typeB, 0, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_INVALID);
  // N is 0
  status = hc.hcblas_sgemm(accl_view, hcOrder, typeA, typeB, M, 0, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_INVALID);
  // K is 0
  status = hc.hcblas_sgemm(accl_view, hcOrder, typeA, typeB, M, N, 0, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_INVALID);
  free(A);
  free(B);
  free(C);
  hc::am_free(devA);
  hc::am_free(devB);
  hc::am_free(devC);
}

// Function to check Sgemm NoTransAB Column Major
void func_check_sgemmNN_Col_type_1(int M, int N, int K, float alpha, float beta, float tolerance) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
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
  // Implementation type I - Inputs and Outputs are HCC device pointers
  float* A = (float*) calloc(M * K, sizeof(float));
  float* B = (float*) calloc(K * N, sizeof(float));
  float* C = (float*) calloc(M * N, sizeof(float));
  float* C_hcblas = (float*) calloc(M * N, sizeof(float));
  float* C_cblas = (float*) calloc(M * N, sizeof(float));
  float* devA = hc::am_alloc(sizeof(float) * M * K, acc, 0);
  float* devB = hc::am_alloc(sizeof(float) * K * N, acc, 0);
  float* devC = hc::am_alloc(sizeof(float) * M * N, acc, 0);
  float X = 2;

  for(int i = 0; i < M * K; i++) {
    A[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
  }

  for(int i = 0; i < K * N; i++) {
    B[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
  }

  for(int i = 0; i < M * N; i++) {
    C[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
    C_cblas[i] = C[i];
  }

  accl_view.copy(A, devA, M * K * sizeof(float));
  accl_view.copy(B, devB, K * N * sizeof(float));
  accl_view.copy(C, devC, M * N * sizeof(float));
  // NoTransA and NoTransB
  typeA = NoTrans;
  typeB = NoTrans;
  Transa = CblasNoTrans;
  Transb = CblasNoTrans;
  // Column major
  lda = M;
  ldb = K ;
  ldc = M;
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devC, C_hcblas, M * N * sizeof(float));
  cblas_sgemm( CblasColMajor, Transa, Transb, M, N, K, alpha, A, lda, B, ldb, beta, C_cblas, ldc);

  float result = sgemmCompareL2fe(C_cblas, C_hcblas, M*N, tolerance);
  EXPECT_LE(result, tolerance);

#if 0
  for(int i = 0 ; i < M * N ; i++) {
    EXPECT_NEAR(C_hcblas[i], C_cblas[i], tolerance);
  }
#endif

  free(A);
  free(B);
  free(C);
  free(C_cblas);
  free(C_hcblas);
  hc::am_free(devA);
  hc::am_free(devB);
  hc::am_free(devC);
}

// Function to check Sgemm NoTransAB row Major
void func_check_sgemmNN_Row_type_1(int M, int N, int K, float alpha, float beta, float tolerance) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
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
  // Implementation type I - Inputs and Outputs are HCC device pointers
  float* A = (float*) calloc(M * K, sizeof(float));
  float* B = (float*) calloc(K * N, sizeof(float));
  float* C = (float*) calloc(M * N, sizeof(float));
  float* C_hcblas = (float*) calloc(M * N, sizeof(float));
  float* C_cblas = (float*) calloc(M * N, sizeof(float));
  float* devA = hc::am_alloc(sizeof(float) * M * K, acc, 0);
  float* devB = hc::am_alloc(sizeof(float) * K * N, acc, 0);
  float* devC = hc::am_alloc(sizeof(float) * M * N, acc, 0);
  float X = 2;

  for(int i = 0; i < M * K; i++) {
    A[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
  }

  for(int i = 0; i < K * N; i++) {
    B[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
  }

  for(int i = 0; i < M * N; i++) {
    C[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
    C_cblas[i] = C[i];
  }

  accl_view.copy(A, devA, M * K * sizeof(float));
  accl_view.copy(B, devB, K * N * sizeof(float));
  accl_view.copy(C, devC, M * N * sizeof(float));
  // NoTransA and NoTransB
  typeA = NoTrans;
  typeB = NoTrans;
  Transa = CblasNoTrans;
  Transb = CblasNoTrans;

  // Row Major
  lda = K;
  ldb = N ;
  ldc = N;
  status = hc.hcblas_sgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devC, C_hcblas, M * N * sizeof(float));
  cblas_sgemm( CblasRowMajor, Transa, Transb, M, N, K, alpha, A, lda, B, ldb, beta, C_cblas, ldc);

  float result = sgemmCompareL2fe(C_cblas, C_hcblas, M*N, tolerance);
  EXPECT_LE(result, tolerance);

#if 0
  for(int i = 0 ; i < M * N ; i++) {
    EXPECT_NEAR(C_hcblas[i], C_cblas[i], tolerance);
  }
#endif

  free(A);
  free(B);
  free(C);
  free(C_cblas);
  free(C_hcblas);
  hc::am_free(devA);
  hc::am_free(devB);
  hc::am_free(devC);
}



// Function to check Sgemm NoTransA Col Major
void func_check_sgemmNT_Col_type_1(int M, int N, int K, float alpha, float beta, float tolerance) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
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
  // Implementation type I - Inputs and Outputs are HCC device pointers
  float* A = (float*) calloc(M * K, sizeof(float));
  float* B = (float*) calloc(K * N, sizeof(float));
  float* C = (float*) calloc(M * N, sizeof(float));
  float* C_hcblas = (float*) calloc(M * N, sizeof(float));
  float* C_cblas = (float*) calloc(M * N, sizeof(float));
  float* devA = hc::am_alloc(sizeof(float) * M * K, acc, 0);
  float* devB = hc::am_alloc(sizeof(float) * K * N, acc, 0);
  float* devC = hc::am_alloc(sizeof(float) * M * N, acc, 0);
  float X = 2;

  for(int i = 0; i < M * K; i++) {
    A[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
  }

  for(int i = 0; i < K * N; i++) {
    B[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
  }

  for(int i = 0; i < M * N; i++) {
    C[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
    C_cblas[i] = C[i];
  }

  accl_view.copy(A, devA, M * K * sizeof(float));
  accl_view.copy(B, devB, K * N * sizeof(float));
  accl_view.copy(C, devC, M * N * sizeof(float));

  // NoTransA TransB
  typeA = NoTrans;
  typeB = Trans;
  Transa = CblasNoTrans;
  Transb = CblasTrans;
  // Column major
  lda = M;
  ldb = N ;
  ldc = M;
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devC, C_hcblas, M * N * sizeof(float));
  cblas_sgemm( CblasColMajor, Transa, Transb, M, N, K, alpha, A, lda, B, ldb, beta, C_cblas, ldc);

  float result = sgemmCompareL2fe(C_cblas, C_hcblas, M*N, tolerance);
  EXPECT_LE(result, tolerance);

#if 0
  for(int i = 0 ; i < M * N ; i++) {
    EXPECT_NEAR(C_hcblas[i], C_cblas[i], tolerance);
  }
#endif

  // alpha and beta are zeroes
  //alpha = 0
  lda = M;
  ldb = N ;
  ldc = M;
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, K, 0, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devC, C_hcblas, M * N * sizeof(float));
  cblas_sgemm( CblasColMajor, Transa, Transb, M, N, K, 0, A, lda, B, ldb, beta, C_cblas, ldc);

  result = sgemmCompareL2fe(C_cblas, C_hcblas, M*N, tolerance);
  EXPECT_LE(result, tolerance);

#if 0
  for(int i = 0 ; i < M * N ; i++) {
    EXPECT_NEAR(C_hcblas[i], C_cblas[i], tolerance);
  }
#endif

  //alpha = 0, beta = 0
  lda = M;
  ldb = N ;
  ldc = M;
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, K, 0, devA, lda, devB, ldb, 0, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devC, C_hcblas, M * N * sizeof(float));
  cblas_sgemm( CblasColMajor, Transa, Transb, M, N, K, 0, A, lda, B, ldb, 0, C_cblas, ldc);

  result = sgemmCompareL2fe(C_cblas, C_hcblas, M*N, tolerance);
  EXPECT_LE(result, tolerance);

#if 0
  for(int i = 0 ; i < M * N ; i++) {
    EXPECT_NEAR(C_hcblas[i], C_cblas[i], tolerance);
  }
#endif

  free(A);
  free(B);
  free(C);
  free(C_cblas);
  free(C_hcblas);
  hc::am_free(devA);
  hc::am_free(devB);
  hc::am_free(devC);
}


// Function to check Sgemm NoTransA Row Major
void func_check_sgemmNT_Row_type_1(int M, int N, int K, float alpha, float beta, float tolerance) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
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
  // Implementation type I - Inputs and Outputs are HCC device pointers
  float* A = (float*) calloc(M * K, sizeof(float));
  float* B = (float*) calloc(K * N, sizeof(float));
  float* C = (float*) calloc(M * N, sizeof(float));
  float* C_hcblas = (float*) calloc(M * N, sizeof(float));
  float* C_cblas = (float*) calloc(M * N, sizeof(float));
  float* devA = hc::am_alloc(sizeof(float) * M * K, acc, 0);
  float* devB = hc::am_alloc(sizeof(float) * K * N, acc, 0);
  float* devC = hc::am_alloc(sizeof(float) * M * N, acc, 0);
  float X = 2;

  for(int i = 0; i < M * K; i++) {
    A[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
  }

  for(int i = 0; i < K * N; i++) {
    B[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
  }

  for(int i = 0; i < M * N; i++) {
    C[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
    C_cblas[i] = C[i];
  }

  accl_view.copy(A, devA, M * K * sizeof(float));
  accl_view.copy(B, devB, K * N * sizeof(float));
  accl_view.copy(C, devC, M * N * sizeof(float));

  // NoTransA TransB
  typeA = NoTrans;
  typeB = Trans;
  Transa = CblasNoTrans;
  Transb = CblasTrans;
  
  // Row Major
  lda = K;
  ldb = K ;
  ldc = N;
  status = hc.hcblas_sgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devC, C_hcblas, M * N * sizeof(float));
  cblas_sgemm(CblasRowMajor, Transa, Transb, M, N, K, alpha, A, lda, B, ldb, beta, C_cblas, ldc);

  float result = sgemmCompareL2fe(C_cblas, C_hcblas, M*N, tolerance);
  EXPECT_LE(result, tolerance);

#if 0
  for(int i = 0 ; i < M * N ; i++) {
    EXPECT_NEAR(C_hcblas[i], C_cblas[i], tolerance);
  }
#endif

  // alpha and beta are zeroes
  // alpha = 0
  lda = K;
  ldb = K ;
  ldc = N;
  status = hc.hcblas_sgemm(accl_view, RowMajor, typeA, typeB, M, N, K, 0, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devC, C_hcblas, M * N * sizeof(float));
  cblas_sgemm(CblasRowMajor, Transa, Transb, M, N, K, 0, A, lda, B, ldb, beta, C_cblas, ldc);

  result = sgemmCompareL2fe(C_cblas, C_hcblas, M*N, tolerance);
  EXPECT_LE(result, tolerance);

#if 0
  for(int i = 0 ; i < M * N ; i++) {
    EXPECT_NEAR(C_hcblas[i], C_cblas[i], tolerance);
  }
#endif

  // alpha = 0, beta = 0
  lda = K;
  ldb = K ;
  ldc = N;
  status = hc.hcblas_sgemm(accl_view, RowMajor, typeA, typeB, M, N, K, 0, devA, lda, devB, ldb, 0, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devC, C_hcblas, M * N * sizeof(float));
  cblas_sgemm(CblasRowMajor, Transa, Transb, M, N, K, 0, A, lda, B, ldb, 0, C_cblas, ldc);

  result = sgemmCompareL2fe(C_cblas, C_hcblas, M*N, tolerance);
  EXPECT_LE(result, tolerance);

#if 0
  for(int i = 0 ; i < M * N ; i++) {
    EXPECT_NEAR(C_hcblas[i], C_cblas[i], tolerance);
  }
#endif

  free(A);
  free(B);
  free(C);
  free(C_cblas);
  free(C_hcblas);
  hc::am_free(devA);
  hc::am_free(devB);
  hc::am_free(devC);
}

// Function to check Sgemm NoTransB Col Major
void func_check_sgemmTN_Col_type_1(int M, int N, int K, float alpha, float beta, float tolerance) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
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
  // Implementation type I - Inputs and Outputs are HCC device pointers
  float* A = (float*) calloc(M * K, sizeof(float));
  float* B = (float*) calloc(K * N, sizeof(float));
  float* C = (float*) calloc(M * N, sizeof(float));
  float* C_hcblas = (float*) calloc(M * N, sizeof(float));
  float* C_cblas = (float*) calloc(M * N, sizeof(float));
  float* devA = hc::am_alloc(sizeof(float) * M * K, acc, 0);
  float* devB = hc::am_alloc(sizeof(float) * K * N, acc, 0);
  float* devC = hc::am_alloc(sizeof(float) * M * N, acc, 0);
  float X = 2;

  for(int i = 0; i < M * K; i++) {
    A[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
  }

  for(int i = 0; i < K * N; i++) {
    B[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
  }

  for(int i = 0; i < M * N; i++) {
    C[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
    C_cblas[i] = C[i];
  }

  accl_view.copy(A, devA, M * K * sizeof(float));
  accl_view.copy(B, devB, K * N * sizeof(float));
  accl_view.copy(C, devC, M * N * sizeof(float));
  
  // TransA NoTransB
  typeA = Trans;
  typeB = NoTrans;
  Transa = CblasTrans;
  Transb = CblasNoTrans;
  // Column major
  lda = K;
  ldb = K ;
  ldc = M;
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devC, C_hcblas, M * N * sizeof(float));
  cblas_sgemm( CblasColMajor, Transa, Transb, M, N, K, alpha, A, lda, B, ldb, beta, C_cblas, ldc);

  float result = sgemmCompareL2fe(C_cblas, C_hcblas, M*N, tolerance);
  EXPECT_LE(result, tolerance);

#if 0
  for(int i = 0 ; i < M * N ; i++) {
    EXPECT_NEAR(C_hcblas[i], C_cblas[i], tolerance);
  }
#endif

  free(A);
  free(B);
  free(C);
  free(C_cblas);
  free(C_hcblas);
  hc::am_free(devA);
  hc::am_free(devB);
  hc::am_free(devC);
}


// Function to check Sgemm NoTransB Row Major
void func_check_sgemmTN_Row_type_1(int M, int N, int K, float alpha, float beta, float tolerance) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
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
  // Implementation type I - Inputs and Outputs are HCC device pointers
  float* A = (float*) calloc(M * K, sizeof(float));
  float* B = (float*) calloc(K * N, sizeof(float));
  float* C = (float*) calloc(M * N, sizeof(float));
  float* C_hcblas = (float*) calloc(M * N, sizeof(float));
  float* C_cblas = (float*) calloc(M * N, sizeof(float));
  float* devA = hc::am_alloc(sizeof(float) * M * K, acc, 0);
  float* devB = hc::am_alloc(sizeof(float) * K * N, acc, 0);
  float* devC = hc::am_alloc(sizeof(float) * M * N, acc, 0);
  float X = 2;

  for(int i = 0; i < M * K; i++) {
    A[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
  }

  for(int i = 0; i < K * N; i++) {
    B[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
  }

  for(int i = 0; i < M * N; i++) {
    C[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
    C_cblas[i] = C[i];
  }

  accl_view.copy(A, devA, M * K * sizeof(float));
  accl_view.copy(B, devB, K * N * sizeof(float));
  accl_view.copy(C, devC, M * N * sizeof(float));
  
  // TransA NoTransB
  typeA = Trans;
  typeB = NoTrans;
  Transa = CblasTrans;
  Transb = CblasNoTrans;

  // Row Major
  lda = M;
  ldb = N ;
  ldc = N;
  status = hc.hcblas_sgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devC, C_hcblas, M * N * sizeof(float));
  cblas_sgemm( CblasRowMajor, Transa, Transb, M, N, K, alpha, A, lda, B, ldb, beta, C_cblas, ldc);

  float result = sgemmCompareL2fe(C_cblas, C_hcblas, M*N, tolerance);
  EXPECT_LE(result, tolerance);

#if 0
  for(int i = 0 ; i < M * N ; i++) {
    EXPECT_NEAR(C_hcblas[i], C_cblas[i], tolerance);
  }
#endif

  free(A);
  free(B);
  free(C);
  free(C_cblas);
  free(C_hcblas);
  hc::am_free(devA);
  hc::am_free(devB);
  hc::am_free(devC);
}

// Function to check Sgemm TransAB Col Major
void func_check_sgemmTT_Col_type_1(int M, int N, int K, float alpha, float beta, float tolerance) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
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
  // Implementation type I - Inputs and Outputs are HCC device pointers
  float* A = (float*) calloc(M * K, sizeof(float));
  float* B = (float*) calloc(K * N, sizeof(float));
  float* C = (float*) calloc(M * N, sizeof(float));
  float* C_hcblas = (float*) calloc(M * N, sizeof(float));
  float* C_cblas = (float*) calloc(M * N, sizeof(float));
  float* devA = hc::am_alloc(sizeof(float) * M * K, acc, 0);
  float* devB = hc::am_alloc(sizeof(float) * K * N, acc, 0);
  float* devC = hc::am_alloc(sizeof(float) * M * N, acc, 0);
  float X = 2;

  for(int i = 0; i < M * K; i++) {
    A[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
  }

  for(int i = 0; i < K * N; i++) {
    B[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
  }

  for(int i = 0; i < M * N; i++) {
    C[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
    C_cblas[i] = C[i];
  }

  accl_view.copy(A, devA, M * K * sizeof(float));
  accl_view.copy(B, devB, K * N * sizeof(float));
  accl_view.copy(C, devC, M * N * sizeof(float));

  // TransA TransB
  typeA = Trans;
  typeB = Trans;
  Transa = CblasTrans;
  Transb = CblasTrans;

  
  // Column major 
  lda = K;
  ldb = N ;
  ldc = M;
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devC, C_hcblas, M * N * sizeof(float));
  cblas_sgemm( CblasColMajor, Transa, Transb, M, N, K, alpha, A, lda, B, ldb, beta, C_cblas, ldc);

  float result = sgemmCompareL2fe(C_cblas, C_hcblas, M*N, tolerance);
  EXPECT_LE(result, tolerance);

#if 0
  for(int i = 0 ; i < M * N ; i++) {
    EXPECT_NEAR(C_hcblas[i], C_cblas[i], tolerance);
  }
#endif

  free(A);
  free(B);
  free(C);
  free(C_cblas);
  free(C_hcblas);
  hc::am_free(devA);
  hc::am_free(devB);
  hc::am_free(devC);
}

// Function to check Sgemm TransAB Row Major
void func_check_sgemmTT_Row_type_1(int M, int N, int K, float alpha, float beta, float tolerance) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
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
  // Implementation type I - Inputs and Outputs are HCC device pointers
  float* A = (float*) calloc(M * K, sizeof(float));
  float* B = (float*) calloc(K * N, sizeof(float));
  float* C = (float*) calloc(M * N, sizeof(float));
  float* C_hcblas = (float*) calloc(M * N, sizeof(float));
  float* C_cblas = (float*) calloc(M * N, sizeof(float));
  float* devA = hc::am_alloc(sizeof(float) * M * K, acc, 0);
  float* devB = hc::am_alloc(sizeof(float) * K * N, acc, 0);
  float* devC = hc::am_alloc(sizeof(float) * M * N, acc, 0);
  float X = 2;

  for(int i = 0; i < M * K; i++) {
    A[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
  }

  for(int i = 0; i < K * N; i++) {
    B[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
  }

  for(int i = 0; i < M * N; i++) {
    C[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
    C_cblas[i] = C[i];
  }

  accl_view.copy(A, devA, M * K * sizeof(float));
  accl_view.copy(B, devB, K * N * sizeof(float));
  accl_view.copy(C, devC, M * N * sizeof(float));

  // TransA TransB
  typeA = Trans;
  typeB = Trans;
  Transa = CblasTrans;
  Transb = CblasTrans;

  // Row Major
  lda = M;
  ldb = K ;
  ldc = N;
  status = hc.hcblas_sgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devC, C_hcblas, M * N * sizeof(float));
  cblas_sgemm( CblasRowMajor, Transa, Transb, M, N, K, alpha, A, lda, B, ldb, beta, C_cblas, ldc);

  float result = sgemmCompareL2fe(C_cblas, C_hcblas, M*N, tolerance);
  EXPECT_LE(result, tolerance);

#if 0
  for(int i = 0 ; i < M * N ; i++) {
    EXPECT_NEAR(C_hcblas[i], C_cblas[i], tolerance);
  }
#endif

  free(A);
  free(B);
  free(C);
  free(C_cblas);
  free(C_hcblas);
  hc::am_free(devA);
  hc::am_free(devB);
  hc::am_free(devC);
}

// Case A:  Square Cases tests
// Order : Column 

// Type NoTransAB
// check square matrices of VVSmall input sizes
TEST(hcblas_sgemm, func_correct_sgemmNN_Col_square_vvsmall_Implementation_type_1) {
 int M, N, K;
 M = N = K = gen_vvsmall();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmNN_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// check square matrices of VSmall input sizes
TEST(hcblas_sgemm, func_correct_sgemmNN_Col_square_vsmall_Implementation_type_1) {
 int M, N, K;
 M = N = K = gen_vsmall();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmNN_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// check square matrices of small input sizes
TEST(hcblas_sgemm, func_correct_sgemmNN_Col_square_small_Implementation_type_1) {
 int M, N, K;
 M = N = K = gen_small();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmNN_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}


// check square matrices of regular input sizes
TEST(hcblas_sgemm, func_correct_sgemmNN_Col_square_regular_Implementation_type_1) {
 int M, N, K;
 M = N = K = gen_regular();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmNN_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// Type NoTransA
// check square matrices of VVSmall input sizes
TEST(hcblas_sgemm, func_correct_sgemmNT_Col_square_vvsmall_Implementation_type_1) {
 int M, N, K;
 M = N = K = gen_vvsmall();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmNT_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// check square matrices of VSmall input sizes
TEST(hcblas_sgemm, func_correct_sgemmNT_Col_square_vsmall_Implementation_type_1) {
 int M, N, K;
 M = N = K = gen_vsmall();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmNT_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// check square matrices of small input sizes
TEST(hcblas_sgemm, func_correct_sgemmNT_Col_square_small_Implementation_type_1) {
 int M, N, K;
 M = N = K = gen_small();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmNT_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}


// check square matrices of regular input sizes
TEST(hcblas_sgemm, func_correct_sgemmNT_Col_square_regular_Implementation_type_1) {
 int M, N, K;
 M = N = K = gen_regular();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmNT_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// check square matrices of large  input sizes
TEST(hcblas_sgemm, func_correct_sgemmNT_Col_square_large_Implementation_type_1) {
 int M, N, K;
 M = N = K = gen_large();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmNT_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// Type NoTransB
// check square matrices of VVSmall input sizes
TEST(hcblas_sgemm, func_correct_sgemmTN_Col_square_vvsmall_Implementation_type_1) {
 int M, N, K;
 M = N = K = gen_vvsmall();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmTN_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// check square matrices of VSmall input sizes
TEST(hcblas_sgemm, func_correct_sgemmTN_Col_square_vsmall_Implementation_type_1) {
 int M, N, K;
 M = N = K = gen_vsmall();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmTN_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// check square matrices of small input sizes
TEST(hcblas_sgemm, func_correct_sgemmTN_Col_square_small_Implementation_type_1) {
 int M, N, K;
 M = N = K = gen_small();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmTN_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}


// check square matrices of regular input sizes
TEST(hcblas_sgemm, func_correct_sgemmTN_Col_square_regular_Implementation_type_1) {
 int M, N, K;
 M = N = K = gen_regular();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmTN_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// check square matrices of large  input sizes
TEST(hcblas_sgemm, func_correct_sgemmTN_Col_square_large_Implementation_type_1) {
 int M, N, K;
 M = N = K = gen_large();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmTN_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// Type TransAB
// check square matrices of VVSmall input sizes
TEST(hcblas_sgemm, func_correct_sgemmTT_Col_square_vvsmall_Implementation_type_1) {
 int M, N, K;
 M = N = K = gen_vvsmall();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmTT_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// check square matrices of VSmall input sizes
TEST(hcblas_sgemm, func_correct_sgemmTT_Col_square_vsmall_Implementation_type_1) {
 int M, N, K;
 M = N = K = gen_vsmall();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmTT_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// check square matrices of small input sizes
TEST(hcblas_sgemm, func_correct_sgemmTT_Col_square_small_Implementation_type_1) {
 int M, N, K;
 M = N = K = gen_small();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmTT_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// check square matrices of regular input sizes
TEST(hcblas_sgemm, func_correct_sgemmTT_Col_square_regular_Implementation_type_1) {
 int M, N, K;
 M = N = K = gen_regular();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmTT_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// CASE 2: Slim A Fat B
// SGEMM NN Case

// check slim A with large M and Vsmall K
TEST(hcblas_sgemm, func_correct_sgemmNN_Col_slimA_vsmallK_Implementation_type_1) {
 int M, N, K;
 M = N = gen_vlarge();
 K = gen_vsmall();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmNN_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// check slim A with large M and small K
TEST(hcblas_sgemm, func_correct_sgemmNN_Col_slimA_smallK_Implementation_type_1) {
 int M, N, K;
 M = N = gen_vlarge();
 K = gen_small();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmNN_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// SGEMM NT Case

// check slim A with large M and Vsmall K
TEST(hcblas_sgemm, func_correct_sgemmNT_Col_slimA_vsmallK_Implementation_type_1) {
 int M, N, K;
 M = N = gen_vlarge();
 K = gen_vsmall();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmNT_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// check slim A with large M and small K
TEST(hcblas_sgemm, func_correct_sgemmNT_Col_slimA_smallK_Implementation_type_1) {
 int M, N, K;
 M = N = gen_vlarge();
 K = gen_small();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmNT_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// SGEMM TN Case

// check slim A with large M and Vsmall K
TEST(hcblas_sgemm, func_correct_sgemmTN_Col_slimA_vsmallK_Implementation_type_1) {
 int M, N, K;
 M = N = gen_vlarge();
 K = gen_vsmall();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmTN_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// check slim A with large M and small K
TEST(hcblas_sgemm, func_correct_sgemmTN_Col_slimA_smallK_Implementation_type_1) {
 int M, N, K;
 M = N = gen_vlarge();
 K = gen_small();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmTN_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// SGEMM TT Case

// check slim A with large M and Vsmall K
TEST(hcblas_sgemm, func_correct_sgemmTT_Col_slimA_vsmallK_Implementation_type_1) {
 int M, N, K;
 M = N = gen_vlarge();
 K = gen_vsmall();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmTT_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// check slim A with large M and small K
TEST(hcblas_sgemm, func_correct_sgemmTT_Col_slimA_smallK_Implementation_type_1) {
 int M, N, K;
 M = N = gen_vlarge();
 K = gen_small();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmTT_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// check slim A with large M and regular K
TEST(hcblas_sgemm, func_correct_sgemmTT_Col_slimA_regularK_Implementation_type_1) {
 int M, N, K;
 M = N = gen_vlarge();
 K = gen_regular();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmTT_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}


// CASE 2: Slim C Fat B Square A
// SGEMM NN Case

// check slim C with large M and Vsmall N
TEST(hcblas_sgemm, func_correct_sgemmNN_Col_slimC_vsmallN_Implementation_type_1) {
 int M, N, K;
 M = K = gen_vlarge();
 N = gen_vsmall();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmNN_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// check slim C with large M and small N
TEST(hcblas_sgemm, func_correct_sgemmNN_Col_slimC_smallN_Implementation_type_1) {
 int M, N, K;
 M = K = gen_vlarge();
 N = gen_small();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmNN_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// check slim C with large M and regular N
TEST(hcblas_sgemm, func_correct_sgemmNN_Col_slimC_regularN_Implementation_type_1) {
 int M, N, K;
 M = K = gen_vlarge();
 N = gen_regular();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmNN_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// SGEMM NT Case

// check slim C with large M and small N
TEST(hcblas_sgemm, func_correct_sgemmNT_Col_slimC_smallN_Implementation_type_1) {
 int M, N, K;
 M = K = gen_vlarge();
 N = gen_small();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmNT_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// check slim C with large M and regular N
TEST(hcblas_sgemm, func_correct_sgemmNT_Col_slimC_regularN_Implementation_type_1) {
 int M, N, K;
 M = K = gen_vlarge();
 N = gen_regular();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmNT_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// SGEMM TN Case
// check slim C with large M and Vsmall N
TEST(hcblas_sgemm, func_correct_sgemmTN_Col_slimC_vsmallN_Implementation_type_1) {
 int M, N, K;
 M = K = gen_vlarge();
 N = gen_vsmall();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmTN_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// check slim C with large M and small N
TEST(hcblas_sgemm, func_correct_sgemmTN_Col_slimC_smallN_Implementation_type_1) {
 int M, N, K;
 M = K = gen_vlarge();
 N = gen_small();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmTN_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// check slim C with large M and regular N
TEST(hcblas_sgemm, func_correct_sgemmTN_Col_slimC_regularN_Implementation_type_1) {
 int M, N, K;
 M = K = gen_vlarge();
 N = gen_regular();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmTN_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}
// SGEMM TT Case

// check slim C with large M and small N
TEST(hcblas_sgemm, func_correct_sgemmTT_Col_slimC_smallN_Implementation_type_1) {
 int M, N, K;
 M = K = gen_vlarge();
 N = gen_small();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmTT_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

// check slim C with large M and regular N
TEST(hcblas_sgemm, func_correct_sgemmTT_Col_slimC_regularN_Implementation_type_1) {
 int M, N, K;
 M = K = gen_vlarge();
 N = gen_regular();
 float alpha = ((float)rand()/(float)(RAND_MAX)) * 1.172; 
 float beta = ((float)rand()/(float)(RAND_MAX)) * 3.414; 
 func_check_sgemmTT_Col_type_1(M, N, K, alpha, beta, 1.0e-5f);
}

TEST(hcblas_sgemm, return_correct_sgemm_Implementation_type_2) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
  int M = 189, N = 9, K = 19;
  float alpha = 1, beta = 1;
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
  // Implementation type II - Inputs and Outputs are HCC float array containers with batch processing
  float* Abatch = (float*) calloc(M * K, sizeof(float));
  float* Bbatch = (float*) calloc(K * N, sizeof(float));
  float* Cbatch = (float*) calloc(M * N * batchSize, sizeof(float));
  float* devAbatch = hc::am_alloc(sizeof(float) * M * K, acc, 0);
  float* devBbatch = hc::am_alloc(sizeof(float) * K * N, acc, 0);
  float* devCbatch = hc::am_alloc(sizeof(float) * M * N * batchSize, acc, 0);

  for(int i = 0; i < M * K; i++) {
    Abatch[i] = rand() % 100;
  }

  for(int i = 0; i < K * N; i++) {
    Bbatch[i] = rand() % 15;
  }

  for(int i = 0; i < M * N * batchSize; i++) {
    Cbatch[i] = rand() % 25;
  }

  accl_view.copy(Abatch, devAbatch, M * K * sizeof(float));
  accl_view.copy(Bbatch, devBbatch, K * N * sizeof(float));
  accl_view.copy(Cbatch, devCbatch, M * N * batchSize * sizeof(float));
  // NoTransA and NoTransB
  typeA = NoTrans;
  typeB = NoTrans;
  // Column major
  lda = M;
  ldb = K ;
  ldc = M;
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  // Row Major
  lda = K;
  ldb = N ;
  ldc = N;
  status = hc.hcblas_sgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  // NoTransA TransB
  typeA = NoTrans;
  typeB = Trans;
  // Column major
  lda = M;
  ldb = N ;
  ldc = M;
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  // Row Major
  lda = K;
  ldb = K ;
  ldc = N;
  status = hc.hcblas_sgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  // TransA NoTransB
  typeA = Trans;
  typeB = NoTrans;
  // Column major
  lda = K;
  ldb = K ;
  ldc = M;
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  // Row Major
  lda = M;
  ldb = N ;
  ldc = N;
  status = hc.hcblas_sgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  // TransA TransB
  typeA = Trans;
  typeB = Trans;
  // Column major
  lda = K;
  ldb = N ;
  ldc = M;
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  // Row Major
  lda = M;
  ldb = K ;
  ldc = N;
  status = hc.hcblas_sgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  typeA = NoTrans;
  typeB = NoTrans;
  lda = M;
  ldb = K ;
  ldc = M;
  float* devA1 = NULL;
  float* devB1 = NULL;
  float* devC1 = NULL;
  /* A, B, C device pointers are not allocated properly */
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devA1, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_INVALID);
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devB1, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_INVALID);
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devC1, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_INVALID);
  // M is 0
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, 0, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_INVALID);
  // N is 0
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, 0, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_INVALID);
  // K is 0
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, 0, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_INVALID);
  free(Abatch);
  free(Bbatch);
  free(Cbatch);
  hc::am_free(devAbatch);
  hc::am_free(devBbatch);
  hc::am_free(devCbatch);
}

TEST(hcblas_sgemm, func_correct_sgemm_Implementation_type_2) {
   hc::accelerator accl;
   Hcblaslibrary hc(&accl);
  int M = 189, N = 9, K = 19;
  float alpha = 1, beta = 1;
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
  // Implementation type II - Inputs and Outputs are HCC float array containers with batch processing
  float* Abatch = (float*) calloc(M * K, sizeof(float));
  float* Bbatch = (float*) calloc(K * N, sizeof(float));
  float* Cbatch = (float*) calloc(M * N * batchSize, sizeof(float));
  float* CCblasbatch = (float*) calloc(M * N * batchSize, sizeof(float));
  float* Chcblasbatch = (float*) calloc(M * N * batchSize, sizeof(float));
  float* devAbatch = hc::am_alloc(sizeof(float) * M * K, acc, 0);
  float* devBbatch = hc::am_alloc(sizeof(float) * K * N, acc, 0);
  float* devCbatch = hc::am_alloc(sizeof(float) * M * N * batchSize, acc, 0);

  for(int i = 0; i < M * K; i++) {
    Abatch[i] = rand() % 100;
  }

  for(int i = 0; i < K * N; i++) {
    Bbatch[i] = rand() % 15;
  }

  for(int i = 0; i < M * N * batchSize; i++) {
    Cbatch[i] = rand() % 25;
    CCblasbatch[i] = Cbatch[i];
  }

  accl_view.copy(Abatch, devAbatch, M * K * sizeof(float));
  accl_view.copy(Bbatch, devBbatch, K * N * sizeof(float));
  accl_view.copy(Cbatch, devCbatch, M * N * batchSize * sizeof(float));
  // NoTransA and NoTransB
  typeA = NoTrans;
  typeB = NoTrans;
  Transa = CblasNoTrans;
  Transb = CblasNoTrans;
  // Column major
  lda = M;
  ldb = K ;
  ldc = M;
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devCbatch, Chcblasbatch, M * N * batchSize * sizeof(float));

  for(int i = 0; i < batchSize; i++) {
    cblas_sgemm( CblasColMajor, Transa, Transb, M, N, K, alpha, Abatch, lda, Bbatch, ldb, beta, CCblasbatch  + i * M * N , ldc );
  }

  for(int i = 0 ; i < M * N * batchSize; i++) {
    EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);
  }

  // alpha = 0
  lda = M;
  ldb = K ;
  ldc = M;
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, K, 0, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devCbatch, Chcblasbatch, M * N * batchSize * sizeof(float));

  for(int i = 0; i < batchSize; i++) {
    cblas_sgemm( CblasColMajor, Transa, Transb, M, N, K, 0, Abatch, lda, Bbatch, ldb, beta, CCblasbatch  + i * M * N , ldc );
  }

  for(int i = 0 ; i < M * N * batchSize; i++) {
    EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);
  }

  // alpha = 0, beta = 0
  lda = M;
  ldb = K ;
  ldc = M;
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, K, 0, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, 0, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devCbatch, Chcblasbatch, M * N * batchSize * sizeof(float));

  for(int i = 0; i < batchSize; i++) {
    cblas_sgemm( CblasColMajor, Transa, Transb, M, N, K, 0, Abatch, lda, Bbatch, ldb, 0, CCblasbatch  + i * M * N , ldc );
  }

  for(int i = 0 ; i < M * N * batchSize; i++) {
    EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);
  }

  // Row Major
  lda = K;
  ldb = N ;
  ldc = N;
  status = hc.hcblas_sgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devCbatch, Chcblasbatch, M * N * batchSize * sizeof(float));

  for(int i = 0; i < batchSize; i++) {
    cblas_sgemm( CblasRowMajor, Transa, Transb, M, N, K, alpha, Abatch, lda, Bbatch, ldb, beta, CCblasbatch  + i * M * N , ldc );
  }

  for(int i = 0 ; i < M * N * batchSize; i++) {
    EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);
  }

  // alpha = 0
  lda = K;
  ldb = N ;
  ldc = N;
  status = hc.hcblas_sgemm(accl_view, RowMajor, typeA, typeB, M, N, K, 0, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devCbatch, Chcblasbatch,  M * N * batchSize * sizeof(float));

  for(int i = 0; i < batchSize; i++) {
    cblas_sgemm( CblasRowMajor, Transa, Transb, M, N, K, 0, Abatch, lda, Bbatch, ldb, beta, CCblasbatch  + i * M * N , ldc );
  }

  for(int i = 0 ; i < M * N * batchSize; i++) {
    EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);
  }

  // alpha = 0, beta = 0
  lda = K;
  ldb = N ;
  ldc = N;
  status = hc.hcblas_sgemm(accl_view, RowMajor, typeA, typeB, M, N, K, 0, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, 0, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devCbatch, Chcblasbatch, M * N * batchSize * sizeof(float));

  for(int i = 0; i < batchSize; i++) {
    cblas_sgemm( CblasRowMajor, Transa, Transb, M, N, K, 0, Abatch, lda, Bbatch, ldb, 0, CCblasbatch  + i * M * N , ldc );
  }

  for(int i = 0 ; i < M * N * batchSize; i++) {
    EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);
  }

  // NoTransA TransB
  typeA = NoTrans;
  typeB = Trans;
  Transa = CblasNoTrans;
  Transb = CblasTrans;
  // Column major
  lda = M;
  ldb = N ;
  ldc = M;
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devCbatch, Chcblasbatch, M * N * batchSize * sizeof(float));

  for(int i = 0; i < batchSize; i++) {
    cblas_sgemm( CblasColMajor, Transa, Transb, M, N, K, alpha, Abatch, lda, Bbatch, ldb, beta, CCblasbatch  + i * M * N , ldc );
  }

  for(int i = 0 ; i < M * N * batchSize; i++) {
    EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);
  }

  // Row Major
  lda = K;
  ldb = K ;
  ldc = N;
  status = hc.hcblas_sgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devCbatch, Chcblasbatch, M * N * batchSize * sizeof(float));

  for(int i = 0; i < batchSize; i++) {
    cblas_sgemm( CblasRowMajor, Transa, Transb, M, N, K, alpha, Abatch, lda, Bbatch, ldb, beta, CCblasbatch  + i * M * N , ldc );
  }

  for(int i = 0 ; i < M * N * batchSize; i++) {
    EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);
  }

  // TransA NoTransB
  typeA = Trans;
  typeB = NoTrans;
  Transa = CblasTrans;
  Transb = CblasNoTrans;
  // Column major
  lda = K;
  ldb = K ;
  ldc = M;
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devCbatch, Chcblasbatch,  M * N * batchSize * sizeof(float));

  for(int i = 0; i < batchSize; i++) {
    cblas_sgemm( CblasColMajor, Transa, Transb, M, N, K, alpha, Abatch, lda, Bbatch, ldb, beta, CCblasbatch  + i * M * N , ldc );
  }

  for(int i = 0 ; i < M * N * batchSize; i++) {
    EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);
  }

  // Row Major
  lda = M;
  ldb = N ;
  ldc = N;
  status = hc.hcblas_sgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devCbatch, Chcblasbatch, M * N * batchSize * sizeof(float));

  for(int i = 0; i < batchSize; i++) {
    cblas_sgemm( CblasRowMajor, Transa, Transb, M, N, K, alpha, Abatch, lda, Bbatch, ldb, beta, CCblasbatch  + i * M * N , ldc );
  }

  for(int i = 0 ; i < M * N * batchSize; i++) {
    EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);
  }

  // TransA TransB
  typeA = Trans;
  typeB = Trans;
  Transa = CblasTrans;
  Transb = CblasTrans;
  // Column major
  lda = K;
  ldb = N ;
  ldc = M;
  status = hc.hcblas_sgemm(accl_view, ColMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devCbatch, Chcblasbatch, M * N * batchSize * sizeof(float));

  for(int i = 0; i < batchSize; i++) {
    cblas_sgemm( CblasColMajor, Transa, Transb, M, N, K, alpha, Abatch, lda, Bbatch, ldb, beta, CCblasbatch  + i * M * N , ldc );
  }

  for(int i = 0 ; i < M * N * batchSize; i++) {
    EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);
  }

  // Row Major
  lda = M;
  ldb = K ;
  ldc = N;
  status = hc.hcblas_sgemm(accl_view, RowMajor, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);
  EXPECT_EQ(status, HCBLAS_SUCCEEDS);
  accl_view.copy(devCbatch, Chcblasbatch, M * N * batchSize * sizeof(float));

  for(int i = 0; i < batchSize; i++) {
    cblas_sgemm( CblasRowMajor, Transa, Transb, M, N, K, alpha, Abatch, lda, Bbatch, ldb, beta, CCblasbatch  + i * M * N , ldc );
  }

  for(int i = 0 ; i < M * N * batchSize; i++) {
    EXPECT_EQ(Chcblasbatch[i], CCblasbatch[i]);
  }

  free(Abatch);
  free(Bbatch);
  free(Cbatch);
  free(CCblasbatch);
  free(Chcblasbatch);
  hc::am_free(devAbatch);
  hc::am_free(devBbatch);
  hc::am_free(devCbatch);
}

