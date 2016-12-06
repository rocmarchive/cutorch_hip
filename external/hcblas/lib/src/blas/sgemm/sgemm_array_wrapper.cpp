#include "sgemm_array_kernels.h"

// Sgemm Wrapper routine that invokes the appropriate kernel routines depending on the input dimension M N and K
hcblasStatus gemm_HC(hc::accelerator_view accl_view,
                     const int order, char TransA, char TransB,
                     const int M, const int N, const int K,
                     const float alpha, float *A_mat,
                     long aOffset, long lda,
                     float *B_mat,
                     long bOffset, long ldb, const float beta,
                     float *C_mat,
                     long cOffset, long ldc,
                     long A_batchOffset = 0, long B_batchOffset = 0, long C_batchOffset = 0, int batchSize = 0) {
  hcblasStatus status = HCBLAS_SUCCEEDS;
  // Start the operations

  if (order) {
    if(batchSize > 0) {
      if (TransB == 'n') {
        if (TransA == 'n') {
          status = gemm_NoTransAB(accl_view, A_mat, aOffset, A_batchOffset, B_mat, bOffset, B_batchOffset, C_mat, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
        } else {
          status = gemm_NoTransB(accl_view, A_mat, aOffset, A_batchOffset, B_mat, bOffset, B_batchOffset, C_mat, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
        }
      } else if (TransA == 'n') {
        status = gemm_NoTransA(accl_view, A_mat, aOffset, A_batchOffset, B_mat, bOffset, B_batchOffset, C_mat, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
      } else {
        status = gemm_TransAB(accl_view, A_mat, aOffset, A_batchOffset, B_mat, bOffset, B_batchOffset, C_mat, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
      }
    } else {
      if (TransB == 'n') {
        if (TransA == 'n') {
          status = gemm_NoTransAB(accl_view, A_mat, aOffset, B_mat, bOffset, C_mat, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        } else {
          status = gemm_NoTransB(accl_view, A_mat, aOffset, B_mat, bOffset, C_mat, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
      } else if (TransA == 'n') {
        status = gemm_NoTransA(accl_view, A_mat, aOffset, B_mat, bOffset, C_mat, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
      } else {
        status = gemm_TransAB(accl_view, A_mat, aOffset, B_mat, bOffset, C_mat, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
      }
    }
  } else {
    if(batchSize > 0) {
      if (TransB == 'n') {
        if (TransA == 'n') {
          status = gemm_NoTransAB_rMajor(accl_view, A_mat, aOffset, A_batchOffset, B_mat, bOffset, B_batchOffset, C_mat, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
        } else {
          status = gemm_NoTransB_rMajor(accl_view, A_mat, aOffset, A_batchOffset, B_mat, bOffset, B_batchOffset, C_mat, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
        }
      } else if (TransA == 'n') {
        status = gemm_NoTransA_rMajor(accl_view, A_mat, aOffset, A_batchOffset, B_mat, bOffset, B_batchOffset, C_mat, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
      } else {
        status = gemm_TransAB_rMajor(accl_view, A_mat, aOffset, A_batchOffset, B_mat, bOffset, B_batchOffset, C_mat, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
      }
    } else {
      if (TransB == 'n') {
        if (TransA == 'n') {
          status = gemm_NoTransAB_rMajor(accl_view, A_mat, aOffset, B_mat, bOffset, C_mat, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        } else {
          status = gemm_NoTransB_rMajor(accl_view, A_mat, aOffset, B_mat, bOffset, C_mat, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
        }
      } else if (TransA == 'n') {
        status = gemm_NoTransA_rMajor(accl_view, A_mat, aOffset, B_mat, bOffset, C_mat, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
      } else {
        status = gemm_TransAB_rMajor(accl_view, A_mat, aOffset, B_mat, bOffset, C_mat, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
      }
    }
  }

  return status;
}

// Type 1 -  alpha = 0 Kernel

hcblasStatus gemm_alpha0_col(hc::accelerator_view accl_view,
		             float *A, long aOffset,
			     float *B, long bOffset,
			     float *C, long cOffset,
			     int M, int N, int K, int lda, int ldb, int ldc,
			     float alpha, float beta) {
#define GEMM_BLOCK 256
  hc::extent<2> grdExt(N, M * GEMM_BLOCK);
  hc::tiled_extent<2> t_ext = grdExt.tile(1, GEMM_BLOCK);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {
    int threadIdx = tidx.local[1];
    int Row = tidx.tile[0];
    int Col = tidx.tile[1];
    if (threadIdx == 0 && Col < M && Row < N) {
        long C_index = cOffset + Row * ldc + Col;
        C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
	if (alpha == 0 ) {
	  if ( beta == 0 ) {
            C[C_index] = 0.0;
	  }
	  else {
            C[C_index] *= beta;
	  }
        }
    }
 });
#undef GEMM_BLOCK
    return HCBLAS_SUCCEEDS;
}

// Type 2 - alpha = 0 kernel

hcblasStatus gemm_alpha0_col_batch(hc::accelerator_view accl_view,
                                   float *A, long aOffset, long A_batchOffset,
                                   float *B, long bOffset, long B_batchOffset,
                                   float *C, long cOffset, long C_batchOffset,
                                   int M, int N, int K, int lda, int ldb, int ldc,
                                   float alpha, float beta, int batchSize) {
#define GEMM_BLOCK 256
  hc::extent<3> grdExt(batchSize, N, M * GEMM_BLOCK);
  hc::tiled_extent<3> t_ext = grdExt.tile(1, 1, GEMM_BLOCK);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int elt = tidx.tile[0];
    int threadIdx = tidx.local[2];
    int Row = tidx.tile[1];
    int Col = tidx.tile[2];
    if (threadIdx == 0 && Col < M && Row < N) {
    long C_index = cOffset + C_batchOffset * elt + Row * ldc + Col;
    C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
    if (alpha == 0 ) {
      if ( beta == 0 ) {
        C[C_index] = 0.0;
      }
      else {
        C[C_index] *= beta;
      }
    }
}
});
#undef GEMM_BLOCK
    return HCBLAS_SUCCEEDS;
}

// Type 1 -  alpha = 0 Kernel

hcblasStatus gemm_alpha0_row(hc::accelerator_view accl_view,
                             float *A, long aOffset,
			     float *B, long bOffset,
			     float *C, long cOffset,
			     int M, int N, int K, int lda, int ldb, int ldc,
			     float alpha, float beta) {
#define GEMM_BLOCK 256
  hc::extent<2> grdExt(N, M * GEMM_BLOCK);
  hc::tiled_extent<2> t_ext = grdExt.tile(1, GEMM_BLOCK);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {
    int threadIdx = tidx.local[1];
    int Row = tidx.tile[0];
    int Col = tidx.tile[1];
    if (threadIdx == 0 && Col < M && Row < N) {
       long C_index = cOffset + Row + Col * ldc;
       C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
       if (alpha == 0 ) {
	 if ( beta == 0 ) {
           C[C_index] = 0.0;
	 }
	 else {
           C[C_index] *= beta;
         }
       }
    }
});
#undef GEMM_BLOCK
    return HCBLAS_SUCCEEDS;
}

// Type 2 - alpha = 0 kernel

hcblasStatus gemm_alpha0_row_batch(hc::accelerator_view accl_view,
                                   float *A, long aOffset, long A_batchOffset,
                                   float *B, long bOffset, long B_batchOffset,
                                   float *C, long cOffset, long C_batchOffset,
                                   int M, int N, int K, int lda, int ldb, int ldc,
                                   float alpha, float beta, int batchSize) {
#define GEMM_BLOCK 256
  hc::extent<3> grdExt(batchSize, N, M * GEMM_BLOCK);
  hc::tiled_extent<3> t_ext = grdExt.tile(1, 1, GEMM_BLOCK);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int elt = tidx.tile[0];
    int threadIdx = tidx.local[2];
    int Row = tidx.tile[1];
    int Col = tidx.tile[2];
    if (threadIdx == 0 && Col < M && Row < N) {
       long C_index = cOffset + C_batchOffset * elt + Row + Col * ldc;
       C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
       if (alpha == 0 ) {
         if ( beta == 0 ) {
           C[C_index] = 0.0;
         }
         else {
           C[C_index] *= beta;
         }
       }
    }
});
#undef GEMM_BLOCK
      return HCBLAS_SUCCEEDS;
}

// Sgemm Call Type I: Inputs and outputs are HCC device pointers
hcblasStatus  Hcblaslibrary :: hcblas_sgemm(hc::accelerator_view accl_view,
					    hcblasOrder order,
					    hcblasTranspose typeA,
					    hcblasTranspose typeB, const int M,
					    const int N, const int K, const float &alpha,
					    float *A, const long lda,
					    float *B, const long ldb,
					    const float &beta,
					    float *C, const long ldc,
					    const long aOffset, const long bOffset, const long cOffset) {
  int i, j;
  float temp;
  hcblasStatus status = HCBLAS_SUCCEEDS;

  // Quick return if possible
  if (A == NULL || B == NULL || C == NULL || !M || !N || !K) {
    return HCBLAS_INVALID;
  }

  // For alpha = 0
  if (alpha == 0) {
    if (order) {
       status = gemm_alpha0_col(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
    }
    else {
       status = gemm_alpha0_row(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
    }
   return status;
  }
  status = gemm_HC(accl_view, order, typeA, typeB, M, N, K, alpha, A,
                                aOffset, lda, B, bOffset, ldb, beta, C,
                                cOffset, ldc);
  return status;
}

/* SGEMM- Overloaded function with arguments related to batch processing */
hcblasStatus Hcblaslibrary :: hcblas_sgemm(hc::accelerator_view accl_view,
					   hcblasOrder order,
					   hcblasTranspose typeA,
					   hcblasTranspose typeB, const int M,
					   const int N, const int K, const float &alpha,
					   float *A, const long lda, const long A_batchOffset,
					   float *B, const long ldb, const long B_batchOffset,
					   const float &beta,
					   float *C, const long ldc, const long C_batchOffset,
					   const long aOffset, const long bOffset, const long cOffset, const int batchSize) {
  int i, j, k;
  float temp;
  hcblasStatus status = HCBLAS_SUCCEEDS;

  // Quick return if possible
  if (A == NULL || B == NULL || C == NULL || !M || !N || !K) {
    return HCBLAS_INVALID;
  }


  // For alpha = 0
  if (alpha == 0) {
    if (order) {
        status = gemm_alpha0_col_batch(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
    }
    else {
        status = gemm_alpha0_row_batch(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
    }
    return status;
  }

  status = gemm_HC(accl_view, order, typeA, typeB, M, N, K, alpha, A, aOffset, lda, B,
          bOffset, ldb, beta, C, cOffset, ldc, A_batchOffset, B_batchOffset, C_batchOffset, batchSize);
  return status;
}

