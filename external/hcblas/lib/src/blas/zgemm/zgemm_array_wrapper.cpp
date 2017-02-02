#include "zgemm_array_kernels.h"

hcblasStatus zgemm_alpha0_col(hc::accelerator_view accl_view,
                              double_2 *A, long aOffset,
                              double_2 *B, long bOffset,
                              double_2 *C, long cOffset,
                              int M, int N, int K, int lda, int ldb, int ldc,
                              double_2 alpha, double_2 beta) {
#define THREADS   16
#define TILE_DIM  16
  hc::extent<2> grdExt((N + (THREADS - 1)) & ~(THREADS - 1), (M + (THREADS - 1)) & ~(THREADS - 1));
  hc::tiled_extent<2> t_ext = grdExt.tile(THREADS, THREADS);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {
    int Row = tidx.tile[0] * TILE_DIM + tidx.local[0];
    int Col = tidx.tile[1] * TILE_DIM + tidx.local[1];
    float CReal = 0.0;
    float CImg = 0.0;
    if (Row < N && Col < M) {
      CReal = C[cOffset + (tidx.global[0] * ldc) + tidx.global[1]].x;
      CImg = C[cOffset + (tidx.global[0] * ldc) + tidx.global[1]].y;
      CReal = (isnan(CReal) || isinf(CReal)) ? 0 : CReal;
      CImg = (isnan(CImg) || isinf(CImg)) ? 0 : CImg;
      if( !alpha.x && !alpha.y) {
       if( !beta.x && !beta.y) {
        C[cOffset + (tidx.global[0] * ldc) + tidx.global[1]].x = 0.0;
        C[cOffset + (tidx.global[0] * ldc) + tidx.global[1]].y = 0.0;
       }
       else{
        C[cOffset + (tidx.global[0] * ldc) + tidx.global[1]].x = ((CReal * beta.x) - (CImg * beta.y));
        C[cOffset + (tidx.global[0] * ldc) + tidx.global[1]].y = ((CReal * beta.y) + (CImg * beta.x));
       }
      }
    }
  });
#undef THREADS
#undef TILE_DIM
  return HCBLAS_SUCCEEDS;
}

hcblasStatus zgemm_alpha0_colbatch(hc::accelerator_view accl_view,
                                   double_2 *A, long aOffset, long A_batchOffset,
                                   double_2 *B, long bOffset, long B_batchOffset,
                                   double_2 *C, long cOffset, long C_batchOffset,
                                   int M, int N, int K, int lda, int ldb, int ldc,
                                   double_2 alpha, double_2 beta, int batchSize) {
#define THREADS   16
#define TILE_DIM  16
  hc::extent<3> grdExt(batchSize, (N + (THREADS - 1)) & ~(THREADS - 1), (M + (THREADS - 1)) & ~(THREADS - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, THREADS, THREADS);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int elt = tidx.tile[0];
    int Row = tidx.tile[1] * TILE_DIM + tidx.local[1];
    int Col = tidx.tile[2] * TILE_DIM + tidx.local[2];
    float CReal = 0.0;
    float CImg = 0.0;
    if (Row < N && Col < M) {
      CReal = C[cOffset + C_batchOffset * elt + (tidx.global[1] * ldc) + tidx.global[2]].x;
      CImg = C[cOffset + C_batchOffset * elt + (tidx.global[1] * ldc) + tidx.global[2]].y;
      CReal = (isnan(CReal) || isinf(CReal)) ? 0 : CReal;
      CImg = (isnan(CImg) || isinf(CImg)) ? 0 : CImg;
      if( !alpha.x && !alpha.y) {
       if( !beta.x && !beta.y) {
        C[cOffset + C_batchOffset * elt + (tidx.global[1] * ldc) + tidx.global[2]].x = 0.0;
        C[cOffset + C_batchOffset * elt + (tidx.global[1] * ldc) + tidx.global[2]].y = 0.0;
       }
       else {
        C[cOffset + C_batchOffset * elt + (tidx.global[1] * ldc) + tidx.global[2]].x = ((CReal * beta.x) - (CImg * beta.y));
        C[cOffset + C_batchOffset * elt + (tidx.global[1] * ldc) + tidx.global[2]].y = ((CReal * beta.y) + (CImg * beta.x));
       }
     }
    }
  });
#undef THREADS
#undef TILE_DIM
  return HCBLAS_SUCCEEDS;
}

hcblasStatus zgemm_alpha0_row(hc::accelerator_view accl_view,
                              double_2 *A, long aOffset,
                              double_2 *B, long bOffset,
                              double_2 *C, long cOffset,
                              int M, int N, int K, int lda, int ldb, int ldc,
                              double_2 alpha, double_2 beta) {
#define THREADS   16
#define TILE_DIM  16
  hc::extent<2> grdExt((M + (THREADS - 1)) & ~(THREADS - 1), (N + (THREADS - 1)) & ~(THREADS - 1));
  hc::tiled_extent<2> t_ext = grdExt.tile(THREADS, THREADS);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {
    int Row = tidx.tile[1] * TILE_DIM + tidx.local[1];
    int Col = tidx.tile[0] * TILE_DIM + tidx.local[0];
    float CReal = 0.0;
    float CImg = 0.0;
    if (Row < N && Col < M) {
      CReal = C[cOffset + tidx.global[1] + (tidx.global[0] * ldc)].x;
      CImg = C[cOffset + tidx.global[1] + (tidx.global[0] * ldc)].y;
      CReal = (isnan(CReal) || isinf(CReal)) ? 0 : CReal;
      CImg = (isnan(CImg) || isinf(CImg)) ? 0 : CImg;
      if( !alpha.x && !alpha.y) {
       if( !beta.x && !beta.y) {
        C[cOffset + tidx.global[1] + (tidx.global[0] * ldc)].x = 0.0;
        C[cOffset + tidx.global[1] + (tidx.global[0] * ldc)].y = 0.0;
       }
       else {
        C[cOffset + tidx.global[1] + (tidx.global[0] * ldc)].x = ((CReal * beta.x) - (CImg * beta.y));
        C[cOffset + tidx.global[1] + (tidx.global[0] * ldc)].y = ((CReal * beta.y) + (CImg * beta.x));
       }
     }
   }
  });
#undef THREADS
#undef TILE_DIM
  return HCBLAS_SUCCEEDS;
}

hcblasStatus zgemm_alpha0_rowbatch(hc::accelerator_view accl_view,
                                   double_2 *A, long aOffset, long A_batchOffset,
                                   double_2 *B, long bOffset, long B_batchOffset,
                                   double_2 *C, long cOffset, long C_batchOffset,
                                   int M, int N, int K, int lda, int ldb, int ldc,
                                   double_2 alpha, double_2 beta, int batchSize) {
#define THREADS   16
#define TILE_DIM  16
  hc::extent<3> grdExt(batchSize, (M + (THREADS - 1)) & ~(THREADS - 1), (N + (THREADS - 1)) & ~(THREADS - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, THREADS, THREADS);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3> &tidx) __attribute__((hc, cpu)) {
    int elt = tidx.tile[0];
    int Row = tidx.tile[2] * TILE_DIM + tidx.local[2];
    int Col = tidx.tile[1] * TILE_DIM + tidx.local[1];
    float CReal = 0.0;
    float CImg = 0.0;
    if (Row < N && Col < M) {
      CReal = C[cOffset + C_batchOffset * elt + tidx.global[2] + (tidx.global[1] * ldc)].x;
      CImg = C[cOffset + C_batchOffset * elt + tidx.global[2] + (tidx.global[1] * ldc)].y;
      CReal = (isnan(CReal) || isinf(CReal)) ? 0 : CReal;
      CImg = (isnan(CImg) || isinf(CImg)) ? 0 : CImg;
      if( !alpha.x && !alpha.y) {
       if( !beta.x && !beta.y) {
        C[cOffset + C_batchOffset * elt + tidx.global[2] + (tidx.global[1] * ldc)].x = 0.0;
        C[cOffset + C_batchOffset * elt + tidx.global[2] + (tidx.global[1] * ldc)].y = 0.0;
       }
       else {
        C[cOffset + C_batchOffset * elt + tidx.global[2] + (tidx.global[1] * ldc)].x = ((CReal * beta.x) - (CImg * beta.y));
        C[cOffset + C_batchOffset * elt + tidx.global[2] + (tidx.global[1] * ldc)].y = ((CReal * beta.y) + (CImg * beta.x));
       }
      }
    }
  });
#undef THREADS
#undef TILE_DIM
  return HCBLAS_SUCCEEDS;
}

// ZGEMM Call Type I: Inputs and outputs are C++ HC float array containers
hcblasStatus Hcblaslibrary :: hcblas_zgemm(hc::accelerator_view accl_view,
				           hcblasOrder order, hcblasTranspose typeA,
					   hcblasTranspose typeB, const int M,
					   const int N, const int K,
					   const double_2 &Calpha,
					   double_2 *Acmplx, long aOffset, long lda,
					   double_2 *Bcmplx, long bOffset, long ldb,
					   const double_2 &Cbeta,
					   double_2 *Ccmplx, long cOffset, long ldc) {
  int i, j;
  hcblasStatus status = HCBLAS_SUCCEEDS;
  float tempReal = 0.0, tempImg = 0.0;
  // Quick return if possible
  if (Acmplx == NULL || Bcmplx == NULL || Ccmplx == NULL || !M || !N || !K) {
    return HCBLAS_INVALID;
  }

  if (!Calpha.x  && !Calpha.y) {
    if(order)
      status = zgemm_alpha0_col(accl_view, Acmplx, aOffset, Bcmplx, bOffset, Ccmplx, cOffset, M, N, K, lda, ldb, ldc, Calpha, Cbeta);
    else
      status = zgemm_alpha0_row(accl_view, Acmplx, aOffset, Bcmplx, bOffset, Ccmplx, cOffset, M, N, K, lda, ldb, ldc, Calpha, Cbeta);
    return status;
  }  

  if(order) {
    if (typeB == NoTrans) {
      if (typeA == NoTrans) {
        status = zgemm_NoTransAB(accl_view, Acmplx, aOffset, Bcmplx, bOffset, Ccmplx, cOffset, M, N, K, lda, ldb, ldc, Calpha, Cbeta);
      } else {
        status = zgemm_NoTransB(accl_view, Acmplx, aOffset, Bcmplx, bOffset, Ccmplx, cOffset, M, N, K, lda, ldb, ldc, Calpha, Cbeta);
      }
    } else if (typeA == NoTrans) {
      status = zgemm_NoTransA(accl_view, Acmplx, aOffset, Bcmplx, bOffset, Ccmplx, cOffset, M, N, K, lda, ldb, ldc, Calpha, Cbeta);
    } else {
      status = zgemm_TransAB(accl_view, Acmplx, aOffset, Bcmplx, bOffset, Ccmplx, cOffset, M, N, K, lda, ldb, ldc, Calpha, Cbeta);
    }
  } else {
    if (typeB == NoTrans) {
      if (typeA == NoTrans) {
        status = zgemm_NoTransAB_rMajor(accl_view, Acmplx, aOffset, Bcmplx, bOffset, Ccmplx, cOffset, M, N, K, lda, ldb, ldc, Calpha, Cbeta);
      } else {
        status = zgemm_NoTransB_rMajor(accl_view, Acmplx, aOffset, Bcmplx, bOffset, Ccmplx, cOffset, M, N, K, lda, ldb, ldc, Calpha, Cbeta);
      }
    } else if (typeA == NoTrans) {
      status = zgemm_NoTransA_rMajor(accl_view, Acmplx, aOffset, Bcmplx, bOffset, Ccmplx, cOffset, M, N, K, lda, ldb, ldc, Calpha, Cbeta);
    } else {
      status = zgemm_TransAB_rMajor(accl_view, Acmplx, aOffset, Bcmplx, bOffset, Ccmplx, cOffset, M, N, K, lda, ldb, ldc, Calpha, Cbeta);
    }
  }

  return status;
}

/* ZGEMM Call Type II - Overloaded function with arguments related to batch processing */
hcblasStatus Hcblaslibrary :: hcblas_zgemm(hc::accelerator_view accl_view,
					   hcblasOrder order, hcblasTranspose typeA,
					   hcblasTranspose typeB, const int M,
					   const int N, const int K,
					   const double_2 &Calpha,
					   double_2 *Acmplx,
					   const long aOffset, const long A_batchOffset, const long lda,
					   double_2 *Bcmplx,
					   const long bOffset, const long B_batchOffset, const long ldb,
					   const double_2 &Cbeta,
					   double_2 *Ccmplx,
					   const long cOffset, const long C_batchOffset, const long ldc, const int batchSize) {
  int i, j, k;
  hcblasStatus status = HCBLAS_SUCCEEDS;
  float tempReal = 0.0, tempImg = 0.0;
  // Quick return if possible
  if (Acmplx == NULL || Bcmplx == NULL || Ccmplx == NULL || !M || !N || !K) {
    return HCBLAS_INVALID;
  }

  if (!Calpha.x  && !Calpha.y) {
    if(order)
      status = zgemm_alpha0_colbatch(accl_view, Acmplx, aOffset, A_batchOffset, Bcmplx, bOffset, B_batchOffset, Ccmplx, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, Calpha, Cbeta, batchSize);
    else
      status = zgemm_alpha0_rowbatch(accl_view, Acmplx, aOffset, A_batchOffset, Bcmplx, bOffset, B_batchOffset, Ccmplx, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, Calpha, Cbeta, batchSize);
    return status;
  }
 
  if(order) {
    if (typeB == NoTrans) {
      if (typeA == NoTrans) {
        status = zgemm_NoTransAB(accl_view, Acmplx, aOffset, A_batchOffset, Bcmplx, bOffset, B_batchOffset, Ccmplx, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, Calpha, Cbeta, batchSize);
      } else {
        status = zgemm_NoTransB(accl_view, Acmplx, aOffset, A_batchOffset, Bcmplx, bOffset, B_batchOffset, Ccmplx, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, Calpha, Cbeta, batchSize);
      }
    } else if (typeA == NoTrans) {
      status = zgemm_NoTransA(accl_view, Acmplx, aOffset, A_batchOffset, Bcmplx, bOffset, B_batchOffset, Ccmplx, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, Calpha, Cbeta, batchSize);
    } else {
      status = zgemm_TransAB(accl_view, Acmplx, aOffset, A_batchOffset, Bcmplx, bOffset, B_batchOffset, Ccmplx, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, Calpha, Cbeta, batchSize);
    }
  } else {
    if (typeB == NoTrans) {
      if (typeA == NoTrans) {
        status = zgemm_NoTransAB_rMajor(accl_view, Acmplx, aOffset, A_batchOffset, Bcmplx, bOffset, B_batchOffset, Ccmplx, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, Calpha, Cbeta, batchSize);
      } else {
        status = zgemm_NoTransB_rMajor(accl_view, Acmplx, aOffset, A_batchOffset, Bcmplx, bOffset, B_batchOffset, Ccmplx, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, Calpha, Cbeta, batchSize);
      }
    } else if (typeA == NoTrans) {
      status = zgemm_NoTransA_rMajor(accl_view, Acmplx, aOffset, A_batchOffset, Bcmplx, bOffset, B_batchOffset, Ccmplx, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, Calpha, Cbeta, batchSize);
    } else {
      status = zgemm_TransAB_rMajor(accl_view, Acmplx, aOffset, A_batchOffset, Bcmplx, bOffset, B_batchOffset, Ccmplx, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, Calpha, Cbeta, batchSize);
    }
  }

  return status;
}


