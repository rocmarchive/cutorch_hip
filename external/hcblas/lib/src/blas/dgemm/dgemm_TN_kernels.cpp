#include "dgemm_array_kernels.h"
#include <cmath>
#include "hc_math.hpp"

hcblasStatus gemm_NoTransB_MICRO_NBK_M064_N064_K064_TS16XMTS4(hc::accelerator_view accl_view,
					       const double *A, long aOffset,
					       const double *B, long bOffset,
					       double *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       double alpha, double beta) {
#undef TILESIZE
#undef MICROTILESIZE
#define TILESIZE 16
#define MICROTILESIZE 4
  int M_ = M >> 2;
  int N_ = N >> 2;
  int N_R = (N_ + (TILESIZE - 1)) & ~(TILESIZE - 1);
  int M_R = (M_ + (TILESIZE - 1)) & ~(TILESIZE - 1);
  int K_R = (K + (TILESIZE - 1)) & ~(TILESIZE - 1);
  hc::extent<2> grdExt(N_R, M_R);
  hc::tiled_extent<2> t_ext = grdExt.tile(TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {
    int shiftTS = hc::fast_math::log2(TILESIZE);
    double rC[MICROTILESIZE][MICROTILESIZE] = {{(double)0}};
    double rA[1][MICROTILESIZE];
    double rB[1][MICROTILESIZE];
    tile_static double lA[TOTMICROTILEPROD + TILESIZE];
    tile_static double lB[TOTMICROTILEPROD + TILESIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int block_k = 0;
    int lIndex = (idx * BANKMICROTILESIZE) + idy;
    int AIndex = aOffset + ((gidx * MICROTILEPROD) + idy) * lda + idx;
    int BIndex = bOffset + ((gidy * MICROTILEPROD) + idy) * ldb + idx;
    long CIndex = cOffset + (gidx * MICROTILEPROD) + idx + (((gidy * MICROTILEPROD) + idy) * ldc);
    long CinitOffset = 0;

    do {
      tidx.barrier.wait();

      lB[lIndex + 0 * TILESIZE] = B[BIndex + block_k * TILESIZE + 0 * TILESIZE * ldb];
      lB[lIndex + 1 * TILESIZE] = B[BIndex + block_k * TILESIZE + 1 * TILESIZE * ldb];
      lB[lIndex + 2 * TILESIZE] = B[BIndex + block_k * TILESIZE + 2 * TILESIZE * ldb];
      lB[lIndex + 3 * TILESIZE] = B[BIndex + block_k * TILESIZE + 3 * TILESIZE * ldb];
      lA[lIndex + 0 * TILESIZE] = A[AIndex + block_k * TILESIZE + 0 * TILESIZE * lda];
      lA[lIndex + 1 * TILESIZE] = A[AIndex + block_k * TILESIZE + 1 * TILESIZE * lda];
      lA[lIndex + 2 * TILESIZE] = A[AIndex + block_k * TILESIZE + 2 * TILESIZE * lda];
      lA[lIndex + 3 * TILESIZE] = A[AIndex + block_k * TILESIZE + 3 * TILESIZE * lda];

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < TILESIZE; iter+=8) {
          M4x4
          M4x4
          M4x4
          M4x4
          M4x4
          M4x4
          M4x4
          M4x4
      }

      tidx.barrier.wait();
    } while (++block_k < (K_R / TILESIZE));

    C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[0][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[0][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
    CinitOffset += TILESIZE;
    C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[1][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[1][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
    CinitOffset += TILESIZE;
    C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[2][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[2][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[2][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[2][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
    CinitOffset += TILESIZE;
    C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[3][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[3][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[3][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[3][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
  });
#undef TILESIZE
#undef MICROTILESIZE
  return HCBLAS_SUCCEEDS;
}

hcblasStatus gemm_NoTransB_MICRO_NBK_M096_N096_K096_TS16XMTS6(hc::accelerator_view accl_view,
					       const double *A, long aOffset,
					       const double *B, long bOffset,
					       double *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       double alpha, double beta) {
#undef TILESIZE
#undef MICROTILESIZE
#define TILESIZE 16
#define MICROTILESIZE 6
  int M_ = M / 6;
  int N_ = N / 6;
  int N_R = (N_ + (TILESIZE - 1)) & ~(TILESIZE - 1);
  int M_R = (M_ + (TILESIZE - 1)) & ~(TILESIZE - 1);
  int K_R = (K + (TILESIZE - 1)) & ~(TILESIZE - 1);
  hc::extent<2> grdExt(N_R, M_R);
  hc::tiled_extent<2> t_ext = grdExt.tile(TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {
    int shiftTS = hc::fast_math::log2(TILESIZE);
    double rC[MICROTILESIZE][MICROTILESIZE] = {{(double)0}};
    double rA[1][MICROTILESIZE];
    double rB[1][MICROTILESIZE];
    tile_static double lA[TOTMICROTILEPROD + TILESIZE];
    tile_static double lB[TOTMICROTILEPROD + TILESIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int block_k = 0;
    int lIndex = (idx * BANKMICROTILESIZE) + idy;
    int AIndex = aOffset + ((gidx * MICROTILEPROD) + idy) * lda + idx;
    int BIndex = bOffset + ((gidy * MICROTILEPROD) + idy) * ldb + idx;
    long CIndex = cOffset + (gidx * MICROTILEPROD) + idx + (((gidy * MICROTILEPROD) + idy) * ldc);
    long CinitOffset = 0;

    do {
      tidx.barrier.wait();

      lB[lIndex + 0 * TILESIZE] = B[BIndex + block_k * TILESIZE + 0 * TILESIZE * ldb];
      lB[lIndex + 1 * TILESIZE] = B[BIndex + block_k * TILESIZE + 1 * TILESIZE * ldb];
      lB[lIndex + 2 * TILESIZE] = B[BIndex + block_k * TILESIZE + 2 * TILESIZE * ldb];
      lB[lIndex + 3 * TILESIZE] = B[BIndex + block_k * TILESIZE + 3 * TILESIZE * ldb];
      lB[lIndex + 4 * TILESIZE] = B[BIndex + block_k * TILESIZE + 4 * TILESIZE * ldb];
      lB[lIndex + 5 * TILESIZE] = B[BIndex + block_k * TILESIZE + 5 * TILESIZE * ldb];
      lA[lIndex + 0 * TILESIZE] = A[AIndex + block_k * TILESIZE + 0 * TILESIZE * lda];
      lA[lIndex + 1 * TILESIZE] = A[AIndex + block_k * TILESIZE + 1 * TILESIZE * lda];
      lA[lIndex + 2 * TILESIZE] = A[AIndex + block_k * TILESIZE + 2 * TILESIZE * lda];
      lA[lIndex + 3 * TILESIZE] = A[AIndex + block_k * TILESIZE + 3 * TILESIZE * lda];
      lA[lIndex + 4 * TILESIZE] = A[AIndex + block_k * TILESIZE + 4 * TILESIZE * lda];
      lA[lIndex + 5 * TILESIZE] = A[AIndex + block_k * TILESIZE + 5 * TILESIZE * lda];

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < TILESIZE; iter+=8) {
          M6x6
          M6x6
          M6x6
          M6x6
          M6x6
          M6x6
          M6x6
          M6x6
      }

      tidx.barrier.wait();
    } while (++block_k < (K_R / TILESIZE));

    C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[0][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[0][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 4 * TILESIZE * ldc] = alpha*rC[0][4] + beta * C[CIndex + CinitOffset + 4 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 5 * TILESIZE * ldc] = alpha*rC[0][5] + beta * C[CIndex + CinitOffset + 5 * TILESIZE * ldc] ;
    CinitOffset += TILESIZE;
    C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[1][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[1][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 4 * TILESIZE * ldc] = alpha*rC[1][4] + beta * C[CIndex + CinitOffset + 4 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 5 * TILESIZE * ldc] = alpha*rC[1][5] + beta * C[CIndex + CinitOffset + 5 * TILESIZE * ldc] ;
    CinitOffset += TILESIZE;
    C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[2][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[2][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[2][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[2][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 4 * TILESIZE * ldc] = alpha*rC[2][4] + beta * C[CIndex + CinitOffset + 4 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 5 * TILESIZE * ldc] = alpha*rC[2][5] + beta * C[CIndex + CinitOffset + 5 * TILESIZE * ldc] ;
    CinitOffset += TILESIZE;
    C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[3][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[3][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[3][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[3][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 4 * TILESIZE * ldc] = alpha*rC[3][4] + beta * C[CIndex + CinitOffset + 4 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 5 * TILESIZE * ldc] = alpha*rC[3][5] + beta * C[CIndex + CinitOffset + 5 * TILESIZE * ldc] ;
    CinitOffset += TILESIZE;
    C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[4][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[4][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[4][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[4][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 4 * TILESIZE * ldc] = alpha*rC[4][4] + beta * C[CIndex + CinitOffset + 4 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 5 * TILESIZE * ldc] = alpha*rC[4][5] + beta * C[CIndex + CinitOffset + 5 * TILESIZE * ldc] ;
    CinitOffset += TILESIZE;
    C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[5][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[5][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[5][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[5][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 4 * TILESIZE * ldc] = alpha*rC[5][4] + beta * C[CIndex + CinitOffset + 4 * TILESIZE * ldc] ;
    C[CIndex + CinitOffset + 5 * TILESIZE * ldc] = alpha*rC[5][5] + beta * C[CIndex + CinitOffset + 5 * TILESIZE * ldc] ;
  });
#undef TILESIZE
#undef MICROTILESIZE
  return HCBLAS_SUCCEEDS;
}
hcblasStatus gemm_NoTransB_MICRO_NBK_M_N_K_TS16XMTS2(hc::accelerator_view accl_view,
					       const double *A, long aOffset,
					       const double *B, long bOffset,
					       double *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       double alpha, double beta) {
#undef TILESIZE
#undef MICROTILESIZE
#define TILESIZE 16
#define MICROTILESIZE 2
  int M_ = (M-1) / MICROTILESIZE + 1;
  int N_ = (N-1) / MICROTILESIZE + 1;
  int N_R = (N_ + (TILESIZE - 1)) & ~(TILESIZE - 1);
  int M_R = (M_ + (TILESIZE - 1)) & ~(TILESIZE - 1);
  int K_R = (K + (TILESIZE - 1)) & ~(TILESIZE - 1);
  hc::extent<2> grdExt(N_R, M_R);
  hc::tiled_extent<2> t_ext = grdExt.tile(TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {
    int shiftTS = hc::fast_math::log2(TILESIZE);
    double rC[MICROTILESIZE][MICROTILESIZE] = {{(double)0}};
    double rA[1][MICROTILESIZE];
    double rB[1][MICROTILESIZE];
    tile_static double lA[TOTMICROTILEPROD + TILESIZE];
    tile_static double lB[TOTMICROTILEPROD + TILESIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int block_k = 0;
    int lIndex = (idx * BANKMICROTILESIZE) + idy;
    int AIndex = aOffset + ((gidx * MICROTILEPROD) + idy) * lda + idx;
    int BIndex = bOffset + ((gidy * MICROTILEPROD) + idy) * ldb + idx;
    long CIndex = cOffset + (gidx * MICROTILEPROD) + idx + (((gidy * MICROTILEPROD) + idy) * ldc);
    int N_block = N_R/TILESIZE;
    int M_block = M_R/TILESIZE;
    int K_block = K_R/TILESIZE;
    long CinitOffset = 0;

    do {
      tidx.barrier.wait();

      if (gidx == M_block-1 || gidy == N_block-1 || block_k == K_block-1)
      {
          for(int sec = 0; sec < MICROTILESIZE; ++sec) {
            int secVal = sec << shiftTS;

            if((gidy*MICROTILEPROD + idy + secVal) < N && (block_k * TILESIZE + idx) < K) {
              lB[ lIndex + secVal] = B[BIndex + block_k * TILESIZE + secVal * ldb]; 
            } else {
              lB[lIndex + secVal] = 0;
            }

            if(((gidx*MICROTILEPROD) + idy + secVal) < M && (block_k * TILESIZE + idx) < K) {
              lA[lIndex + secVal] = A[AIndex + block_k * TILESIZE + secVal * lda];
            } else {
              lA[lIndex + secVal] = 0;
            }
          }
      }
      else
      {
          lB[lIndex] = B[BIndex + block_k * TILESIZE];
          lB[lIndex + TILESIZE] = B[BIndex + block_k * TILESIZE + TILESIZE * ldb];
          lA[lIndex] = A[AIndex + block_k * TILESIZE];
          lA[lIndex + TILESIZE] = A[AIndex + block_k * TILESIZE + TILESIZE * lda];
      }

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < TILESIZE; iter+=8) {
          M2x2
          M2x2
          M2x2
          M2x2
          M2x2
          M2x2
          M2x2
          M2x2
      }

      tidx.barrier.wait();
    } while (++block_k < (K_R / TILESIZE));

    if(gidx == M_block-1 || gidy == N_block-1)
    {
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 0 * TILESIZE) < N)
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 1 * TILESIZE) < N)
        C[CIndex + CinitOffset + TILESIZE * ldc] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + TILESIZE * ldc] ;
        CinitOffset += TILESIZE;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 0 * TILESIZE) < N)
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 1 * TILESIZE) < N)
        C[CIndex + CinitOffset + TILESIZE * ldc] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + TILESIZE * ldc] ;
    }
    else
    {
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        C[CIndex + CinitOffset + TILESIZE * ldc] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + TILESIZE * ldc] ;
        CinitOffset += TILESIZE;
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        C[CIndex + CinitOffset + TILESIZE * ldc] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + TILESIZE * ldc] ;
    }
  });
#undef TILESIZE
#undef MICROTILESIZE
  return HCBLAS_SUCCEEDS;
}

hcblasStatus gemm_NoTransB_MICRO_NBK_M_N_K_TS16XMTS4(hc::accelerator_view accl_view,
					       const double *A, long aOffset,
					       const double *B, long bOffset,
					       double *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       double alpha, double beta) {
#undef TILESIZE
#undef MICROTILESIZE
#define TILESIZE 16
#define MICROTILESIZE 4
  int M_ = (M-1) / MICROTILESIZE + 1;
  int N_ = (N-1) / MICROTILESIZE + 1;
  int N_R = (N_ + (TILESIZE - 1)) & ~(TILESIZE - 1);
  int M_R = (M_ + (TILESIZE - 1)) & ~(TILESIZE - 1);
  int K_R = (K + (TILESIZE - 1)) & ~(TILESIZE - 1);
  hc::extent<2> grdExt(N_R, M_R);
  hc::tiled_extent<2> t_ext = grdExt.tile(TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {
    int shiftTS = hc::fast_math::log2(TILESIZE);
    double rC[MICROTILESIZE][MICROTILESIZE] = {{(double)0}};
    double rA[1][MICROTILESIZE];
    double rB[1][MICROTILESIZE];
    tile_static double lA[TOTMICROTILEPROD + TILESIZE];
    tile_static double lB[TOTMICROTILEPROD + TILESIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int block_k = 0;
    int lIndex = (idx * BANKMICROTILESIZE) + idy;
    int AIndex = aOffset + ((gidx * MICROTILEPROD) + idy) * lda + idx;
    int BIndex = bOffset + ((gidy * MICROTILEPROD) + idy) * ldb + idx;
    long CIndex = cOffset + (gidx * MICROTILEPROD) + idx + (((gidy * MICROTILEPROD) + idy) * ldc);
    int N_block = N_R/TILESIZE;
    int M_block = M_R/TILESIZE;
    int K_block = K_R/TILESIZE;
    long CinitOffset = 0;

    do {
      tidx.barrier.wait();

      if (gidx == M_block-1 || gidy == N_block-1 || block_k == K_block-1)
      {
          for(int sec = 0; sec < MICROTILESIZE; ++sec) {
            int secVal = sec << shiftTS;

            if((gidy*MICROTILEPROD + idy + secVal) < N && (block_k * TILESIZE + idx) < K) {
              lB[ lIndex + secVal] = B[BIndex + block_k * TILESIZE + secVal * ldb]; 
            } else {
              lB[lIndex + secVal] = 0;
            }

            if(((gidx*MICROTILEPROD) + idy + secVal) < M && (block_k * TILESIZE + idx) < K) {
              lA[lIndex + secVal] = A[AIndex + block_k * TILESIZE + secVal * lda];
            } else {
              lA[lIndex + secVal] = 0;
            }
          }
      }
      else
      {
          lB[lIndex + 0 * TILESIZE] = B[BIndex + block_k * TILESIZE + 0 * TILESIZE * ldb];
          lB[lIndex + 1 * TILESIZE] = B[BIndex + block_k * TILESIZE + 1 * TILESIZE * ldb];
          lB[lIndex + 2 * TILESIZE] = B[BIndex + block_k * TILESIZE + 2 * TILESIZE * ldb];
          lB[lIndex + 3 * TILESIZE] = B[BIndex + block_k * TILESIZE + 3 * TILESIZE * ldb];
          lA[lIndex + 0 * TILESIZE] = A[AIndex + block_k * TILESIZE + 0 * TILESIZE * lda];
          lA[lIndex + 1 * TILESIZE] = A[AIndex + block_k * TILESIZE + 1 * TILESIZE * lda];
          lA[lIndex + 2 * TILESIZE] = A[AIndex + block_k * TILESIZE + 2 * TILESIZE * lda];
          lA[lIndex + 3 * TILESIZE] = A[AIndex + block_k * TILESIZE + 3 * TILESIZE * lda];
      }

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < TILESIZE; iter+=8) {
          M4x4
          M4x4
          M4x4
          M4x4
          M4x4
          M4x4
          M4x4
          M4x4
      }

      tidx.barrier.wait();
    } while (++block_k < (K_R / TILESIZE));

    if(gidx == M_block-1 || gidy == N_block-1)
    {
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 0 * TILESIZE) < N)
        C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 1 * TILESIZE) < N)
        C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 2 * TILESIZE) < N)
        C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[0][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 3 * TILESIZE) < N)
        C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[0][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
        CinitOffset += TILESIZE;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 0 * TILESIZE) < N)
        C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 1 * TILESIZE) < N)
        C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 2 * TILESIZE) < N)
        C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[1][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 3 * TILESIZE) < N)
        C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[1][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
        CinitOffset += TILESIZE;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 0 * TILESIZE) < N)
        C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[2][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 1 * TILESIZE) < N)
        C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[2][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 2 * TILESIZE) < N)
        C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[2][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 3 * TILESIZE) < N)
        C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[2][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
        CinitOffset += TILESIZE;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 0 * TILESIZE) < N)
        C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[3][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 1 * TILESIZE) < N)
        C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[3][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 2 * TILESIZE) < N)
        C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[3][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 3 * TILESIZE) < N)
        C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[3][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
    }
    else
    {
        C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[0][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[0][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
        CinitOffset += TILESIZE;
        C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[1][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[1][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
        CinitOffset += TILESIZE;
        C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[2][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[2][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[2][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[2][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
        CinitOffset += TILESIZE;
        C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[3][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[3][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[3][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[3][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
    }
  });
#undef TILESIZE
#undef MICROTILESIZE
  return HCBLAS_SUCCEEDS;
}

hcblasStatus gemm_NoTransB_MICRO_NBK_M_N_K_TS16XMTS6(hc::accelerator_view accl_view,
					       const double *A, long aOffset,
					       const double *B, long bOffset,
					       double *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       double alpha, double beta) {
#undef TILESIZE
#undef MICROTILESIZE
#define TILESIZE 16
#define MICROTILESIZE 6
  int M_ = (M-1) / MICROTILESIZE + 1;
  int N_ = (N-1) / MICROTILESIZE + 1;
  int N_R = (N_ + (TILESIZE - 1)) & ~(TILESIZE - 1);
  int M_R = (M_ + (TILESIZE - 1)) & ~(TILESIZE - 1);
  int K_R = (K + (TILESIZE - 1)) & ~(TILESIZE - 1);
  hc::extent<2> grdExt(N_R, M_R);
  hc::tiled_extent<2> t_ext = grdExt.tile(TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {
    int shiftTS = hc::fast_math::log2(TILESIZE);
    double rC[MICROTILESIZE][MICROTILESIZE] = {{(double)0}};
    double rA[1][MICROTILESIZE];
    double rB[1][MICROTILESIZE];
    tile_static double lA[TOTMICROTILEPROD + TILESIZE];
    tile_static double lB[TOTMICROTILEPROD + TILESIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int block_k = 0;
    int lIndex = (idx * BANKMICROTILESIZE) + idy;
    int AIndex = aOffset + ((gidx * MICROTILEPROD) + idy) * lda + idx;
    int BIndex = bOffset + ((gidy * MICROTILEPROD) + idy) * ldb + idx;
    long CIndex = cOffset + (gidx * MICROTILEPROD) + idx + (((gidy * MICROTILEPROD) + idy) * ldc);
    int N_block = N_R/TILESIZE;
    int M_block = M_R/TILESIZE;
    int K_block = K_R/TILESIZE;
    long CinitOffset = 0;

    do {
      tidx.barrier.wait();

      if (gidx == M_block-1 || gidy == N_block-1 || block_k == K_block-1)
      {
          for(int sec = 0; sec < MICROTILESIZE; ++sec) {
            int secVal = sec << shiftTS;

            if((gidy*MICROTILEPROD + idy + secVal) < N && (block_k * TILESIZE + idx) < K) {
              lB[ lIndex + secVal] = B[BIndex + block_k * TILESIZE + secVal * ldb]; 
            } else {
              lB[lIndex + secVal] = 0;
            }

            if(((gidx*MICROTILEPROD) + idy + secVal) < M && (block_k * TILESIZE + idx) < K) {
              lA[lIndex + secVal] = A[AIndex + block_k * TILESIZE + secVal * lda];
            } else {
              lA[lIndex + secVal] = 0;
            }
          }
      }
      else
      {
          lB[lIndex + 0 * TILESIZE] = B[BIndex + block_k * TILESIZE + 0 * TILESIZE * ldb];
          lB[lIndex + 1 * TILESIZE] = B[BIndex + block_k * TILESIZE + 1 * TILESIZE * ldb];
          lB[lIndex + 2 * TILESIZE] = B[BIndex + block_k * TILESIZE + 2 * TILESIZE * ldb];
          lB[lIndex + 3 * TILESIZE] = B[BIndex + block_k * TILESIZE + 3 * TILESIZE * ldb];
          lB[lIndex + 4 * TILESIZE] = B[BIndex + block_k * TILESIZE + 4 * TILESIZE * ldb];
          lB[lIndex + 5 * TILESIZE] = B[BIndex + block_k * TILESIZE + 5 * TILESIZE * ldb];
          lA[lIndex + 0 * TILESIZE] = A[AIndex + block_k * TILESIZE + 0 * TILESIZE * lda];
          lA[lIndex + 1 * TILESIZE] = A[AIndex + block_k * TILESIZE + 1 * TILESIZE * lda];
          lA[lIndex + 2 * TILESIZE] = A[AIndex + block_k * TILESIZE + 2 * TILESIZE * lda];
          lA[lIndex + 3 * TILESIZE] = A[AIndex + block_k * TILESIZE + 3 * TILESIZE * lda];
          lA[lIndex + 4 * TILESIZE] = A[AIndex + block_k * TILESIZE + 4 * TILESIZE * lda];
          lA[lIndex + 5 * TILESIZE] = A[AIndex + block_k * TILESIZE + 5 * TILESIZE * lda];
      }

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < TILESIZE; iter+=8) {
          M6x6
          M6x6
          M6x6
          M6x6
          M6x6
          M6x6
          M6x6
          M6x6
      }

      tidx.barrier.wait();
    } while (++block_k < (K_R / TILESIZE));

    if(gidx == M_block-1 || gidy == N_block-1)
    {
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 0 * TILESIZE) < N)
        C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 1 * TILESIZE) < N)
        C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 2 * TILESIZE) < N)
        C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[0][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 3 * TILESIZE) < N)
        C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[0][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 4 * TILESIZE) < N)
        C[CIndex + CinitOffset + 4 * TILESIZE * ldc] = alpha*rC[0][4] + beta * C[CIndex + CinitOffset + 4 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 5 * TILESIZE) < N)
        C[CIndex + CinitOffset + 5 * TILESIZE * ldc] = alpha*rC[0][5] + beta * C[CIndex + CinitOffset + 5 * TILESIZE * ldc] ;
        CinitOffset += TILESIZE;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 0 * TILESIZE) < N)
        C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 1 * TILESIZE) < N)
        C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 2 * TILESIZE) < N)
        C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[1][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 3 * TILESIZE) < N)
        C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[1][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 4 * TILESIZE) < N)
        C[CIndex + CinitOffset + 4 * TILESIZE * ldc] = alpha*rC[1][4] + beta * C[CIndex + CinitOffset + 4 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 5 * TILESIZE) < N)
        C[CIndex + CinitOffset + 5 * TILESIZE * ldc] = alpha*rC[1][5] + beta * C[CIndex + CinitOffset + 5 * TILESIZE * ldc] ;
        CinitOffset += TILESIZE;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 0 * TILESIZE) < N)
        C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[2][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 1 * TILESIZE) < N)
        C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[2][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 2 * TILESIZE) < N)
        C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[2][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 3 * TILESIZE) < N)
        C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[2][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 4 * TILESIZE) < N)
        C[CIndex + CinitOffset + 4 * TILESIZE * ldc] = alpha*rC[2][4] + beta * C[CIndex + CinitOffset + 4 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 5 * TILESIZE) < N)
        C[CIndex + CinitOffset + 5 * TILESIZE * ldc] = alpha*rC[2][5] + beta * C[CIndex + CinitOffset + 5 * TILESIZE * ldc] ;
        CinitOffset += TILESIZE;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 0 * TILESIZE) < N)
        C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[3][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 1 * TILESIZE) < N)
        C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[3][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 2 * TILESIZE) < N)
        C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[3][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 3 * TILESIZE) < N)
        C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[3][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 4 * TILESIZE) < N)
        C[CIndex + CinitOffset + 4 * TILESIZE * ldc] = alpha*rC[3][4] + beta * C[CIndex + CinitOffset + 4 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 5 * TILESIZE) < N)
        C[CIndex + CinitOffset + 5 * TILESIZE * ldc] = alpha*rC[3][5] + beta * C[CIndex + CinitOffset + 5 * TILESIZE * ldc] ;
        CinitOffset += TILESIZE;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 0 * TILESIZE) < N)
        C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[4][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 1 * TILESIZE) < N)
        C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[4][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 2 * TILESIZE) < N)
        C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[4][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 3 * TILESIZE) < N)
        C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[4][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 4 * TILESIZE) < N)
        C[CIndex + CinitOffset + 4 * TILESIZE * ldc] = alpha*rC[4][4] + beta * C[CIndex + CinitOffset + 4 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 5 * TILESIZE) < N)
        C[CIndex + CinitOffset + 5 * TILESIZE * ldc] = alpha*rC[4][5] + beta * C[CIndex + CinitOffset + 5 * TILESIZE * ldc] ;
        CinitOffset += TILESIZE;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 0 * TILESIZE) < N)
        C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[5][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 1 * TILESIZE) < N)
        C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[5][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 2 * TILESIZE) < N)
        C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[5][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 3 * TILESIZE) < N)
        C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[5][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 4 * TILESIZE) < N)
        C[CIndex + CinitOffset + 4 * TILESIZE * ldc] = alpha*rC[5][4] + beta * C[CIndex + CinitOffset + 4 * TILESIZE * ldc] ;
        if((gidx * MICROTILEPROD + idx + CinitOffset)  < M && (gidy * MICROTILEPROD + idy + 5 * TILESIZE) < N)
        C[CIndex + CinitOffset + 5 * TILESIZE * ldc] = alpha*rC[5][5] + beta * C[CIndex + CinitOffset + 5 * TILESIZE * ldc] ;
    }
    else
    {
        C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[0][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[0][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 4 * TILESIZE * ldc] = alpha*rC[0][4] + beta * C[CIndex + CinitOffset + 4 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 5 * TILESIZE * ldc] = alpha*rC[0][5] + beta * C[CIndex + CinitOffset + 5 * TILESIZE * ldc] ;
        CinitOffset += TILESIZE;
        C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[1][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[1][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 4 * TILESIZE * ldc] = alpha*rC[1][4] + beta * C[CIndex + CinitOffset + 4 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 5 * TILESIZE * ldc] = alpha*rC[1][5] + beta * C[CIndex + CinitOffset + 5 * TILESIZE * ldc] ;
        CinitOffset += TILESIZE;
        C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[2][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[2][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[2][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[2][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 4 * TILESIZE * ldc] = alpha*rC[2][4] + beta * C[CIndex + CinitOffset + 4 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 5 * TILESIZE * ldc] = alpha*rC[2][5] + beta * C[CIndex + CinitOffset + 5 * TILESIZE * ldc] ;
        CinitOffset += TILESIZE;
        C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[3][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[3][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[3][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[3][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 4 * TILESIZE * ldc] = alpha*rC[3][4] + beta * C[CIndex + CinitOffset + 4 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 5 * TILESIZE * ldc] = alpha*rC[3][5] + beta * C[CIndex + CinitOffset + 5 * TILESIZE * ldc] ;
        CinitOffset += TILESIZE;
        C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[4][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[4][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[4][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[4][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 4 * TILESIZE * ldc] = alpha*rC[4][4] + beta * C[CIndex + CinitOffset + 4 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 5 * TILESIZE * ldc] = alpha*rC[4][5] + beta * C[CIndex + CinitOffset + 5 * TILESIZE * ldc] ;
        CinitOffset += TILESIZE;
        C[CIndex + CinitOffset + 0 * TILESIZE * ldc] = alpha*rC[5][0] + beta * C[CIndex + CinitOffset + 0 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 1 * TILESIZE * ldc] = alpha*rC[5][1] + beta * C[CIndex + CinitOffset + 1 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 2 * TILESIZE * ldc] = alpha*rC[5][2] + beta * C[CIndex + CinitOffset + 2 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 3 * TILESIZE * ldc] = alpha*rC[5][3] + beta * C[CIndex + CinitOffset + 3 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 4 * TILESIZE * ldc] = alpha*rC[5][4] + beta * C[CIndex + CinitOffset + 4 * TILESIZE * ldc] ;
        C[CIndex + CinitOffset + 5 * TILESIZE * ldc] = alpha*rC[5][5] + beta * C[CIndex + CinitOffset + 5 * TILESIZE * ldc] ;
    }
  });
#undef TILESIZE
#undef MICROTILESIZE
  return HCBLAS_SUCCEEDS;
}
