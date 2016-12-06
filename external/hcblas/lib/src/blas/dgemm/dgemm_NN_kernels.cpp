#include "dgemm_array_kernels.h"
#include <cmath>
#include "hc_math.hpp"

hcblasStatus gemm_NoTransAB_MICRO_NBK_Mini_Batch_M128_N128_K16_TS16XMTS2_MB2(hc::accelerator_view accl_view,
                                                const double *A, long aOffset,
                                                const double *B, long bOffset,
                                                double *C, long cOffset,
                                                int M, int N, int K, int lda, int ldb, int ldc,
                                                double alpha, double beta) {
  int M_ = (M-1)/4 + 1;
  int N_ = (N-1)/4 + 1;
  int N_R = (N_ + 15) & ~15;
  int M_R = (M_ + 15) & ~15;
  int K_R = (K + 15) & ~15;
  hc::extent<2> grdExt(N_R, M_R);
  hc::tiled_extent<2> t_ext = grdExt.tile(16, 16);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2> tidx) __attribute__((hc, cpu)) {
    double rC[4][4] = {{(double)0}};
    double rA[1][4];
    double rB[1][4];
    tile_static double lA[16 * 32 * 2 + 16];
    tile_static double lB[16 * 32 * 2 + 16];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int block_k = 0;
    int alIndex = (idy * (32 * 2 + 1)) + idx * 2;
    int blIndex = (idx * (32 * 2 + 1)) + idy * 2;
    int AIndex = aOffset + (gidx * 32 * 2) + idx * 2 + (idy * lda);
    int BIndex = bOffset + ((gidy * 32 * 2) + idy * 2)*ldb + idx;
    long CIndex = cOffset + (gidx * 32 * 2) + idx * 2 + (((gidy * 32 * 2) + idy * 2) * ldc);
    long AinitOffset = 0;
    long BinitOffset = 0;
    long CinitOffset = 0;
    do {

      tidx.barrier.wait();

          lB[blIndex + 0 * 2 + 0] = B[BIndex + BinitOffset + (0 * 2 + 0) * ldb];
          lB[blIndex + 0 * 2 + 1] = B[BIndex + BinitOffset + (0 * 2 + 1) * ldb];
          lB[blIndex + 16 * 2 + 0] = B[BIndex + BinitOffset +  (16 * 2 + 0) * ldb];
          lB[blIndex + 16 * 2 + 1] = B[BIndex + BinitOffset +  (16 * 2 + 1) * ldb];
          lA[alIndex + 0 * 2 + 0] = A[AIndex + 0 * 2 + 0 + AinitOffset];
          lA[alIndex + 0 * 2 + 1] = A[AIndex + 0 * 2 + 1 + AinitOffset];
          lA[alIndex + 16 * 2 + 0] = A[AIndex + 16 * 2 + 0 + AinitOffset];
          lA[alIndex + 16 * 2 + 1] = A[AIndex + 16 * 2 + 1 + AinitOffset];

      tidx.barrier.wait();

      int offA = idx * 2;
      int offB = idy * 2;

      for (int iter = 0; iter < 16; iter++) {
        M2x2_MB;
      }

      AinitOffset += lda << 4;
      BinitOffset += 16;

    } while (++block_k < (K_R >> 4));

      tidx.barrier.wait();
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] = alpha*rC[0][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] = alpha*rC[0][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] ;
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] = alpha*rC[1][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] = alpha*rC[1][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] ;
        CinitOffset+=16*2;
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] = alpha*rC[2][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] = alpha*rC[2][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] = alpha*rC[2][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] = alpha*rC[2][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] = alpha*rC[3][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] = alpha*rC[3][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] ;
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] = alpha*rC[3][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] = alpha*rC[3][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] ;

  });
  return HCBLAS_SUCCEEDS;
}

hcblasStatus gemm_NoTransAB_MICRO_NBK_Mini_Batch_M128_N128_K16_TS16XMTS4_MB2(hc::accelerator_view accl_view,
					        const double *A, long aOffset,
					        const double *B, long bOffset,
					        double *C, long cOffset,
					        int M, int N, int K, int lda, int ldb, int ldc,
					        double alpha, double beta) {
  int M_ = M >> 3;
  int N_ = N >> 3;
  hc::extent<2> grdExt((N_ + 15) & ~15, (M_ + 15) & ~15);
  hc::tiled_extent<2> t_ext = grdExt.tile(16, 16);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2> tidx) __attribute__((hc, cpu)) {
    double rC[8][8] = {{(double)0}};
    double rA[1][8];
    double rB[1][8];
    tile_static double lA[16 * 64 * 2 + 16];
    tile_static double lB[16 * 64 * 2 + 16];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int block_k = K >> 4;
    int alIndex = (idy * (64 * 2 + 1)) + idx * 2;
    int blIndex = (idx * (64 * 2 + 1)) + idy * 2;
    int AIndex = aOffset + (gidx * 64 * 2) + idx * 2 + (idy * lda);
    int BIndex = bOffset + ((gidy * 64 * 2) + idy * 2)*ldb + idx;
    long CIndex = cOffset + (gidx * 64 * 2) + idx * 2 + (((gidy * 64 * 2) + idy * 2) * ldc);
    long AinitOffset = 0;
    long BinitOffset = 0;
    long CinitOffset = 0;
    do {

      tidx.barrier.wait();
      lB[blIndex + 0 * 2 + 0] = B[BIndex + BinitOffset + (0 * 2 + 0) * ldb];
      lB[blIndex + 0 * 2 + 1] = B[BIndex + BinitOffset + (0 * 2 + 1) * ldb];
      lB[blIndex + 16 * 2 + 0] = B[BIndex + BinitOffset +  (16 * 2 + 0) * ldb];
      lB[blIndex + 16 * 2 + 1] = B[BIndex + BinitOffset +  (16 * 2 + 1) * ldb];
      lB[blIndex + 32 * 2 + 0] = B[BIndex + BinitOffset +  (32 * 2 + 0) * ldb];
      lB[blIndex + 32 * 2 + 1] = B[BIndex + BinitOffset +  (32 * 2 + 1) * ldb];
      lB[blIndex + 48 * 2 + 0] = B[BIndex + BinitOffset +  (48 * 2 + 0) * ldb];
      lB[blIndex + 48 * 2 + 1] = B[BIndex + BinitOffset +  (48 * 2 + 1) * ldb];
      lA[alIndex + 0 * 2 + 0] = A[AIndex + 0 * 2 + 0 + AinitOffset];
      lA[alIndex + 0 * 2 + 1] = A[AIndex + 0 * 2 + 1 + AinitOffset];
      lA[alIndex + 16 * 2 + 0] = A[AIndex + 16 * 2 + 0 + AinitOffset];
      lA[alIndex + 16 * 2 + 1] = A[AIndex + 16 * 2 + 1 + AinitOffset];
      lA[alIndex + 32 * 2 + 0] = A[AIndex + 32 * 2 + 0 + AinitOffset];
      lA[alIndex + 32 * 2 + 1] = A[AIndex + 32 * 2 + 1 + AinitOffset];
      lA[alIndex + 48 * 2 + 0] = A[AIndex + 48 * 2 + 0 + AinitOffset];
      lA[alIndex + 48 * 2 + 1] = A[AIndex + 48 * 2 + 1 + AinitOffset];

      tidx.barrier.wait();

      int offA = idx * 2;
      int offB = idy * 2;

      for (int iter = 0; iter < 16; iter+=1) {
        M4x4_MB;
      }

      AinitOffset += lda << 4;
      BinitOffset += 16;

    } while (--block_k > 0); // (((K + TILESIZE - 1) & ~(TILESIZE - 1)) / TILESIZE));

    C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] ;
    C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] ;
    C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] = alpha*rC[0][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] ;
    C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] = alpha*rC[0][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] ;
    C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] = alpha*rC[0][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] ;
    C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] = alpha*rC[0][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] ;
    C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] = alpha*rC[0][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] ;
    C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] = alpha*rC[0][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] ;
    C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] ;
    C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] ;
    C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] = alpha*rC[1][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] ;
    C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] = alpha*rC[1][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] ;
    C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] = alpha*rC[1][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] ;
    C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] = alpha*rC[1][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] ;
    C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] = alpha*rC[1][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] ;
    C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] = alpha*rC[1][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] ;
    CinitOffset+=16*2;
    C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] = alpha*rC[2][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] ;
    C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] = alpha*rC[2][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] ;
    C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] = alpha*rC[2][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] ;
    C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] = alpha*rC[2][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] ;
    C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] = alpha*rC[2][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] ;
    C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] = alpha*rC[2][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] ;
    C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] = alpha*rC[2][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] ;
    C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] = alpha*rC[2][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] ;
    C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] = alpha*rC[3][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] ;
    C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] = alpha*rC[3][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] ;
    C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] = alpha*rC[3][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] ;
    C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] = alpha*rC[3][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] ;
    C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] = alpha*rC[3][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] ;
    C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] = alpha*rC[3][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] ;
    C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] = alpha*rC[3][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] ;
    C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] = alpha*rC[3][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] ;
    CinitOffset+=16*2;
    C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] = alpha*rC[4][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] ;
    C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] = alpha*rC[4][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] ;
    C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] = alpha*rC[4][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] ;
    C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] = alpha*rC[4][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] ;
    C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] = alpha*rC[4][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] ;
    C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] = alpha*rC[4][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] ;
    C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] = alpha*rC[4][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] ;
    C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] = alpha*rC[4][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] ;
    C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] = alpha*rC[5][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] ;
    C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] = alpha*rC[5][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] ;
    C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] = alpha*rC[5][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] ;
    C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] = alpha*rC[5][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] ;
    C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] = alpha*rC[5][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] ;
    C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] = alpha*rC[5][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] ;
    C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] = alpha*rC[5][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] ;
    C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] = alpha*rC[5][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] ;
    CinitOffset+=16*2;
    C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] = alpha*rC[6][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] ;
    C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] = alpha*rC[6][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] ;
    C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] = alpha*rC[6][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] ;
    C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] = alpha*rC[6][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] ;
    C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] = alpha*rC[6][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] ;
    C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] = alpha*rC[6][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] ;
    C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] = alpha*rC[6][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] ;
    C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] = alpha*rC[6][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] ;
    C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] = alpha*rC[7][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] ;
    C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] = alpha*rC[7][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] ;
    C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] = alpha*rC[7][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] ;
    C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] = alpha*rC[7][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] ;
    C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] = alpha*rC[7][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] ;
    C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] = alpha*rC[7][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] ;
    C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] = alpha*rC[7][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] ;
    C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] = alpha*rC[7][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] ;

  });
  return HCBLAS_SUCCEEDS;
}

hcblasStatus gemm_NoTransAB_MICRO_NBK_Mini_Batch_M_N_K_TS16XMTS2_MB2(hc::accelerator_view accl_view,
                                                const double *A, long aOffset,
                                                const double *B, long bOffset,
                                                double *C, long cOffset,
                                                int M, int N, int K, int lda, int ldb, int ldc,
                                                double alpha, double beta) {
  int M_ = (M-1)/4 + 1;
  int N_ = (N-1)/4 + 1;
  int N_R = (N_ + 15) & ~15;
  int M_R = (M_ + 15) & ~15;
  int K_R = (K + 15) & ~15;
  hc::extent<2> grdExt(N_R, M_R);
  hc::tiled_extent<2> t_ext = grdExt.tile(16, 16);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2> tidx) __attribute__((hc, cpu)) {
    double rC[4][4] = {{(double)0}};
    double rA[1][4];
    double rB[1][4];
    tile_static double lA[16 * 32 * 2 + 16];
    tile_static double lB[16 * 32 * 2 + 16];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int block_k = 0;
    int alIndex = (idy * (32 * 2 + 1)) + idx * 2;
    int blIndex = (idx * (32 * 2 + 1)) + idy * 2;
    int AIndex = aOffset + (gidx * 32 * 2) + idx * 2 + (idy * lda);
    int BIndex = bOffset + ((gidy * 32 * 2) + idy * 2)*ldb + idx;
    long CIndex = cOffset + (gidx * 32 * 2) + idx * 2 + (((gidy * 32 * 2) + idy * 2) * ldc);
    long AinitOffset = 0;
    long BinitOffset = 0;
    long CinitOffset = 0;
    int N_block = N_R >> 4;
    int M_block = M_R >> 4;
    int K_block = K_R >> 4;
    do {

      tidx.barrier.wait();

      if(gidx == M_block-1 || gidy == N_block-1 || block_k == K_block-1)
      {
          for(int sec = 0; sec < 2; ++sec) {
            int secVal = sec << 4;

            if( (gidy*32*2 + idy * 2 + secVal * 2 + 0) < N && (block_k * 16 + idx) < K) {
              lB[ blIndex + secVal * 2 + 0] = B[BIndex + BinitOffset + (secVal * 2 + 0) * ldb];
            } else {
              lB[blIndex + secVal * 2 + 0] = 0;
            }
            if( (gidy*32*2 + idy * 2 + secVal * 2 + 1) < N && (block_k * 16 + idx) < K) {
              lB[ blIndex + secVal * 2 + 1] = B[BIndex + BinitOffset + (secVal * 2 + 1) * ldb];
            } else {
              lB[blIndex + secVal * 2 + 1] = 0;
            }
 
            if( (gidx*32*2 + idx * 2 + secVal * 2 + 0) < M && (block_k * 16 + idy) < K) {
              lA[ alIndex + secVal * 2 + 0] = A[AIndex + secVal * 2 + 0 + AinitOffset];
            } else {
              lA[ alIndex + secVal * 2 + 0] = 0;
            }
            if( (gidx*32*2 + idx * 2 + secVal * 2 + 1) < M && (block_k * 16 + idy) < K) {
              lA[ alIndex + secVal * 2 + 1] = A[AIndex + secVal * 2 + 1 + AinitOffset];
            } else {
              lA[ alIndex + secVal * 2 + 1] = 0;
            }
          }
      }
      else
      {
          lB[blIndex + 0 * 2 + 0] = B[BIndex + BinitOffset + (0 * 2 + 0) * ldb];
          lB[blIndex + 0 * 2 + 1] = B[BIndex + BinitOffset + (0 * 2 + 1) * ldb];
          lB[blIndex + 16 * 2 + 0] = B[BIndex + BinitOffset +  (16 * 2 + 0) * ldb];
          lB[blIndex + 16 * 2 + 1] = B[BIndex + BinitOffset +  (16 * 2 + 1) * ldb];
          lA[alIndex + 0 * 2 + 0] = A[AIndex + 0 * 2 + 0 + AinitOffset];
          lA[alIndex + 0 * 2 + 1] = A[AIndex + 0 * 2 + 1 + AinitOffset];
          lA[alIndex + 16 * 2 + 0] = A[AIndex + 16 * 2 + 0 + AinitOffset];
          lA[alIndex + 16 * 2 + 1] = A[AIndex + 16 * 2 + 1 + AinitOffset];
      }

      tidx.barrier.wait();

      int offA = idx * 2;
      int offB = idy * 2;

      for (int iter = 0; iter < 16; iter++) {
        M2x2_MB;
      }

      AinitOffset += lda << 4;
      BinitOffset += 16;

    } while (++block_k < (K_R >> 4));

      tidx.barrier.wait();

    if(gidx == M_block-1 || gidy == N_block-1)
    {
        if((gidx * 32*2 + idx * 2 + CinitOffset)  < M && (gidy * 32*2 + idy * 2 + (0 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] ;
        if((gidx * 32*2 + idx * 2 + CinitOffset)  < M && (gidy * 32*2 + idy * 2 + (0 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] ;
        if((gidx * 32*2 + idx * 2 + CinitOffset)  < M && (gidy * 32*2 + idy * 2 + (1 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] = alpha*rC[0][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] ;
        if((gidx * 32*2 + idx * 2 + CinitOffset)  < M && (gidy * 32*2 + idy * 2 + (1 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] = alpha*rC[0][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] ;
        if((gidx * 32*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 32*2 + idy * 2 + (0 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] ;
        if((gidx * 32*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 32*2 + idy * 2 + (0 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] ;
        if((gidx * 32*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 32*2 + idy * 2 + (1 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] = alpha*rC[1][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] ;
        if((gidx * 32*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 32*2 + idy * 2 + (1 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] = alpha*rC[1][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] ;
        CinitOffset+=16*2;
        if((gidx * 32*2 + idx * 2 + CinitOffset)  < M && (gidy * 32*2 + idy * 2 + (0 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] = alpha*rC[2][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] ;
        if((gidx * 32*2 + idx * 2 + CinitOffset)  < M && (gidy * 32*2 + idy * 2 + (0 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] = alpha*rC[2][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] ;
        if((gidx * 32*2 + idx * 2 + CinitOffset)  < M && (gidy * 32*2 + idy * 2 + (1 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] = alpha*rC[2][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] ;
        if((gidx * 32*2 + idx * 2 + CinitOffset)  < M && (gidy * 32*2 + idy * 2 + (1 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] = alpha*rC[2][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] ;
        if((gidx * 32*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 32*2 + idy * 2 + (0 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] = alpha*rC[3][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] ;
        if((gidx * 32*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 32*2 + idy * 2 + (0 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] = alpha*rC[3][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] ;
        if((gidx * 32*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 32*2 + idy * 2 + (1 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] = alpha*rC[3][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] ;
        if((gidx * 32*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 32*2 + idy * 2 + (1 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] = alpha*rC[3][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] ;
    }
    else
    {
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] = alpha*rC[0][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] = alpha*rC[0][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] ;
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] = alpha*rC[1][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] = alpha*rC[1][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] ;
        CinitOffset+=16*2;
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] = alpha*rC[2][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] = alpha*rC[2][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] = alpha*rC[2][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] = alpha*rC[2][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] = alpha*rC[3][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] = alpha*rC[3][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] ;
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] = alpha*rC[3][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] = alpha*rC[3][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] ;
    }

  });
  return HCBLAS_SUCCEEDS;
}

hcblasStatus gemm_NoTransAB_MICRO_NBK_Mini_Batch_M_N_K_TS16XMTS4_MB2(hc::accelerator_view accl_view,
					        const double *A, long aOffset,
					        const double *B, long bOffset,
					        double *C, long cOffset,
					        int M, int N, int K, int lda, int ldb, int ldc,
					        double alpha, double beta) {
  int M_ = (M-1)/8 + 1;
  int N_ = (N-1)/8 + 1;
  int N_R = (N_ + 15) & ~15;
  int M_R = (M_ + 15) & ~15;
  int K_R = (K + 15) & ~15;
  hc::extent<2> grdExt(N_R, M_R);
  hc::tiled_extent<2> t_ext = grdExt.tile(16, 16);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2> tidx) __attribute__((hc, cpu)) {
    double rC[8][8] = {{(double)0}};
    double rA[1][8];
    double rB[1][8];
    tile_static double lA[16 * 64 * 2 + 16];
    tile_static double lB[16 * 64 * 2 + 16];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int block_k = 0;
    int alIndex = (idy * (64 * 2 + 1)) + idx * 2;
    int blIndex = (idx * (64 * 2 + 1)) + idy * 2;
    int AIndex = aOffset + (gidx * 64 * 2) + idx * 2 + (idy * lda);
    int BIndex = bOffset + ((gidy * 64 * 2) + idy * 2)*ldb + idx;
    long CIndex = cOffset + (gidx * 64 * 2) + idx * 2 + (((gidy * 64 * 2) + idy * 2) * ldc);
    long AinitOffset = 0;
    long BinitOffset = 0;
    long CinitOffset = 0;
    int N_block = N_R >> 4;
    int M_block = M_R >> 4;
    int K_block = K_R >> 4;
    do {

      tidx.barrier.wait();

      if(gidx == M_block-1 || gidy == N_block-1 || block_k == K_block-1)
      {
          for(int sec = 0; sec < 4; ++sec) {
            int secVal = sec << 4;

            if( (gidy*64*2 + idy * 2 + secVal * 2 + 0) < N && (block_k * 16 + idx) < K) {
              lB[ blIndex + secVal * 2 + 0] = B[BIndex + BinitOffset + (secVal * 2 + 0) * ldb];
            } else {
              lB[blIndex + secVal * 2 + 0] = 0;
            }
            if( (gidy*64*2 + idy * 2 + secVal * 2 + 1) < N && (block_k * 16 + idx) < K) {
              lB[ blIndex + secVal * 2 + 1] = B[BIndex + BinitOffset + (secVal * 2 + 1) * ldb];
            } else {
              lB[blIndex + secVal * 2 + 1] = 0;
            }

            if( (gidx*64*2 + idx * 2 + secVal * 2 + 0) < M && (block_k * 16 + idy) < K) {
              lA[ alIndex + secVal * 2 + 0] = A[AIndex + secVal * 2 + 0 + AinitOffset];
            } else {
              lA[ alIndex + secVal * 2 + 0] = 0;
            }
            if( (gidx*64*2 + idx * 2 + secVal * 2 + 1) < M && (block_k * 16 + idy) < K) {
              lA[ alIndex + secVal * 2 + 1] = A[AIndex + secVal * 2 + 1 + AinitOffset];
            } else {
              lA[ alIndex + secVal * 2 + 1] = 0;
            }
          }
      }
      else
      {
          lB[blIndex + 0 * 2 + 0] = B[BIndex + BinitOffset + (0 * 2 + 0) * ldb];
          lB[blIndex + 0 * 2 + 1] = B[BIndex + BinitOffset + (0 * 2 + 1) * ldb];
          lB[blIndex + 16 * 2 + 0] = B[BIndex + BinitOffset +  (16 * 2 + 0) * ldb];
          lB[blIndex + 16 * 2 + 1] = B[BIndex + BinitOffset +  (16 * 2 + 1) * ldb];
          lB[blIndex + 32 * 2 + 0] = B[BIndex + BinitOffset +  (32 * 2 + 0) * ldb];
          lB[blIndex + 32 * 2 + 1] = B[BIndex + BinitOffset +  (32 * 2 + 1) * ldb];
          lB[blIndex + 48 * 2 + 0] = B[BIndex + BinitOffset +  (48 * 2 + 0) * ldb];
          lB[blIndex + 48 * 2 + 1] = B[BIndex + BinitOffset +  (48 * 2 + 1) * ldb];
          lA[alIndex + 0 * 2 + 0] = A[AIndex + 0 * 2 + 0 + AinitOffset];
          lA[alIndex + 0 * 2 + 1] = A[AIndex + 0 * 2 + 1 + AinitOffset];
          lA[alIndex + 16 * 2 + 0] = A[AIndex + 16 * 2 + 0 + AinitOffset];
          lA[alIndex + 16 * 2 + 1] = A[AIndex + 16 * 2 + 1 + AinitOffset];
          lA[alIndex + 32 * 2 + 0] = A[AIndex + 32 * 2 + 0 + AinitOffset];
          lA[alIndex + 32 * 2 + 1] = A[AIndex + 32 * 2 + 1 + AinitOffset];
          lA[alIndex + 48 * 2 + 0] = A[AIndex + 48 * 2 + 0 + AinitOffset];
          lA[alIndex + 48 * 2 + 1] = A[AIndex + 48 * 2 + 1 + AinitOffset];
      }

      tidx.barrier.wait();

      int offA = idx * 2;
      int offB = idy * 2;

      for (int iter = 0; iter < 16; iter+=1) {
        M4x4_MB;
      }

      AinitOffset += lda << 4;
      BinitOffset += 16;

    } while (++block_k < (K_R >> 4));

    if(gidx == M_block-1 || gidy == N_block-1)
    {
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (0 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (0 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (1 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] = alpha*rC[0][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (1 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] = alpha*rC[0][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (2 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] = alpha*rC[0][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (2 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] = alpha*rC[0][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (3 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] = alpha*rC[0][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (3 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] = alpha*rC[0][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (0 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (0 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (1 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] = alpha*rC[1][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (1 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] = alpha*rC[1][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (2 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] = alpha*rC[1][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (2 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] = alpha*rC[1][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (3 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] = alpha*rC[1][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (3 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] = alpha*rC[1][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] ;
        CinitOffset+=16*2;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (0 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] = alpha*rC[2][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (0 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] = alpha*rC[2][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (1 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] = alpha*rC[2][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (1 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] = alpha*rC[2][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (2 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] = alpha*rC[2][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (2 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] = alpha*rC[2][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (3 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] = alpha*rC[2][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (3 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] = alpha*rC[2][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (0 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] = alpha*rC[3][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (0 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] = alpha*rC[3][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (1 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] = alpha*rC[3][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (1 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] = alpha*rC[3][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (2 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] = alpha*rC[3][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (2 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] = alpha*rC[3][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (3 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] = alpha*rC[3][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (3 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] = alpha*rC[3][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] ;
        CinitOffset+=16*2;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (0 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] = alpha*rC[4][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (0 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] = alpha*rC[4][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (1 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] = alpha*rC[4][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (1 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] = alpha*rC[4][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (2 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] = alpha*rC[4][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (2 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] = alpha*rC[4][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (3 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] = alpha*rC[4][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (3 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] = alpha*rC[4][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (0 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] = alpha*rC[5][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (0 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] = alpha*rC[5][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (1 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] = alpha*rC[5][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (1 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] = alpha*rC[5][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (2 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] = alpha*rC[5][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (2 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] = alpha*rC[5][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (3 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] = alpha*rC[5][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (3 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] = alpha*rC[5][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] ;
        CinitOffset+=16*2;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (0 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] = alpha*rC[6][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (0 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] = alpha*rC[6][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (1 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] = alpha*rC[6][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (1 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] = alpha*rC[6][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (2 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] = alpha*rC[6][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (2 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] = alpha*rC[6][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (3 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] = alpha*rC[6][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset)  < M && (gidy * 64*2 + idy * 2 + (3 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] = alpha*rC[6][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (0 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] = alpha*rC[7][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (0 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] = alpha*rC[7][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (1 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] = alpha*rC[7][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (1 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] = alpha*rC[7][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (2 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] = alpha*rC[7][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (2 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] = alpha*rC[7][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (3 * 16 * 2 + 0)) < N)
        C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] = alpha*rC[7][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] ;
        if((gidx * 64*2 + idx * 2 + CinitOffset + 1)  < M && (gidy * 64*2 + idy * 2 + (3 * 16 * 2 + 1)) < N)
        C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] = alpha*rC[7][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] ;
    }
    else
    {
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] = alpha*rC[0][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] = alpha*rC[0][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] = alpha*rC[0][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] = alpha*rC[0][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] = alpha*rC[0][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] = alpha*rC[0][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] ;
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] = alpha*rC[1][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] = alpha*rC[1][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] ;
        C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] = alpha*rC[1][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] = alpha*rC[1][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] ;
        C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] = alpha*rC[1][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] = alpha*rC[1][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] ;
        CinitOffset+=16*2;
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] = alpha*rC[2][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] = alpha*rC[2][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] = alpha*rC[2][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] = alpha*rC[2][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] = alpha*rC[2][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] = alpha*rC[2][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] = alpha*rC[2][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] = alpha*rC[2][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] = alpha*rC[3][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] = alpha*rC[3][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] ;
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] = alpha*rC[3][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] = alpha*rC[3][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] ;
        C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] = alpha*rC[3][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] = alpha*rC[3][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] ;
        C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] = alpha*rC[3][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] = alpha*rC[3][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] ;
        CinitOffset+=16*2;
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] = alpha*rC[4][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] = alpha*rC[4][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] = alpha*rC[4][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] = alpha*rC[4][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] = alpha*rC[4][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] = alpha*rC[4][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] = alpha*rC[4][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] = alpha*rC[4][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] = alpha*rC[5][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] = alpha*rC[5][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] ;
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] = alpha*rC[5][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] = alpha*rC[5][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] ;
        C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] = alpha*rC[5][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] = alpha*rC[5][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] ;
        C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] = alpha*rC[5][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] = alpha*rC[5][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] ;
        CinitOffset+=16*2;
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] = alpha*rC[6][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] = alpha*rC[6][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] = alpha*rC[6][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] = alpha*rC[6][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] = alpha*rC[6][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] = alpha*rC[6][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] = alpha*rC[6][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 0] ;
        C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] = alpha*rC[6][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 0] ;
        C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] = alpha*rC[7][0] + beta * C[CIndex + CinitOffset + (0 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] = alpha*rC[7][1] + beta * C[CIndex + CinitOffset + (0 * 2 + 1) * ldc + 1] ;
        C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] = alpha*rC[7][2] + beta * C[CIndex + CinitOffset + (16 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] = alpha*rC[7][3] + beta * C[CIndex + CinitOffset + (16 * 2 + 1) * ldc + 1] ;
        C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] = alpha*rC[7][4] + beta * C[CIndex + CinitOffset + (32 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] = alpha*rC[7][5] + beta * C[CIndex + CinitOffset + (32 * 2 + 1) * ldc + 1] ;
        C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] = alpha*rC[7][6] + beta * C[CIndex + CinitOffset + (48 * 2 + 0) * ldc + 1] ;
        C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] = alpha*rC[7][7] + beta * C[CIndex + CinitOffset + (48 * 2 + 1) * ldc + 1] ;
    }

  });
  return HCBLAS_SUCCEEDS;
}

hcblasStatus gemm_NoTransAB_MICRO_NBK_MX064_NX064_KX16_TS16XMTS4(hc::accelerator_view accl_view,
					        const double *A, long aOffset,
					        const double *B, long bOffset,
					        double *C, long cOffset,
					        int M, int N, int K, int lda, int ldb, int ldc,
					        double alpha, double beta) {
  int M_ = M >> 2;
  int N_ = N >> 2;
  hc::extent<2> grdExt((N_ + 15) & ~15, (M_ + 15) & ~15);
  hc::tiled_extent<2> t_ext = grdExt.tile(16, 16);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2> tidx) __attribute__((hc, cpu)) {
    double rC[4][4] = {{(double)0}};
    double rA[1][4];
    double rB[1][4];
    tile_static double lA[1056];
    tile_static double lB[1056];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int block_k = K >> 4;
    int alIndex = (idy * 65) + idx;
    int blIndex = (idx * 65) + idy;
    int AIndex = aOffset + (gidx * 64) + idx + (idy * lda);
    int BIndex = bOffset + ((gidy * 64) + idy)*ldb + idx;
    long CIndex = cOffset + (gidx * 64) + idx + (((gidy * 64) + idy) * ldc);
    long AinitOffset = 0;
    long BinitOffset = 0;
    long CinitOffset = 0;
    do {

      tidx.barrier.wait();
      lB[blIndex] = B[BIndex + BinitOffset];
      lB[blIndex + 16] = B[BIndex + BinitOffset +  16 * ldb];
      lB[blIndex + 32] = B[BIndex + BinitOffset +  32 * ldb];
      lB[blIndex + 48] = B[BIndex + BinitOffset +  48 * ldb];
      lA[alIndex] = A[AIndex + AinitOffset ];
      lA[alIndex + 16] = A[AIndex + 16 + AinitOffset];
      lA[alIndex + 32] = A[AIndex + 32 + AinitOffset];
      lA[alIndex + 48] = A[AIndex + 48 + AinitOffset];

      tidx.barrier.wait();

      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < 16; iter+=8) {
        M4x4;
        M4x4;
        M4x4;
        M4x4;
        M4x4;
        M4x4;
        M4x4;
        M4x4;
      }

      AinitOffset += lda << 4;
      BinitOffset += 16;

    } while (--block_k > 0); // (((K + TILESIZE - 1) & ~(TILESIZE - 1)) / TILESIZE));

    C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
    C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
    C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[0][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
    C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[0][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
    CinitOffset+=16;
    C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
    C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
    C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[1][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
    C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[1][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
    CinitOffset+=16;
    C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[2][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
    C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[2][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
    C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[2][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
    C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[2][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
    CinitOffset+=16;
    C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[3][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
    C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[3][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
    C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[3][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
    C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[3][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;

  });
  return HCBLAS_SUCCEEDS;
}

hcblasStatus gemm_NoTransAB_MICRO_NBK_M_N_K_TS16XMTS2(hc::accelerator_view accl_view,
                                                const double *A, long aOffset,
                                                const double *B, long bOffset,
                                                double *C, long cOffset,
                                                int M, int N, int K, int lda, int ldb, int ldc,
                                                double alpha, double beta) {
  int M_ = (M-1)/2 + 1;
  int N_ = (N-1)/2 + 1;
  int N_R = (N_ + 15) & ~15;
  int M_R = (M_ + 15) & ~15;
  int K_R = (K + 15) & ~15;
  hc::extent<2> grdExt(N_R, M_R);
  hc::tiled_extent<2> t_ext = grdExt.tile(16, 16);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2> tidx) __attribute__((hc, cpu)) {
    double rC[2][2] = {{(double)0}};
    double rA[1][2];
    double rB[1][2];
    tile_static double lA[528];
    tile_static double lB[528];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int block_k = 0;
    int alIndex = (idy * 33) + idx;
    int blIndex = (idx * 33) + idy;
    int AIndex = aOffset + (gidx * 32) + idx + (idy * lda);
    int BIndex = bOffset + ((gidy * 32) + idy)*ldb + idx;
    long CIndex = cOffset + (gidx * 32) + idx + (((gidy * 32) + idy) * ldc);
    long AinitOffset = 0;
    long BinitOffset = 0;
    long CinitOffset = 0;
    int N_block = N_R >> 4;
    int M_block = M_R >> 4;
    int K_block = K_R >> 4;
    do {

      tidx.barrier.wait();

      if(gidx == M_block-1 || gidy == N_block-1 || block_k == K_block-1)
      {
          for(int sec = 0; sec < 2; ++sec) {
            int secVal = sec << 4;

            if( (gidy*32 + idy + secVal) < N && (block_k * 16 + idx) < K) {
              lB[ blIndex + secVal] = B[BIndex + BinitOffset + secVal * ldb];
            } else {
              lB[blIndex + secVal] = 0;
            }
 
            if( (gidx*32 + idx + secVal) < M && (block_k * 16 + idy) < K) {
              lA[ alIndex + secVal] = A[AIndex + secVal + AinitOffset];
            } else {
              lA[ alIndex + secVal] = 0;
            }
          }
      }
      else
      {
          lB[blIndex] = B[BIndex + BinitOffset];
          lB[blIndex + 16] = B[BIndex + BinitOffset +  16 * ldb];
          lA[alIndex] = A[AIndex + AinitOffset ];
          lA[alIndex + 16] = A[AIndex + 16 + AinitOffset];
      }

      tidx.barrier.wait();

      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < 16; iter+=8) {
        M2x2;
        M2x2;
        M2x2;
        M2x2;
        M2x2;
        M2x2;
        M2x2;
        M2x2;
      }

      AinitOffset += lda << 4;
      BinitOffset += 16;

    } while (++block_k < (K_R >> 4));

      tidx.barrier.wait();
    if(gidx == M_block-1 || gidy == N_block-1)
    {
        if((gidx * 32 + idx + CinitOffset)  < M && (gidy * 32 + idy + 0 * 16) < N)
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        if((gidx * 32 + idx + CinitOffset)  < M && (gidy * 32 + idy + 1 * 16) < N)
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        CinitOffset+=16;
        if((gidx * 32 + idx + CinitOffset)  < M && (gidy * 32 + idy + 0 * 16) < N)
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        if((gidx * 32 + idx + CinitOffset)  < M && (gidy * 32 + idy + 1 * 16) < N)
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
    }
    else
    {
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        CinitOffset+=16;
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
    }

  });
  return HCBLAS_SUCCEEDS;
}

hcblasStatus gemm_NoTransAB_MICRO_NBK_M_N_K_TS8XMTS4(hc::accelerator_view accl_view,
                                                const double *A, long aOffset,
                                                const double *B, long bOffset,
                                                double *C, long cOffset,
                                                int M, int N, int K, int lda, int ldb, int ldc,
                                                double alpha, double beta) {
  int M_ = (M-1)/4 + 1;
  int N_ = (N-1)/4 + 1;
  int N_R = (N_ + 7) & ~7;
  int M_R = (M_ + 7) & ~7;
  int K_R = (K + 7) & ~7;
  hc::extent<2> grdExt(N_R, M_R);
  hc::tiled_extent<2> t_ext = grdExt.tile(8, 8);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2> tidx) __attribute__((hc, cpu)) {
    double rC[4][4] = {{(double)0}};
    double rA[1][4];
    double rB[1][4];
    tile_static double lA[264];
    tile_static double lB[264];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int block_k = 0;
    int alIndex = (idy * 33) + idx;
    int blIndex = (idx * 33) + idy;
    int AIndex = aOffset + (gidx * 32) + idx + (idy * lda);
    int BIndex = bOffset + ((gidy * 32) + idy)*ldb + idx;
    long CIndex = cOffset + (gidx * 32) + idx + (((gidy * 32) + idy) * ldc);
    long AinitOffset = 0;
    long BinitOffset = 0;
    long CinitOffset = 0;
    int N_block = N_R >> 3;
    int M_block = M_R >> 3;
    int K_block = K_R >> 3;
    do {

      tidx.barrier.wait();

      if(gidx == M_block-1 || gidy == N_block-1 || block_k == K_block-1)
      {
          for(int sec = 0; sec < 4; ++sec) {
            int secVal = sec << 3;

            if( (gidy*32 + idy + secVal) < N && (block_k * 8 + idx) < K) {
              lB[ blIndex + secVal] = B[BIndex + BinitOffset + secVal * ldb];
            } else {
              lB[blIndex + secVal] = 0;
            }

            if( (gidx*32 + idx + secVal) < M && (block_k * 8 + idy) < K) {
              lA[ alIndex + secVal] = A[AIndex + secVal + AinitOffset];
            } else {
              lA[ alIndex + secVal] = 0;
            }
          }
      }
      else
      {
          lB[blIndex] = B[BIndex + BinitOffset];
          lB[blIndex + 8] = B[BIndex + BinitOffset +  8 * ldb];
          lB[blIndex + 16] = B[BIndex + BinitOffset +  16 * ldb];
          lB[blIndex + 24] = B[BIndex + BinitOffset +  24 * ldb];
          lA[alIndex] = A[AIndex + AinitOffset ];
          lA[alIndex + 8] = A[AIndex + 8 + AinitOffset];
          lA[alIndex + 16] = A[AIndex + 16 + AinitOffset];
          lA[alIndex + 24] = A[AIndex + 24 + AinitOffset];
      }

      tidx.barrier.wait();

      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < 8; iter+=4) {
        M4x4;
        M4x4;
        M4x4;
        M4x4;
      }

      AinitOffset += lda << 3;
      BinitOffset += 8;

    } while (++block_k < (K_R >> 3));

      tidx.barrier.wait();
    if(gidx == M_block-1 || gidy == N_block-1)
    {
        if((gidx * 32 + idx + CinitOffset)  < M && (gidy * 32 + idy + 0 * 8) < N)
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        if((gidx * 32 + idx + CinitOffset)  < M && (gidy * 32 + idy + 1 * 8) < N)
        C[CIndex + CinitOffset + 8 * ldc] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + 8 * ldc] ;
        if((gidx * 32 + idx + CinitOffset)  < M && (gidy * 32 + idy + 2 * 8) < N)
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[0][2] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        if((gidx * 32 + idx + CinitOffset)  < M && (gidy * 32 + idy + 3 * 8) < N)
        C[CIndex + CinitOffset + 24 * ldc] = alpha*rC[0][3] + beta * C[CIndex + CinitOffset + 24 * ldc] ;
        CinitOffset+=8;
        if((gidx * 32 + idx + CinitOffset)  < M && (gidy * 32 + idy + 0 * 8) < N)
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        if((gidx * 32 + idx + CinitOffset)  < M && (gidy * 32 + idy + 1 * 8) < N)
        C[CIndex + CinitOffset + 8 * ldc] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + 8 * ldc] ;
        if((gidx * 32 + idx + CinitOffset)  < M && (gidy * 32 + idy + 2 * 8) < N)
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[1][2] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        if((gidx * 32 + idx + CinitOffset)  < M && (gidy * 32 + idy + 3 * 8) < N)
        C[CIndex + CinitOffset + 24 * ldc] = alpha*rC[1][3] + beta * C[CIndex + CinitOffset + 24 * ldc] ;
        CinitOffset+=8;
        if((gidx * 32 + idx + CinitOffset)  < M && (gidy * 32 + idy + 0 * 8) < N)
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[2][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        if((gidx * 32 + idx + CinitOffset)  < M && (gidy * 32 + idy + 1 * 8) < N)
        C[CIndex + CinitOffset + 8 * ldc] = alpha*rC[2][1] + beta * C[CIndex + CinitOffset + 8 * ldc] ;
        if((gidx * 32 + idx + CinitOffset)  < M && (gidy * 32 + idy + 2 * 8) < N)
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[2][2] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        if((gidx * 32 + idx + CinitOffset)  < M && (gidy * 32 + idy + 3 * 8) < N)
        C[CIndex + CinitOffset + 24 * ldc] = alpha*rC[2][3] + beta * C[CIndex + CinitOffset + 24 * ldc] ;
        CinitOffset+=8;
        if((gidx * 32 + idx + CinitOffset)  < M && (gidy * 32 + idy + 0 * 8) < N)
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[3][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        if((gidx * 32 + idx + CinitOffset)  < M && (gidy * 32 + idy + 1 * 8) < N)
        C[CIndex + CinitOffset + 8 * ldc] = alpha*rC[3][1] + beta * C[CIndex + CinitOffset + 8 * ldc] ;
        if((gidx * 32 + idx + CinitOffset)  < M && (gidy * 32 + idy + 2 * 8) < N)
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[3][2] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        if((gidx * 32 + idx + CinitOffset)  < M && (gidy * 32 + idy + 3 * 8) < N)
        C[CIndex + CinitOffset + 24 * ldc] = alpha*rC[3][3] + beta * C[CIndex + CinitOffset + 24 * ldc] ;
    }
    else
    {
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        C[CIndex + CinitOffset + 8 * ldc] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + 8 * ldc] ;
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[0][2] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        C[CIndex + CinitOffset + 24 * ldc] = alpha*rC[0][3] + beta * C[CIndex + CinitOffset + 24 * ldc] ;
        CinitOffset+=8;
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        C[CIndex + CinitOffset + 8 * ldc] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + 8 * ldc] ;
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[1][2] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        C[CIndex + CinitOffset + 24 * ldc] = alpha*rC[1][3] + beta * C[CIndex + CinitOffset + 24 * ldc] ;
        CinitOffset+=8;
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[2][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        C[CIndex + CinitOffset + 8 * ldc] = alpha*rC[2][1] + beta * C[CIndex + CinitOffset + 8 * ldc] ;
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[2][2] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        C[CIndex + CinitOffset + 24 * ldc] = alpha*rC[2][3] + beta * C[CIndex + CinitOffset + 24 * ldc] ;
        CinitOffset+=8;
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[3][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        C[CIndex + CinitOffset + 8 * ldc] = alpha*rC[3][1] + beta * C[CIndex + CinitOffset + 8 * ldc] ;
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[3][2] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        C[CIndex + CinitOffset + 24 * ldc] = alpha*rC[3][3] + beta * C[CIndex + CinitOffset + 24 * ldc] ;
    }

  });
  return HCBLAS_SUCCEEDS;
}

hcblasStatus gemm_NoTransAB_MICRO_NBK_M_N_K_TS16XMTS4(hc::accelerator_view accl_view,
					        const double *A, long aOffset,
					        const double *B, long bOffset,
					        double *C, long cOffset,
					        int M, int N, int K, int lda, int ldb, int ldc,
					        double alpha, double beta) {
  int M_ = (M-1)/4 + 1;
  int N_ = (N-1)/4 + 1;
  int N_R = (N_ + 15) & ~15;
  int M_R = (M_ + 15) & ~15;
  int K_R = (K + 15) & ~15;
  hc::extent<2> grdExt(N_R, M_R);
  hc::tiled_extent<2> t_ext = grdExt.tile(16, 16);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2> tidx) __attribute__((hc, cpu)) {
    double rC[4][4] = {{(double)0}};
    double rA[1][4];
    double rB[1][4];
    tile_static double lA[1056];
    tile_static double lB[1056];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int block_k = 0;
    int alIndex = (idy * 65) + idx;
    int blIndex = (idx * 65) + idy;
    int AIndex = aOffset + (gidx * 64) + idx + (idy * lda);
    int BIndex = bOffset + ((gidy * 64) + idy)*ldb + idx;
    long CIndex = cOffset + (gidx * 64) + idx + (((gidy * 64) + idy) * ldc);
    long AinitOffset = 0;
    long BinitOffset = 0;
    long CinitOffset = 0;
    int N_block = N_R >> 4;
    int M_block = M_R >> 4;
    int K_block = K_R >> 4;
    do {

      tidx.barrier.wait();

      if(gidx == M_block-1 || gidy == N_block-1 || block_k == K_block-1)
      {
          for(int sec = 0; sec < 4; ++sec) {
            int secVal = sec << 4;

            if( (gidy*64 + idy + secVal) < N && (block_k * 16 + idx) < K) {
              lB[ blIndex + secVal] = B[BIndex + BinitOffset + secVal * ldb];
            } else {
              lB[blIndex + secVal] = 0;
            }

            if( (gidx*64 + idx + secVal) < M && (block_k * 16 + idy) < K) {
              lA[ alIndex + secVal] = A[AIndex + secVal + AinitOffset];
            } else {
              lA[ alIndex + secVal] = 0;
            }
          }
      }
      else
      {
          lB[blIndex] = B[BIndex + BinitOffset];
          lB[blIndex + 16] = B[BIndex + BinitOffset +  16 * ldb];
          lB[blIndex + 32] = B[BIndex + BinitOffset +  32 * ldb];
          lB[blIndex + 48] = B[BIndex + BinitOffset +  48 * ldb];
          lA[alIndex] = A[AIndex + AinitOffset ];
          lA[alIndex + 16] = A[AIndex + 16 + AinitOffset];
          lA[alIndex + 32] = A[AIndex + 32 + AinitOffset];
          lA[alIndex + 48] = A[AIndex + 48 + AinitOffset];
      }

      tidx.barrier.wait();

      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < 16; iter+=8) {
        M4x4;
        M4x4;
        M4x4;
        M4x4;
        M4x4;
        M4x4;
        M4x4;
        M4x4;
      }

      AinitOffset += lda << 4;
      BinitOffset += 16;

    } while (++block_k < (K_R >> 4));

      tidx.barrier.wait();
    if(gidx == M_block-1 || gidy == N_block-1)
    {
        if((gidx * 64 + idx + CinitOffset)  < M && (gidy * 64 + idy + 0 * 16) < N)
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        if((gidx * 64 + idx + CinitOffset)  < M && (gidy * 64 + idy + 1 * 16) < N)
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        if((gidx * 64 + idx + CinitOffset)  < M && (gidy * 64 + idy + 2 * 16) < N)
        C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[0][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
        if((gidx * 64 + idx + CinitOffset)  < M && (gidy * 64 + idy + 3 * 16) < N)
        C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[0][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
        CinitOffset+=16;
        if((gidx * 64 + idx + CinitOffset)  < M && (gidy * 64 + idy + 0 * 16) < N)
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        if((gidx * 64 + idx + CinitOffset)  < M && (gidy * 64 + idy + 1 * 16) < N)
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        if((gidx * 64 + idx + CinitOffset)  < M && (gidy * 64 + idy + 2 * 16) < N)
        C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[1][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
        if((gidx * 64 + idx + CinitOffset)  < M && (gidy * 64 + idy + 3 * 16) < N)
        C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[1][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
        CinitOffset+=16;
        if((gidx * 64 + idx + CinitOffset)  < M && (gidy * 64 + idy + 0 * 16) < N)
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[2][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        if((gidx * 64 + idx + CinitOffset)  < M && (gidy * 64 + idy + 1 * 16) < N)
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[2][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        if((gidx * 64 + idx + CinitOffset)  < M && (gidy * 64 + idy + 2 * 16) < N)
        C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[2][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
        if((gidx * 64 + idx + CinitOffset)  < M && (gidy * 64 + idy + 3 * 16) < N)
        C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[2][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
        CinitOffset+=16;
        if((gidx * 64 + idx + CinitOffset)  < M && (gidy * 64 + idy + 0 * 16) < N)
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[3][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        if((gidx * 64 + idx + CinitOffset)  < M && (gidy * 64 + idy + 1 * 16) < N)
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[3][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        if((gidx * 64 + idx + CinitOffset)  < M && (gidy * 64 + idy + 2 * 16) < N)
        C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[3][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
        if((gidx * 64 + idx + CinitOffset)  < M && (gidy * 64 + idy + 3 * 16) < N)
        C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[3][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
    }
    else
    {
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[0][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
        C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[0][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
        CinitOffset+=16;
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[1][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
        C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[1][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
        CinitOffset+=16;
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[2][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[2][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[2][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
        C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[2][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
        CinitOffset+=16;
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[3][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[3][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[3][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
        C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[3][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
    }

  });
  return HCBLAS_SUCCEEDS;
}

hcblasStatus gemm_NoTransAB_MICRO_NBK_M_N_K_TS16XMTS6(hc::accelerator_view accl_view,
                                                const double *A, long aOffset,
                                                const double *B, long bOffset,
                                                double *C, long cOffset,
                                                int M, int N, int K, int lda, int ldb, int ldc,
                                                double alpha, double beta) {
  int M_ = (M-1)/6 + 1;
  int N_ = (N-1)/6 + 1;
  int N_R = (N_ + 15) & ~15;
  int M_R = (M_ + 15) & ~15;
  int K_R = (K + 15) & ~15;
  hc::extent<2> grdExt(N_R, M_R);
  hc::tiled_extent<2> t_ext = grdExt.tile(16, 16);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2> tidx) __attribute__((hc, cpu)) {
    double rC[6][6] = {{(double)0}};
    double rA[1][6];
    double rB[1][6];
    tile_static double lA[1552];
    tile_static double lB[1552];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int block_k = 0;
    int alIndex = (idy * 97) + idx;
    int blIndex = (idx * 97) + idy;
    int AIndex = aOffset + (gidx * 96) + idx + (idy * lda);
    int BIndex = bOffset + ((gidy * 96) + idy)*ldb + idx;
    long CIndex = cOffset + (gidx * 96) + idx + (((gidy * 96) + idy) * ldc);
    long AinitOffset = 0;
    long BinitOffset = 0;
    long CinitOffset = 0;
    int N_block = N_R >> 4;
    int M_block = M_R >> 4;
    int K_block = K_R >> 4;
    do {

      tidx.barrier.wait();

      if(gidx == M_block-1 || gidy == N_block-1 || block_k == K_block-1)
      {
          for(int sec = 0; sec < 6; ++sec) {
            int secVal = sec << 4;

            if( (gidy*96 + idy + secVal) < N && (block_k * 16 + idx) < K) {
              lB[ blIndex + secVal] = B[BIndex + BinitOffset + secVal * ldb];
            } else {
              lB[blIndex + secVal] = 0;
            }

            if( (gidx*96 + idx + secVal) < M && (block_k * 16 + idy) < K) {
              lA[ alIndex + secVal] = A[AIndex + secVal + AinitOffset];
            } else {
              lA[ alIndex + secVal] = 0;
            }
          }
      }
      else
      {
          lB[blIndex] = B[BIndex + BinitOffset];
          lB[blIndex + 16] = B[BIndex + BinitOffset +  16 * ldb];
          lB[blIndex + 32] = B[BIndex + BinitOffset +  32 * ldb];
          lB[blIndex + 48] = B[BIndex + BinitOffset +  48 * ldb];
          lB[blIndex + 64] = B[BIndex + BinitOffset +  64 * ldb];
          lB[blIndex + 80] = B[BIndex + BinitOffset +  80 * ldb];
          lA[alIndex] = A[AIndex + AinitOffset ];
          lA[alIndex + 16] = A[AIndex + 16 + AinitOffset];
          lA[alIndex + 32] = A[AIndex + 32 + AinitOffset];
          lA[alIndex + 48] = A[AIndex + 48 + AinitOffset];
          lA[alIndex + 64] = A[AIndex + 64 + AinitOffset];
          lA[alIndex + 80] = A[AIndex + 80 + AinitOffset];
      }

      tidx.barrier.wait();

      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < 16; iter+=8) {
        M6x6;
        M6x6;
        M6x6;
        M6x6;
        M6x6;
        M6x6;
        M6x6;
        M6x6;
      }

      AinitOffset += lda << 4;
      BinitOffset += 16;

    } while (++block_k < (K_R >> 4));

      tidx.barrier.wait();
    if(gidx == M_block-1 || gidy == N_block-1)
    {
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 0 * 16) < N)
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 1 * 16) < N)
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 2 * 16) < N)
        C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[0][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 3 * 16) < N)
        C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[0][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 4 * 16) < N)
        C[CIndex + CinitOffset + 64 * ldc] = alpha*rC[0][4] + beta * C[CIndex + CinitOffset + 64 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 5 * 16) < N)
        C[CIndex + CinitOffset + 80 * ldc] = alpha*rC[0][5] + beta * C[CIndex + CinitOffset + 80 * ldc] ;
        CinitOffset+=16;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 0 * 16) < N)
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 1 * 16) < N)
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 2 * 16) < N)
        C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[1][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 3 * 16) < N)
        C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[1][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 4 * 16) < N)
        C[CIndex + CinitOffset + 64 * ldc] = alpha*rC[1][4] + beta * C[CIndex + CinitOffset + 64 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 5 * 16) < N)
        C[CIndex + CinitOffset + 80 * ldc] = alpha*rC[1][5] + beta * C[CIndex + CinitOffset + 80 * ldc] ;
        CinitOffset+=16;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 0 * 16) < N)
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[2][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 1 * 16) < N)
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[2][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 2 * 16) < N)
        C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[2][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 3 * 16) < N)
        C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[2][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 4 * 16) < N)
        C[CIndex + CinitOffset + 64 * ldc] = alpha*rC[2][4] + beta * C[CIndex + CinitOffset + 64 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 5 * 16) < N)
        C[CIndex + CinitOffset + 80 * ldc] = alpha*rC[2][5] + beta * C[CIndex + CinitOffset + 80 * ldc] ;
        CinitOffset+=16;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 0 * 16) < N)
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[3][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 1 * 16) < N)
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[3][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 2 * 16) < N)
        C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[3][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 3 * 16) < N)
        C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[3][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 4 * 16) < N)
        C[CIndex + CinitOffset + 64 * ldc] = alpha*rC[3][4] + beta * C[CIndex + CinitOffset + 64 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 5 * 16) < N)
        C[CIndex + CinitOffset + 80 * ldc] = alpha*rC[3][5] + beta * C[CIndex + CinitOffset + 80 * ldc] ;
        CinitOffset+=16;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 0 * 16) < N)
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[4][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 1 * 16) < N)
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[4][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 2 * 16) < N)
        C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[4][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 3 * 16) < N)
        C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[4][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 4 * 16) < N)
        C[CIndex + CinitOffset + 64 * ldc] = alpha*rC[4][4] + beta * C[CIndex + CinitOffset + 64 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 5 * 16) < N)
        C[CIndex + CinitOffset + 80 * ldc] = alpha*rC[4][5] + beta * C[CIndex + CinitOffset + 80 * ldc] ;
        CinitOffset+=16;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 0 * 16) < N)
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[5][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 1 * 16) < N)
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[5][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 2 * 16) < N)
        C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[5][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 3 * 16) < N)
        C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[5][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 4 * 16) < N)
        C[CIndex + CinitOffset + 64 * ldc] = alpha*rC[5][4] + beta * C[CIndex + CinitOffset + 64 * ldc] ;
        if((gidx * 96 + idx + CinitOffset)  < M && (gidy * 96 + idy + 5 * 16) < N)
        C[CIndex + CinitOffset + 80 * ldc] = alpha*rC[5][5] + beta * C[CIndex + CinitOffset + 80 * ldc] ;
    }
    else
    {
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[0][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
        C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[0][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
        C[CIndex + CinitOffset + 64 * ldc] = alpha*rC[0][4] + beta * C[CIndex + CinitOffset + 64 * ldc] ;
        C[CIndex + CinitOffset + 80 * ldc] = alpha*rC[0][5] + beta * C[CIndex + CinitOffset + 80 * ldc] ;
        CinitOffset+=16;
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[1][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
        C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[1][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
        C[CIndex + CinitOffset + 64 * ldc] = alpha*rC[1][4] + beta * C[CIndex + CinitOffset + 64 * ldc] ;
        C[CIndex + CinitOffset + 80 * ldc] = alpha*rC[1][5] + beta * C[CIndex + CinitOffset + 80 * ldc] ;
        CinitOffset+=16;
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[2][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[2][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[2][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
        C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[2][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
        C[CIndex + CinitOffset + 64 * ldc] = alpha*rC[2][4] + beta * C[CIndex + CinitOffset + 64 * ldc] ;
        C[CIndex + CinitOffset + 80 * ldc] = alpha*rC[2][5] + beta * C[CIndex + CinitOffset + 80 * ldc] ;
        CinitOffset+=16;
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[3][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[3][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[3][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
        C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[3][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
        C[CIndex + CinitOffset + 64 * ldc] = alpha*rC[3][4] + beta * C[CIndex + CinitOffset + 64 * ldc] ;
        C[CIndex + CinitOffset + 80 * ldc] = alpha*rC[3][5] + beta * C[CIndex + CinitOffset + 80 * ldc] ;
        CinitOffset+=16;
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[4][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[4][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[4][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
        C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[4][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
        C[CIndex + CinitOffset + 64 * ldc] = alpha*rC[4][4] + beta * C[CIndex + CinitOffset + 64 * ldc] ;
        C[CIndex + CinitOffset + 80 * ldc] = alpha*rC[4][5] + beta * C[CIndex + CinitOffset + 80 * ldc] ;
        CinitOffset+=16;
        C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[5][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
        C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[5][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
        C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[5][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
        C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[5][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
        C[CIndex + CinitOffset + 64 * ldc] = alpha*rC[5][4] + beta * C[CIndex + CinitOffset + 64 * ldc] ;
        C[CIndex + CinitOffset + 80 * ldc] = alpha*rC[5][5] + beta * C[CIndex + CinitOffset + 80 * ldc] ;
    }

  });
  return HCBLAS_SUCCEEDS;
}

hcblasStatus gemm_NoTransAB_MICRO_NBK_MX096_NX096_KX16_TS16XMTS6(hc::accelerator_view accl_view,
					        const double *A, long aOffset,
					        const double *B, long bOffset,
					        double *C, long cOffset,
					        int M, int N, int K, int lda, int ldb, int ldc,
					        double alpha, double beta) {
  int M_ = M/6;
  int N_ = N/6 ;
  hc::extent<2> grdExt((N_ + 15) & ~15, (M_ + 15) & ~15);
  hc::tiled_extent<2> t_ext = grdExt.tile(16, 16);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2> tidx) __attribute__((hc, cpu)) {
    double rC[6][6] = {{(double)0}};
    double rA[1][6];
    double rB[1][6];
    tile_static double lA[1552];
    tile_static double lB[1552];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int block_k = K >> 4;
    int alIndex = (idy * 97) + idx;
    int blIndex = (idx * 97) + idy;
    int AIndex = aOffset + (gidx * 96) + idx + (idy * lda);
    int BIndex = bOffset + ((gidy * 96) + idy)*ldb + idx;
    long CIndex = cOffset + (gidx * 96) + idx + (((gidy * 96) + idy) * ldc);
    long AinitOffset = 0;
    long BinitOffset = 0;
    long CinitOffset = 0;
    do {

      tidx.barrier.wait();
      lB[blIndex] = B[BIndex + BinitOffset];
      lB[blIndex + 16] = B[BIndex + BinitOffset +  16 * ldb];
      lB[blIndex + 32] = B[BIndex + BinitOffset +  32 * ldb];
      lB[blIndex + 48] = B[BIndex + BinitOffset +  48 * ldb];
      lB[blIndex + 64] = B[BIndex + BinitOffset +  64 * ldb];
      lB[blIndex + 80] = B[BIndex + BinitOffset +  80 * ldb];
      lA[alIndex] = A[AIndex + AinitOffset ];
      lA[alIndex + 16] = A[AIndex + 16 + AinitOffset];
      lA[alIndex + 32] = A[AIndex + 32 + AinitOffset];
      lA[alIndex + 48] = A[AIndex + 48 + AinitOffset];
      lA[alIndex + 64] = A[AIndex + 64 + AinitOffset];
      lA[alIndex + 80] = A[AIndex + 80 + AinitOffset];

      tidx.barrier.wait();

      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < 16; iter+=8) {
        M6x6;
        M6x6;
        M6x6;
        M6x6;
        M6x6;
        M6x6;
        M6x6;
        M6x6;
      }

      AinitOffset += lda << 4;
      BinitOffset += 16;

    } while (--block_k > 0); // (((K + TILESIZE - 1) & ~(TILESIZE - 1)) / TILESIZE));

    C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[0][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
    C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[0][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
    C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[0][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
    C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[0][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
    C[CIndex + CinitOffset + 64 * ldc] = alpha*rC[0][4] + beta * C[CIndex + CinitOffset + 64 * ldc] ;
    C[CIndex + CinitOffset + 80 * ldc] = alpha*rC[0][5] + beta * C[CIndex + CinitOffset + 80 * ldc] ;
    CinitOffset+=16;
    C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[1][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
    C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[1][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
    C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[1][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
    C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[1][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
    C[CIndex + CinitOffset + 64 * ldc] = alpha*rC[1][4] + beta * C[CIndex + CinitOffset + 64 * ldc] ;
    C[CIndex + CinitOffset + 80 * ldc] = alpha*rC[1][5] + beta * C[CIndex + CinitOffset + 80 * ldc] ;
    CinitOffset+=16;
    C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[2][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
    C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[2][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
    C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[2][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
    C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[2][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
    C[CIndex + CinitOffset + 64 * ldc] = alpha*rC[2][4] + beta * C[CIndex + CinitOffset + 64 * ldc] ;
    C[CIndex + CinitOffset + 80 * ldc] = alpha*rC[2][5] + beta * C[CIndex + CinitOffset + 80 * ldc] ;
    CinitOffset+=16;
    C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[3][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
    C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[3][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
    C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[3][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
    C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[3][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
    C[CIndex + CinitOffset + 64 * ldc] = alpha*rC[3][4] + beta * C[CIndex + CinitOffset + 64 * ldc] ;
    C[CIndex + CinitOffset + 80 * ldc] = alpha*rC[3][5] + beta * C[CIndex + CinitOffset + 80 * ldc] ;
    CinitOffset+=16;
    C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[4][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
    C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[4][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
    C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[4][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
    C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[4][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
    C[CIndex + CinitOffset + 64 * ldc] = alpha*rC[4][4] + beta * C[CIndex + CinitOffset + 64 * ldc] ;
    C[CIndex + CinitOffset + 80 * ldc] = alpha*rC[4][5] + beta * C[CIndex + CinitOffset + 80 * ldc] ;
    CinitOffset+=16;
    C[CIndex + CinitOffset + 0 * ldc] = alpha*rC[5][0] + beta * C[CIndex + CinitOffset + 0 * ldc] ;
    C[CIndex + CinitOffset + 16 * ldc] = alpha*rC[5][1] + beta * C[CIndex + CinitOffset + 16 * ldc] ;
    C[CIndex + CinitOffset + 32 * ldc] = alpha*rC[5][2] + beta * C[CIndex + CinitOffset + 32 * ldc] ;
    C[CIndex + CinitOffset + 48 * ldc] = alpha*rC[5][3] + beta * C[CIndex + CinitOffset + 48 * ldc] ;
    C[CIndex + CinitOffset + 64 * ldc] = alpha*rC[5][4] + beta * C[CIndex + CinitOffset + 64 * ldc] ;
    C[CIndex + CinitOffset + 80 * ldc] = alpha*rC[5][5] + beta * C[CIndex + CinitOffset + 80 * ldc] ;

  });
  return HCBLAS_SUCCEEDS;
}

hcblasStatus gemm_NoTransAB_STEP_NBK_M_N_K_TS16XMS4(hc::accelerator_view accl_view,
					     double *A, long aOffset,
					     double *B, long bOffset,
					     double *C, long cOffset,
					     int M, int N, int K, int lda, int ldb, int ldc,
					     double alpha, double beta) {
  int M_R = (M + 15) & ~(15);
  int N_R = (N + 15) & ~(15);
  int K_R = (K + 63) & ~(63);
  int M_blocks = M_R/16;
  int N_blocks = N_R/16;
  int K_blocks = K_R/64;
  hc::extent<2> grdExt(N_R, M_R);
  hc::tiled_extent<2> t_ext = grdExt.tile(16, 16);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {

   double rC[1][1];
   double rA[1][4];
   double rB[1][4];
   tile_static double lA[1088];
   tile_static double lB[1088];
   int gidx = tidx.tile[1];
   int gidy = tidx.tile[0];
   int idx = tidx.local[1];
   int idy = tidx.local[0];
   int block_k = 0;
   int i = 0;
   int alIndex = idx * 17 + idy;
   int blIndex = idy * 17 + idx;
   int AIndex = aOffset + (gidx * 16) + idx + (idy * lda);
   int BIndex = bOffset + ((gidy * 16) + idy)*ldb + idx;
   long CIndex = cOffset + (gidx * 16) + idx + (((gidy * 16) + idy) * ldc);
   long AinitOffset = 0;
   long BinitOffset = 0;

   do {

     tidx.barrier.wait();

     if ( gidx == M_blocks-1 || gidy == N_blocks-1 || block_k == K_blocks-1) {
       for (int sec = 0; sec < 4; sec++)  {

          int secVal = sec << 4;
          if (gidy * 16 + idy < N && (idx + block_k * 64 + secVal) < K)
            lB[blIndex + 272 * sec] = B[BIndex + BinitOffset + secVal];
          else
            lB[blIndex + 272 * sec] = 0;

          if (gidx * 16 + idx  < M  && (block_k * 64 + idy + secVal) < K)
            lA[alIndex + 272 * sec] = A[AIndex + AinitOffset + secVal * lda];
          else
            lA[alIndex + 272 * sec] = 0;

       }
     } else {
       lB[blIndex] = B[BIndex + BinitOffset];
       lB[blIndex + 272] = B[BIndex + BinitOffset + 16];
       lB[blIndex + 544] = B[BIndex + BinitOffset + 32]; 
       lB[blIndex + 816] = B[BIndex + BinitOffset + 48];
       lA[alIndex] = A[AIndex + AinitOffset];
       lA[alIndex + 272] = A[AIndex + AinitOffset + 16 * lda];
       lA[alIndex + 544] = A[AIndex + AinitOffset + 32 * lda];
       lA[alIndex + 816] = A[AIndex + AinitOffset + 48 * lda];
     }

     tidx.barrier.wait();

     int offA = idx * 17;
     int offB = idy * 17;

     for (int iter = 0; iter < 16; ++iter) {
       MSS4X4;
     }

     AinitOffset += lda << 6;
     BinitOffset += 64;

   } while (++block_k < K_blocks); // (((K + TILESIZE - 1) & ~(TILESIZE - 1)) / TILESIZE));


  if (gidx == M_blocks-1 || gidy == N_blocks-1) {
    if( gidy * 16 + idy < N && gidx * 16 + idx  < M)
       C[CIndex] = alpha*rC[0][0] + beta * C[CIndex] ;
    }
    else  C[CIndex] = alpha*rC[0][0] + beta * C[CIndex];

  });

  return HCBLAS_SUCCEEDS;
}

hcblasStatus gemm_NoTransAB_STEP_NBK_M_N_K_TS16XMS6(hc::accelerator_view accl_view,
					     double *A, long aOffset,
					     double *B, long bOffset,
					     double *C, long cOffset,
					     int M, int N, int K, int lda, int ldb, int ldc,
					     double alpha, double beta) {

  int M_R = (M + 15) & ~(15);
  int N_R = (N + 15) & ~(15);
  int M_blocks = M_R/16;
  int N_blocks = N_R/16;
  int K_blocks = (K -1) / 96 + 1;
  hc::extent<2> grdExt(N_R, M_R);
  hc::tiled_extent<2> t_ext = grdExt.tile(16, 16);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {

   double rC[1][1];
   double rA[1][6];
   double rB[1][6];
   tile_static double lA[1632];
   tile_static double lB[1632];
   int gidx = tidx.tile[1];
   int gidy = tidx.tile[0];
   int idx = tidx.local[1];
   int idy = tidx.local[0];
   int block_k = 0;
   int i = 0;
   int alIndex = idx * 17 + idy;
   int blIndex = idy * 17 + idx;
   int AIndex = aOffset + (gidx * 16) + idx + (idy * lda);
   int BIndex = bOffset + ((gidy * 16) + idy)*ldb + idx;
   long CIndex = cOffset + (gidx * 16) + idx + (((gidy * 16) + idy) * ldc);
   long AinitOffset = 0;
   long BinitOffset = 0;

   do {

     tidx.barrier.wait();

     if ( gidx == M_blocks-1 || gidy == N_blocks-1 || block_k == K_blocks-1) {
       for (int sec = 0; sec < 6; sec++)  {

          int secVal = sec << 4;
          if (gidy * 16 + idy < N && (idx + block_k * 96 + secVal) < K)
            lB[blIndex + 272 * sec] = B[BIndex + BinitOffset + secVal];
          else
            lB[blIndex + 272 * sec] = 0;

          if (gidx * 16 + idx  < M  && (block_k * 96 + idy + secVal) < K)
            lA[alIndex + 272 * sec] = A[AIndex + AinitOffset + secVal * lda];
          else
            lA[alIndex + 272 * sec] = 0;

       }
     } else {
       lB[blIndex] = B[BIndex + BinitOffset];
       lB[blIndex + 272] = B[BIndex + BinitOffset + 16];
       lB[blIndex + 544] = B[BIndex + BinitOffset + 32]; 
       lB[blIndex + 816] = B[BIndex + BinitOffset + 48];
       lB[blIndex + 1088] = B[BIndex + BinitOffset + 64];
       lB[blIndex + 1360] = B[BIndex + BinitOffset + 80];
       lA[alIndex] = A[AIndex + AinitOffset];
       lA[alIndex + 272] = A[AIndex + AinitOffset + 16 * lda];
       lA[alIndex + 544] = A[AIndex + AinitOffset + 32 * lda];
       lA[alIndex + 816] = A[AIndex + AinitOffset + 48 * lda];
       lA[alIndex + 1088] = A[AIndex + AinitOffset + 64 * lda];
       lA[alIndex + 1360] = A[AIndex + AinitOffset + 80 * lda];
     }

     tidx.barrier.wait();

     int offA = idx * 17;
     int offB = idy * 17;

     for (int iter = 0; iter < 16; ++iter) {
       MSS6X6;
     }

     AinitOffset += lda * 96;
     BinitOffset += 96;

   } while (++block_k < K_blocks); // (((K + TILESIZE - 1) & ~(TILESIZE - 1)) / TILESIZE));

  if (gidx == M_blocks-1 || gidy == N_blocks-1) {
    if( gidy * 16 + idy < N && gidx * 16 + idx  < M)
       C[CIndex] = alpha*rC[0][0] + beta * C[CIndex] ;
    }
    else  C[CIndex] = alpha*rC[0][0] + beta * C[CIndex];

  });

  return HCBLAS_SUCCEEDS;
}
hcblasStatus gemm_NoTransAB_STEP_NBK_Mx16_NX16_KX64_TS16XMS4(hc::accelerator_view accl_view,
					     double *A, long aOffset,
					     double *B, long bOffset,
					     double *C, long cOffset,
					     int M, int N, int K, int lda, int ldb, int ldc,
					     double alpha, double beta) {
  hc::extent<2> grdExt((N + 15) & ~15, (M + 15) & ~15);
  hc::tiled_extent<2> t_ext = grdExt.tile(16, 16);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {

   double rC[1][1];
   double rA[1][4];
   double rB[1][4];
   tile_static double lA[1088];
   tile_static double lB[1088];
   int gidx = tidx.tile[1];
   int gidy = tidx.tile[0];
   int idx = tidx.local[1];
   int idy = tidx.local[0];
   int block_k = K >> 6;
   int i = 0;
   int alIndex = idx * 17 + idy;
   int blIndex = idy * 17 + idx;
   int AIndex = aOffset + (gidx * 16) + idx + (idy * lda);
   int BIndex = bOffset + ((gidy * 16) + idy)*ldb + idx;
   long CIndex = cOffset + (gidx * 16) + idx + (((gidy * 16) + idy) * ldc);
   long AinitOffset = 0;
   long BinitOffset = 0;

   do {

     tidx.barrier.wait();
     lB[blIndex] = B[BIndex + BinitOffset];
     lB[blIndex + 272] = B[BIndex + BinitOffset + 16];
     lB[blIndex + 544] = B[BIndex + BinitOffset + 32];
     lB[blIndex + 816] = B[BIndex + BinitOffset + 48];
     lA[alIndex] = A[AIndex + AinitOffset];
     lA[alIndex + 272] = A[AIndex + AinitOffset + 16 * lda];
     lA[alIndex + 544] = A[AIndex + AinitOffset + 32 * lda];
     lA[alIndex + 816] = A[AIndex + AinitOffset + 48 * lda];

     tidx.barrier.wait();

     int offA = idx * 17;
     int offB = idy * 17;

     for (int iter = 0; iter < 16; ++iter) {
       MSS4X4;
     }

     AinitOffset += lda << 6;
     BinitOffset += 64;

   } while (--block_k > 0); // (((K + TILESIZE - 1) & ~(TILESIZE - 1)) / TILESIZE));

  C[CIndex] = alpha*rC[0][0] + beta * C[CIndex] ;

  });

  return HCBLAS_SUCCEEDS;
}

hcblasStatus gemm_NoTransAB_STEP_NBK_Mx16_NX16_KX96_TS16XMS6(hc::accelerator_view accl_view,
					     double *A, long aOffset,
					     double *B, long bOffset,
					     double *C, long cOffset,
					     int M, int N, int K, int lda, int ldb, int ldc,
					     double alpha, double beta) {
  hc::extent<2> grdExt((N + 15) & ~15, (M + 15) & ~15);
  hc::tiled_extent<2> t_ext = grdExt.tile(16, 16);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {

   double rC[1][1];
   double rA[1][6];
   double rB[1][6];
   tile_static double lA[1632];
   tile_static double lB[1632];
   int gidx = tidx.tile[1];
   int gidy = tidx.tile[0];
   int idx = tidx.local[1];
   int idy = tidx.local[0];
   int block_k = K / 96;
   int i = 0;
   int alIndex = idx * 17 + idy;
   int blIndex = idy * 17 + idx;
   int AIndex = aOffset + (gidx * 16) + idx + (idy * lda);
   int BIndex = bOffset + ((gidy * 16) + idy)*ldb + idx;
   long CIndex = cOffset + (gidx * 16) + idx + (((gidy * 16) + idy) * ldc);
   long AinitOffset = 0;
   long BinitOffset = 0;

   do {

     tidx.barrier.wait();
     lB[blIndex] = B[BIndex + BinitOffset];
     lB[blIndex + 272] = B[BIndex + BinitOffset + 16];
     lB[blIndex + 544] = B[BIndex + BinitOffset + 32];
     lB[blIndex + 816] = B[BIndex + BinitOffset + 48];
     lB[blIndex + 1088] = B[BIndex + BinitOffset + 64];
     lB[blIndex + 1360] = B[BIndex + BinitOffset + 80];
     lA[alIndex] = A[AIndex + AinitOffset];
     lA[alIndex + 272] = A[AIndex + AinitOffset + 16 * lda];
     lA[alIndex + 544] = A[AIndex + AinitOffset + 32 * lda];
     lA[alIndex + 816] = A[AIndex + AinitOffset + 48 * lda];
     lA[alIndex + 1088] = A[AIndex + AinitOffset + 64 * lda];
     lA[alIndex + 1360] = A[AIndex + AinitOffset + 80 * lda];

     tidx.barrier.wait();

     int offA = idx * 17;
     int offB = idy * 17;

     for (int iter = 0; iter < 16; ++iter) {
       MSS6X6;
     }

     AinitOffset += lda * 96;
     BinitOffset += 96;

   } while (--block_k > 0); // (((K + TILESIZE - 1) & ~(TILESIZE - 1)) / TILESIZE));

  C[CIndex] = alpha*rC[0][0] + beta * C[CIndex] ;

  });

  return HCBLAS_SUCCEEDS;
}
