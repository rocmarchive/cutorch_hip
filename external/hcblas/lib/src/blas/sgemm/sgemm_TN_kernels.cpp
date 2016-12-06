#include "sgemm_array_kernels.h"
#include <cmath>
#include "hc_math.hpp"

hcblasStatus gemm_NoTransB_MICRO_NBK_Mini_Batch_M128_N128_K16_TS16XMTS2_MB2(hc::accelerator_view accl_view,
					       const float *A, long aOffset,
					       const float *B, long bOffset,
					       float *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       float alpha, float beta) {
  int M_ = (M-1)/4 + 1;
  int N_ = (N-1)/4 + 1;
  int N_R = (N_ + 15) & ~15;
  int M_R = (M_ + 15) & ~15;
  int K_R = (K + 15) & ~15;
  hc::extent<2> grdExt(N_R, M_R);
  hc::tiled_extent<2> t_ext = grdExt.tile(16, 16);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2> tidx) __attribute__((hc, cpu)) {
    float rC[4][4] = {{(float)0}};
    float rA[1][4];
    float rB[1][4];
    tile_static float lA[16 * 32 * 2 + 16];
    tile_static float lB[16 * 32 * 2 + 16];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int block_k = 0;
    int lIndex = (idx * (32 * 2 + 1)) + idy * 2;
    int AIndex = aOffset + (gidx * 32 * 2 + idy * 2) * lda + idx;
    int BIndex = bOffset + (gidy * 32 * 2 + idy * 2) * ldb + idx;
    long CIndex = cOffset + (gidx * 32 * 2) + idx * 2 + (((gidy * 32 * 2) + idy * 2) * ldc);
    long CinitOffset = 0;
    do {

      tidx.barrier.wait();

      lB[lIndex + 0 * 16 * 2 + 0] = B[BIndex + block_k * 16 + (0 * 16 * 2 + 0) * ldb];
      lB[lIndex + 0 * 16 * 2 + 1] = B[BIndex + block_k * 16 + (0 * 16 * 2 + 1) * ldb];
      lB[lIndex + 1 * 16 * 2 + 0] = B[BIndex + block_k * 16 + (1 * 16 * 2 + 0) * ldb];
      lB[lIndex + 1 * 16 * 2 + 1] = B[BIndex + block_k * 16 + (1 * 16 * 2 + 1) * ldb];
      lA[lIndex + 0 * 16 * 2 + 0] = A[AIndex + block_k * 16 + (0 * 16 * 2 + 0) * lda];
      lA[lIndex + 0 * 16 * 2 + 1] = A[AIndex + block_k * 16 + (0 * 16 * 2 + 1) * lda];
      lA[lIndex + 1 * 16 * 2 + 0] = A[AIndex + block_k * 16 + (1 * 16 * 2 + 0) * lda];
      lA[lIndex + 1 * 16 * 2 + 1] = A[AIndex + block_k * 16 + (1 * 16 * 2 + 1) * lda];

      tidx.barrier.wait();

      int offA = idx * 2;
      int offB = idy * 2;

      for (int iter = 0; iter < 16; iter++) {
        M2x2_MB;
      }

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

hcblasStatus gemm_NoTransB_MICRO_NBK_Mini_Batch_M128_N128_K16_TS16XMTS4_MB2(hc::accelerator_view accl_view,
					       const float *A, long aOffset,
					       const float *B, long bOffset,
					       float *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       float alpha, float beta) {
  int M_ = M >> 3;
  int N_ = N >> 3;
  int N_R = (N_ + 15) & ~15;
  int M_R = (M_ + 15) & ~15;
  int K_R = (K + 15) & ~15;
  hc::extent<2> grdExt(N_R, M_R);
  hc::tiled_extent<2> t_ext = grdExt.tile(16, 16);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {
    float rC[8][8] = {{(float)0}};
    float rA[1][8];
    float rB[1][8];
    tile_static float lA[16 * 64 * 2 + 16];
    tile_static float lB[16 * 64 * 2 + 16];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int block_k = 0;
    int lIndex = (idx * (64 * 2 + 1)) + idy * 2;
    int AIndex = aOffset + (gidx * 64 * 2 + idy * 2) * lda + idx;
    int BIndex = bOffset + (gidy * 64 * 2 + idy * 2) * ldb + idx;
    long CIndex = cOffset + (gidx * 64 * 2) + idx * 2 + (((gidy * 64 * 2) + idy * 2) * ldc);
    long CinitOffset = 0;

    do {
      tidx.barrier.wait();

      lB[lIndex + 0 * 16 * 2 + 0] = B[BIndex + block_k * 16 + (0 * 16 * 2 + 0) * ldb];
      lB[lIndex + 0 * 16 * 2 + 1] = B[BIndex + block_k * 16 + (0 * 16 * 2 + 1) * ldb];
      lB[lIndex + 1 * 16 * 2 + 0] = B[BIndex + block_k * 16 + (1 * 16 * 2 + 0) * ldb];
      lB[lIndex + 1 * 16 * 2 + 1] = B[BIndex + block_k * 16 + (1 * 16 * 2 + 1) * ldb];
      lB[lIndex + 2 * 16 * 2 + 0] = B[BIndex + block_k * 16 + (2 * 16 * 2 + 0) * ldb];
      lB[lIndex + 2 * 16 * 2 + 1] = B[BIndex + block_k * 16 + (2 * 16 * 2 + 1) * ldb];
      lB[lIndex + 3 * 16 * 2 + 0] = B[BIndex + block_k * 16 + (3 * 16 * 2 + 0) * ldb];
      lB[lIndex + 3 * 16 * 2 + 1] = B[BIndex + block_k * 16 + (3 * 16 * 2 + 1) * ldb];
      lA[lIndex + 0 * 16 * 2 + 0] = A[AIndex + block_k * 16 + (0 * 16 * 2 + 0) * lda];
      lA[lIndex + 0 * 16 * 2 + 1] = A[AIndex + block_k * 16 + (0 * 16 * 2 + 1) * lda];
      lA[lIndex + 1 * 16 * 2 + 0] = A[AIndex + block_k * 16 + (1 * 16 * 2 + 0) * lda];
      lA[lIndex + 1 * 16 * 2 + 1] = A[AIndex + block_k * 16 + (1 * 16 * 2 + 1) * lda];
      lA[lIndex + 2 * 16 * 2 + 0] = A[AIndex + block_k * 16 + (2 * 16 * 2 + 0) * lda];
      lA[lIndex + 2 * 16 * 2 + 1] = A[AIndex + block_k * 16 + (2 * 16 * 2 + 1) * lda];
      lA[lIndex + 3 * 16 * 2 + 0] = A[AIndex + block_k * 16 + (3 * 16 * 2 + 0) * lda];
      lA[lIndex + 3 * 16 * 2 + 1] = A[AIndex + block_k * 16 + (3 * 16 * 2 + 1) * lda];

      tidx.barrier.wait();
      int offA = idx * 2;
      int offB = idy * 2;

      for (int iter = 0; iter < 16; iter++) {
          M4x4_MB
      }

      tidx.barrier.wait();
    } while (++block_k < (K_R / 16));

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
#undef TILESIZE
#undef MICROTILESIZE
  return HCBLAS_SUCCEEDS;
}

hcblasStatus gemm_NoTransB_MICRO_NBK_Mini_Batch_M_N_K_TS16XMTS2_MB2(hc::accelerator_view accl_view,
                                                const float *A, long aOffset,
                                                const float *B, long bOffset,
                                                float *C, long cOffset,
                                                int M, int N, int K, int lda, int ldb, int ldc,
                                                float alpha, float beta) {
  int M_ = (M-1)/4 + 1;
  int N_ = (N-1)/4 + 1;
  int N_R = (N_ + 15) & ~15;
  int M_R = (M_ + 15) & ~15;
  int K_R = (K + 15) & ~15;
  hc::extent<2> grdExt(N_R, M_R);
  hc::tiled_extent<2> t_ext = grdExt.tile(16, 16);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2> tidx) __attribute__((hc, cpu)) {
    float rC[4][4] = {{(float)0}};
    float rA[1][4];
    float rB[1][4];
    tile_static float lA[16 * 32 * 2 + 16];
    tile_static float lB[16 * 32 * 2 + 16];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int block_k = 0;
    int alIndex = (idx * (32 * 2 + 1)) + idy * 2;
    int blIndex = (idx * (32 * 2 + 1)) + idy * 2;
    int AIndex = aOffset + ((gidx * 32 * 2) + idy * 2)*lda + idx;
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
 
            if( (gidx*32*2 + idy * 2 + secVal * 2 + 0) < M && (block_k * 16 + idx) < K) {
              lA[ alIndex + secVal * 2 + 0] = A[AIndex + (secVal * 2 + 0) * lda + AinitOffset];
            } else {
              lA[ alIndex + secVal * 2 + 0] = 0;
            }
            if( (gidx*32*2 + idy * 2 + secVal * 2 + 1) < M && (block_k * 16 + idx) < K) {
              lA[ alIndex + secVal * 2 + 1] = A[AIndex + (secVal * 2 + 1) * lda + AinitOffset];
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
          lA[alIndex + 0 * 2 + 0] = A[AIndex + (0 * 2 + 0) * lda + AinitOffset];
          lA[alIndex + 0 * 2 + 1] = A[AIndex + (0 * 2 + 1) * lda + AinitOffset];
          lA[alIndex + 16 * 2 + 0] = A[AIndex + (16 * 2 + 0) * lda + AinitOffset];
          lA[alIndex + 16 * 2 + 1] = A[AIndex + (16 * 2 + 1) * lda + AinitOffset];
      }

      tidx.barrier.wait();

      int offA = idx * 2;
      int offB = idy * 2;

      for (int iter = 0; iter < 16; iter++) {
        M2x2_MB;
      }

      AinitOffset += 16;
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

hcblasStatus gemm_NoTransB_MICRO_NBK_Mini_Batch_M_N_K_TS16XMTS4_MB2(hc::accelerator_view accl_view,
					        const float *A, long aOffset,
					        const float *B, long bOffset,
					        float *C, long cOffset,
					        int M, int N, int K, int lda, int ldb, int ldc,
					        float alpha, float beta) {
  int M_ = (M-1)/8 + 1;
  int N_ = (N-1)/8 + 1;
  int N_R = (N_ + 15) & ~15;
  int M_R = (M_ + 15) & ~15;
  int K_R = (K + 15) & ~15;
  hc::extent<2> grdExt(N_R, M_R);
  hc::tiled_extent<2> t_ext = grdExt.tile(16, 16);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2> tidx) __attribute__((hc, cpu)) {
    float rC[8][8] = {{(float)0}};
    float rA[1][8];
    float rB[1][8];
    tile_static float lA[16 * 64 * 2 + 16];
    tile_static float lB[16 * 64 * 2 + 16];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int block_k = 0;
    int alIndex = (idx * (64 * 2 + 1)) + idy * 2;
    int blIndex = (idx * (64 * 2 + 1)) + idy * 2;
    int AIndex = aOffset + ((gidx * 64 * 2) + idy * 2)*lda + idx;
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

            if( (gidx*64*2 + idy * 2 + secVal * 2 + 0) < M && (block_k * 16 + idx) < K) {
              lA[ alIndex + secVal * 2 + 0] = A[AIndex + (secVal * 2 + 0) * lda + AinitOffset];
            } else {
              lA[ alIndex + secVal * 2 + 0] = 0;
            }
            if( (gidx*64*2 + idy * 2 + secVal * 2 + 1) < M && (block_k * 16 + idx) < K) {
              lA[ alIndex + secVal * 2 + 1] = A[AIndex + (secVal * 2 + 1) * lda + AinitOffset];
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
          lA[alIndex + 0 * 2 + 0] = A[AIndex + (0 * 2 + 0) * lda + AinitOffset];
          lA[alIndex + 0 * 2 + 1] = A[AIndex + (0 * 2 + 1) * lda + AinitOffset];
          lA[alIndex + 16 * 2 + 0] = A[AIndex + (16 * 2 + 0) * lda + AinitOffset];
          lA[alIndex + 16 * 2 + 1] = A[AIndex + (16 * 2 + 1) * lda + AinitOffset];
          lA[alIndex + 32 * 2 + 0] = A[AIndex + (32 * 2 + 0) * lda + AinitOffset];
          lA[alIndex + 32 * 2 + 1] = A[AIndex + (32 * 2 + 1) * lda + AinitOffset];
          lA[alIndex + 48 * 2 + 0] = A[AIndex + (48 * 2 + 0) * lda + AinitOffset];
          lA[alIndex + 48 * 2 + 1] = A[AIndex + (48 * 2 + 1) * lda + AinitOffset];
      }

      tidx.barrier.wait();

      int offA = idx * 2;
      int offB = idy * 2;

      for (int iter = 0; iter < 16; iter+=1) {
        M4x4_MB;
      }

      AinitOffset += 16;
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

hcblasStatus gemm_NoTransB_MICRO_NBK_M064_N064_K064_TS16XMTS4(hc::accelerator_view accl_view,
					       const float *A, long aOffset,
					       const float *B, long bOffset,
					       float *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       float alpha, float beta) {
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
    float rC[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
    float rA[1][MICROTILESIZE];
    float rB[1][MICROTILESIZE];
    tile_static float lA[TOTMICROTILEPROD + TILESIZE];
    tile_static float lB[TOTMICROTILEPROD + TILESIZE];
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
					       const float *A, long aOffset,
					       const float *B, long bOffset,
					       float *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       float alpha, float beta) {
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
    float rC[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
    float rA[1][MICROTILESIZE];
    float rB[1][MICROTILESIZE];
    tile_static float lA[TOTMICROTILEPROD + TILESIZE];
    tile_static float lB[TOTMICROTILEPROD + TILESIZE];
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
					       const float *A, long aOffset,
					       const float *B, long bOffset,
					       float *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       float alpha, float beta) {
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
    float rC[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
    float rA[1][MICROTILESIZE];
    float rB[1][MICROTILESIZE];
    tile_static float lA[TOTMICROTILEPROD + TILESIZE];
    tile_static float lB[TOTMICROTILEPROD + TILESIZE];
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
					       const float *A, long aOffset,
					       const float *B, long bOffset,
					       float *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       float alpha, float beta) {
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
    float rC[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
    float rA[1][MICROTILESIZE];
    float rB[1][MICROTILESIZE];
    tile_static float lA[TOTMICROTILEPROD + TILESIZE];
    tile_static float lB[TOTMICROTILEPROD + TILESIZE];
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
					       const float *A, long aOffset,
					       const float *B, long bOffset,
					       float *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       float alpha, float beta) {
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
    float rC[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
    float rA[1][MICROTILESIZE];
    float rB[1][MICROTILESIZE];
    tile_static float lA[TOTMICROTILEPROD + TILESIZE];
    tile_static float lB[TOTMICROTILEPROD + TILESIZE];
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
