#include "sgemm_array_kernels.h"
#include "hc_math.hpp"

using namespace hc::fast_math;
/*
* SGEMM - NoTransAB case - Row major Access
* STEP with Non Bank Conflict Implementation
* TILESIZE = 8 STEPSIZE = 8
*/
hcblasStatus gemm_NoTransAB_rMajor_batch_STEP_NBK_TS8XSS8(hc::accelerator_view accl_view,
						          float *A, long aOffset, long A_batchOffset,
						          float *B, long bOffset, long B_batchOffset,
						          float *C, long cOffset, long C_batchOffset,
						          int M, int N, int K, int lda, int ldb, int ldc,	
						          float alpha, float beta, int batchSize) {
#define TILESIZE 8
#define STEPSIZE 8
  hc::extent<3> grdExt(batchSize, (M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int shiftFactor = hc::fast_math::log2(STEPSIZE);
    int elt = tidx.tile[0];
    float rC[1][1] = {{(float)0}};
    float rA[1][STEPTILERATIO];
    float rB[1][STEPTILERATIO];
    tile_static float lA[STEPTILEPROD + STEPSIZE];
    tile_static float lB[STEPTILEPROD + STEPSIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[2];
    int idx = tidx.local[1];
    int idy = tidx.local[2];
    int idt = TILESIZE * idy + idx;
    int idxT = idt % TILESIZE;
    int idyT = idt / TILESIZE;
    int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftFactor;
    int i = 0;

    do {
      tidx.barrier.wait();

      for(int sec = 0; sec < STEPSIZE / TILESIZE; sec++ ) {
        if(gidy * TILESIZE + idxT < N && i * STEPSIZE + idyT + (sec * TILESIZE) < K) {
          lB[((idxT + sec * TILESIZE) * BANKTILESIZE) + idyT] = B[bOffset + B_batchOffset * elt + gidy * TILESIZE + idxT + ((idyT + (sec * TILESIZE)) * ldb) + i * (ldb << shiftFactor)];
        } else {
          lB[((idxT + sec * TILESIZE) * BANKTILESIZE) + idyT] = 0;
        }

        if(gidx * TILESIZE + idxT < M && i * STEPSIZE + idyT + (sec * TILESIZE) < K) {
          lA[(sec * BANKNUMTILEELMTS) + idyT + idxT * BANKTILESIZE] = A[aOffset  + A_batchOffset * elt + (gidx * TILESIZE + idxT) * lda + idyT + i * STEPSIZE + (sec * TILESIZE)];
        } else {
          lA[(sec * BANKNUMTILEELMTS ) + idyT + idxT * BANKTILESIZE] = 0;
        }
      }

      tidx.barrier.wait();
      int offA = idx * BANKTILESIZE;
      int offB = idy * BANKTILESIZE;

      for (int iter = 0; iter < TILESIZE; ++iter) {
        MS1x1_NOBANK(1);
      }

      i++;
    } while (--block_k > 0);

    tidx.barrier.wait();

    if(gidx * TILESIZE + idx < M && gidy * TILESIZE + idy < N) {
      long C_index = cOffset + C_batchOffset * elt + (gidx * TILESIZE + idx) * ldc + (gidy * TILESIZE + idy);
      C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
      C[C_index] = alpha * rC[0][0] + beta * C[C_index];
    }
  });
#undef TILESIZE
#undef STEPSIZE
  return HCBLAS_SUCCEEDS;
}
/*
* SGEMM - NoTransAB case - Row major Access
* STEP with Non Bank Conflict Implementation
* TILESIZE = 16 STEPSIZE = 16
*/

hcblasStatus gemm_NoTransAB_rMajor_batch_STEP_NBK_TS16XSS16(hc::accelerator_view accl_view,
							    float *A, long aOffset, long A_batchOffset,
							    float *B, long bOffset, long B_batchOffset,
							    float *C, long cOffset, long C_batchOffset,
							    int M, int N, int K, int lda, int ldb, int ldc,
							    float alpha, float beta, int batchSize) {
#define TILESIZE 16
#define STEPSIZE 16
  hc::extent<3> grdExt(batchSize, (M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int elt = tidx.tile[0];
    int shiftFactor = hc::fast_math::log2(STEPSIZE);
    float rC[1][1] = {{(float)0}};
    float rA[1][STEPTILERATIO];
    float rB[1][STEPTILERATIO];
    tile_static float lA[STEPTILEPROD + STEPSIZE];
    tile_static float lB[STEPTILEPROD + STEPSIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[2];
    int idx = tidx.local[1];
    int idy = tidx.local[2];
    int idt = TILESIZE * idy + idx;
    int idxT = idt % TILESIZE;
    int idyT = idt / TILESIZE;
    int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftFactor;
    int i = 0;

    do {
      tidx.barrier.wait();

      for(int sec = 0; sec < STEPSIZE / TILESIZE; sec++ ) {
        if(gidy * TILESIZE + idxT < N && i * STEPSIZE + idyT + (sec * TILESIZE) < K) {
          lB[((idxT + sec * TILESIZE) * BANKTILESIZE) + idyT] = B[bOffset + B_batchOffset * elt + gidy * TILESIZE + idxT + ((idyT + (sec * TILESIZE)) * ldb) + i * (ldb << shiftFactor)];
        } else {
          lB[((idxT + sec * TILESIZE) * BANKTILESIZE) + idyT] = 0;
        }

        if(gidx * TILESIZE + idxT < M && i * STEPSIZE + idyT + (sec * TILESIZE) < K) {
          lA[(sec * BANKNUMTILEELMTS) + idyT + idxT * BANKTILESIZE] = A[aOffset  + A_batchOffset * elt + (gidx * TILESIZE + idxT) * lda + idyT + i * STEPSIZE + (sec * TILESIZE)];
        } else {
          lA[(sec * BANKNUMTILEELMTS ) + idyT + idxT * BANKTILESIZE] = 0;
        }
      }

      tidx.barrier.wait();
      int offA = idx * BANKTILESIZE;
      int offB = idy * BANKTILESIZE;

      for (int iter = 0; iter < TILESIZE; ++iter) {
        MS1x1_NOBANK(1);
      }

      i++;
    } while (--block_k > 0);

    tidx.barrier.wait();

    if(gidx * TILESIZE + idx < M && gidy * TILESIZE + idy < N) {
      long C_index = cOffset + C_batchOffset * elt + (gidx * TILESIZE + idx) * ldc + (gidy * TILESIZE + idy);
      C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
      C[C_index] = alpha * rC[0][0] + beta * C[C_index];
    }
  });
#undef TILESIZE
#undef STEPSIZE
  return HCBLAS_SUCCEEDS;
}

/*
* SGEMM - NoTransAB case - Row major Access
* SUBMICROTILE Implementation
* TILESIZE = 16 MICROTILESIZE = 2
*/
hcblasStatus gemm_NoTransAB_rMajor_batch_MICRO_TS16XMTS2(hc::accelerator_view accl_view,
						         float *A, long aOffset, long A_batchOffset,
						         float *B, long bOffset, long B_batchOffset,
						         float *C, long cOffset, long C_batchOffset,
						         int M, int N, int K, int lda, int ldb, int ldc,
						         float alpha, float beta, int batchSize) {
#define TILESIZE 16
#define MICROTILESIZE 2
  int M_ = hc::fast_math::fmax(1, (M / MICROTILESIZE + 1));
  int N_ = hc::fast_math::fmax(1, (N / MICROTILESIZE + 1));
  hc::extent<3> grdExt(batchSize, (M_  + (TILESIZE - 1)) & ~(TILESIZE - 1), (N_ + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int elt = tidx.tile[0];
    float rC[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
    float rA[1][MICROTILESIZE];
    float rB[1][MICROTILESIZE];
    tile_static float lA[TILESIZE * TILESIZE * MICROTILESIZE];
    tile_static float lB[TILESIZE * TILESIZE * MICROTILESIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[2];
    int idx = tidx.local[1];
    int idy = tidx.local[2];
    int idt = TILESIZE * idy + idx;
    int idxT = idt % TILESIZE;
    int idyT = idt / TILESIZE;
    int block_k = 0;

    do {
      tidx.barrier.wait();

      for(int sec = 0; sec < MICROTILESIZE; ++sec) {
        if(gidy * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < N && block_k * TILESIZE + idyT < K) {
          lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = B[bOffset + B_batchOffset * elt + (gidy * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE) + idyT * ldb + block_k * (ldb * TILESIZE)];
        } else {
          lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
        }

        if(gidx * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < M && block_k * TILESIZE + idyT < K) {
          lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = A[aOffset + A_batchOffset * elt + (gidx * TILESIZE * MICROTILESIZE + idxT + sec * TILESIZE) * lda +  idyT + block_k * TILESIZE];
        } else {
          lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
        }
      }

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < TILESIZE; ++iter) {
        MTS;
      }

      tidx.barrier.wait();
    } while (++block_k < (((K + TILESIZE - 1) & ~(TILESIZE - 1)) / TILESIZE));

    int xIndex = (gidx * TILESIZE * MICROTILESIZE + idx) * ldc;
    int yIndex = (gidy * TILESIZE * MICROTILESIZE + idy);

    for( int row = 0; row < MICROTILESIZE; row++) {
      for( int col = 0; col < MICROTILESIZE ; col++) {
        if((xIndex / ldc) + (TILESIZE * col) < M && (yIndex) + (TILESIZE * row) < N) {
          long C_index = cOffset + C_batchOffset * elt + (xIndex + (TILESIZE * col) * N) + yIndex + (TILESIZE * row);
          C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
          C[C_index] = alpha * rC[col][row] + beta * C[C_index];
        }
      }
    }
  });
#undef TILESIZE
#undef MICROTILESIZE
  return HCBLAS_SUCCEEDS;
}

/*
* SGEMM - NoTransA case - Row major Access
* STEP Implementation
* TILESIZE = 8 STEPSIZE = 8
*/

hcblasStatus gemm_NoTransA_rMajor_batch_STEP_TS8XSS8(hc::accelerator_view accl_view,
						     float *A, long aOffset, long A_batchOffset,
						     float *B, long bOffset, long B_batchOffset,
						     float *C, long cOffset, long C_batchOffset,
						     int M, int N, int K, int lda, int ldb, int ldc,
						     float alpha, float beta, int batchSize) {
#define TILESIZE 8
#define STEPSIZE 8
  hc::extent<3> grdExt(batchSize, (M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int shiftFactor = hc::fast_math::log2(STEPSIZE);
    int elt = tidx.tile[0];
    float rC[1][1] = {{(float)0}};
    float rA[1][STEPSIZE / TILESIZE];
    float rB[1][STEPSIZE / TILESIZE];
    tile_static float lA[TILESIZE + TILESIZE * STEPSIZE];
    tile_static float lB[TILESIZE + TILESIZE * STEPSIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[2];
    int idx = tidx.local[1];
    int idy = tidx.local[2];
    int idt = TILESIZE * idy + idx;
    int idxT = idt % TILESIZE;
    int idyT = idt / TILESIZE;
    int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftFactor;
    int i = 0;

    do {
      tidx.barrier.wait();

      for(int sec = 0; sec < STEPSIZE / TILESIZE; ++sec) {
        if(gidy * TILESIZE + idxT < N && i * STEPSIZE + idyT + (sec * TILESIZE) < K) {
          lB[(sec * TILESIZE * TILESIZE) + idyT + idxT * TILESIZE] = B[bOffset + B_batchOffset * elt + (gidy * TILESIZE + idxT) * ldb + idyT + i * STEPSIZE + (sec * TILESIZE)];
        } else {
          lB[(sec * TILESIZE * TILESIZE ) + idyT + idxT * TILESIZE] = 0;
        }

        if(gidx * TILESIZE + idxT < M && i * STEPSIZE + idyT + (sec * TILESIZE ) < K) {
          lA[(sec * TILESIZE * TILESIZE) + idyT + idxT * TILESIZE] = A[aOffset + A_batchOffset * elt + (gidx * TILESIZE + idxT) * lda + idyT + i * STEPSIZE + (sec * TILESIZE)];
        } else {
          lA[(sec * TILESIZE * TILESIZE ) + idyT + idxT * TILESIZE] = 0;
        }
      }

      tidx.barrier.wait();
      int offA = idx * TILESIZE;
      int offB = idy * TILESIZE;
      int offset = 1;

      for (int iter = 0; iter < TILESIZE; ++iter) {
        MS1x1(offset, offset);
      }

      i++;
    } while (--block_k > 0);

    tidx.barrier.wait();

    if(gidx * TILESIZE + idx < M && gidy * TILESIZE + idy < N) {
      long C_index = cOffset + C_batchOffset * elt + (gidx * TILESIZE + idx) * ldc + (gidy * TILESIZE + idy);
      C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
      C[C_index] = alpha * rC[0][0] + beta * C[C_index];
    }
  });
#undef TILESIZE
#undef STEPSIZE
  return HCBLAS_SUCCEEDS;
}

/*
* SGEMM - NoTransA case - Row major Access
* STEP with Non Bank Conflict Implementation
* TILESIZE = 8 STEPSIZE = 8
*/

hcblasStatus gemm_NoTransA_rMajor_batch_STEP_NBK_TS8XSS8(hc::accelerator_view accl_view,
						         float *A, long aOffset, long A_batchOffset,
						         float *B, long bOffset, long B_batchOffset,
						         float *C, long cOffset, long C_batchOffset,
						         int M, int N, int K, int lda, int ldb, int ldc,
						         float alpha, float beta, int batchSize) {
#define TILESIZE 8
#define STEPSIZE 8
  hc::extent<3> grdExt(batchSize, (M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int tilemulshift = (int)hc::fast_math::log2(TILESIZE);
    int shiftfactor = (int)hc::fast_math::log2(STEPSIZE);
    int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftfactor;
    int elt = tidx.tile[0];
    float rC[1][1] = {{0.0}};
    float rA[1][STEPTILERATIO];
    float rB[1][STEPTILERATIO];
    tile_static float lA[STEPTILEPROD + STEPSIZE];
    tile_static float lB[STEPTILEPROD + STEPSIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[2];
    int idx = tidx.local[1];
    int idy = tidx.local[2];
    int idt = (idy << tilemulshift) + idx; //(idy * TILESIZE + idx)
    int ids = (idy << shiftfactor) + idx; //(idy * STEPSIZE + idx)
    int idxS = ids & (STEPSIZE - 1);
    int idyT = (idt) >> tilemulshift;
    int gidyOffset = gidy << tilemulshift;
    int gidxOffset = gidx << tilemulshift;
    int idyTOffset = idyT * BANKTILESIZE;
    int i = 0;

    do {
      tidx.barrier.wait();
      int iOffset = i << shiftfactor;

      for(int sec = 0; sec < STEPTILERATIO; ++sec) {
        int secOffset  = sec << tilemulshift;
        int secStartPt = (sec << tilemulshift) * BANKTILESIZE;
        int localIdx = secStartPt + idxS + idyTOffset;
        int kIndex = iOffset + idxS + secOffset;
        // Initialize the local memory with zero
        lB[localIdx] = 0;
        lA[localIdx] = 0;

        if(gidyOffset + idyT < N && kIndex < K) {
          lB[localIdx] = B[bOffset + B_batchOffset * elt + (gidyOffset + idyT) * ldb + kIndex];
        }

        if(gidxOffset + idyT < M && kIndex < K) {
          lA[localIdx] = A[aOffset + A_batchOffset * elt + (gidxOffset + idyT) * lda + kIndex];
        }
      }

      tidx.barrier.wait();
      int offA = idx * BANKTILESIZE;
      int offB = idy * BANKTILESIZE;

      for (int piter = 0; piter < TILESIZE; ++piter) {
        MS1x1_NOBANK(1);
      }

      i++;
    } while (--block_k > 0);

    tidx.barrier.wait();
    int crow = (gidxOffset + idx) * ldc;
    int ccolprod = (gidyOffset + idy);

    if(crow / ldc < M && ccolprod < N) {
      long C_index = cOffset + C_batchOffset * elt + crow + ccolprod;
      C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
      C[C_index] = alpha * rC[0][0] + beta * C[C_index];
    }
  });
#undef TILESIZE
#undef STEPSIZE
  return HCBLAS_SUCCEEDS;
}

/*
* SGEMM - NoTransA case - Row major Access
* STEP with Non Bank Conflict Implementation
* TILESIZE = 16 STEPSIZE = 16
*/

hcblasStatus gemm_NoTransA_rMajor_batch_STEP_NBK_TS16XSS16(hc::accelerator_view accl_view,
						           float *A, long aOffset, long A_batchOffset,
						           float *B, long bOffset, long B_batchOffset,
						           float *C, long cOffset, long C_batchOffset,
						           int M, int N, int K, int lda, int ldb, int ldc,
						           float alpha, float beta, int batchSize) {
#define TILESIZE 16
#define STEPSIZE 16
  hc::extent<3> grdExt(batchSize, (M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int tilemulshift = (int)hc::fast_math::log2(TILESIZE);
    int shiftfactor = (int)hc::fast_math::log2(STEPSIZE);
    int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftfactor;
    int elt = tidx.tile[0];
    float rC[1][1] = {{0.0}};
    float rA[1][STEPTILERATIO];
    float rB[1][STEPTILERATIO];
    tile_static float lA[STEPTILEPROD + STEPSIZE];
    tile_static float lB[STEPTILEPROD + STEPSIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[2];
    int idx = tidx.local[1];
    int idy = tidx.local[2];
    int idt = (idy << tilemulshift) + idx; //(idy * TILESIZE + idx)
    int ids = (idy << shiftfactor) + idx; //(idy * STEPSIZE + idx)
    int idxS = ids & (STEPSIZE - 1);
    int idyT = (idt) >> tilemulshift;
    int gidyOffset = gidy << tilemulshift;
    int gidxOffset = gidx << tilemulshift;
    int idyTOffset = idyT * BANKTILESIZE;
    int i = 0;

    do {
      tidx.barrier.wait();
      int iOffset = i << shiftfactor;

      for(int sec = 0; sec < STEPTILERATIO; ++sec) {
        int secOffset  = sec << tilemulshift;
        int secStartPt = (sec << tilemulshift) * BANKTILESIZE;
        int localIdx = secStartPt + idxS + idyTOffset;
        int kIndex = iOffset + idxS + secOffset;
        // Initialize the local memory with zero
        lB[localIdx] = 0;
        lA[localIdx] = 0;

        if(gidyOffset + idyT < N && kIndex < K) {
          lB[localIdx] = B[bOffset + B_batchOffset * elt + (gidyOffset + idyT) * ldb + kIndex];
        }

        if(gidxOffset + idyT < M && kIndex < K) {
          lA[localIdx] = A[aOffset + A_batchOffset * elt + (gidxOffset + idyT) * lda + kIndex];
        }
      }

      tidx.barrier.wait();
      int offA = idx * BANKTILESIZE;
      int offB = idy * BANKTILESIZE;

      for (int piter = 0; piter < TILESIZE; ++piter) {
        MS1x1_NOBANK(1);
      }

      i++;
    } while (--block_k > 0);

    tidx.barrier.wait();
    int crow = (gidxOffset + idx) * ldc;
    int ccolprod = (gidyOffset + idy);

    if(crow / ldc < M && ccolprod < N) {
      long C_index = cOffset + C_batchOffset * elt + crow + ccolprod;
      C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
      C[C_index] = alpha * rC[0][0] + beta * C[C_index];
    }
  });
#undef TILESIZE
#undef STEPSIZE
  return HCBLAS_SUCCEEDS;
}

/*
* SGEMM - NoTransA case - Row major Access
* SUBMICROTILE with Non Bank Conflict Implementation
* TILESIZE = 16 MICROTILESIZE = 8
*/

hcblasStatus gemm_NoTransA_rMajor_batch_MICRO_NBK_TS16XMTS2(hc::accelerator_view accl_view,
							    float *A, long aOffset, long A_batchOffset,
							    float *B, long bOffset, long B_batchOffset,
							    float *C, long cOffset, long C_batchOffset,
							    int M, int N, int K, int lda, int ldb, int ldc,
							    float alpha, float beta, int batchSize) {
#define TILESIZE 16
#define MICROTILESIZE 2
  int M_ = hc::fast_math::fmax(1, (M / MICROTILESIZE) + 1);
  int N_ = hc::fast_math::fmax(1, (N / MICROTILESIZE) + 1); 
  hc::extent<3> grdExt(batchSize, (M_  + (TILESIZE - 1)) & ~(TILESIZE - 1), (N_ + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int shiftTS = hc::fast_math::log2(TILESIZE);
    int elt = tidx.tile[0];
    float rC[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
    float rA[1][MICROTILESIZE];
    float rB[1][MICROTILESIZE];
    tile_static float lA[TOTMICROTILEPROD + TILESIZE];
    tile_static float lB[TOTMICROTILEPROD + TILESIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[2];
    int idx = tidx.local[1];
    int idy = tidx.local[2];
    int idt = ( idy << shiftTS ) + idx;
    int idxT = idt % TILESIZE ;
    int idyT = idt / TILESIZE;
    int block_k = 0;

    do {
      int colIndex = ( block_k << shiftTS ) + idyT;
      int lIndex = (idyT * BANKMICROTILESIZE) + idxT;
      tidx.barrier.wait();

      for(int sec = 0; sec < MICROTILESIZE; ++sec) {
        int secVal = sec << shiftTS;
        int BrowIndex = ( gidy * MICROTILEPROD) + idxT + secVal;
        int ArowIndex = ( gidx * MICROTILEPROD) + idxT + secVal;
        tidx.barrier.wait();

        if( BrowIndex < N && colIndex < K) {
          lB[ lIndex + secVal] = B[ bOffset + B_batchOffset * elt + BrowIndex * ldb + colIndex ];
        } else {
          lB[ lIndex + secVal] = 0;
        }

        if( ArowIndex < M && colIndex < K) {
          lA[ lIndex + secVal] = A[aOffset + A_batchOffset * elt + ArowIndex * lda +  colIndex];
        } else {
          lA[ lIndex + secVal] = 0;
        }
      }

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < TILESIZE; ++iter) {
        MTS_NOBANK;
      }

      tidx.barrier.wait();
    } while (++block_k < (((K + TILESIZE - 1) & ~(TILESIZE - 1)) >> shiftTS));

    int xIndex = ((gidx * MICROTILEPROD) + idx) * ldc;
    int yIndex = ((gidy * MICROTILEPROD) + idy);

    for( int row = 0; row < MICROTILESIZE; row++) {
      for( int col = 0; col < MICROTILESIZE ; col++) {
        if((xIndex / ldc) + (col << shiftTS) < M && (yIndex) + (row << shiftTS) < N) {
          long C_index = cOffset + C_batchOffset * elt + (xIndex + ((col << shiftTS) * N)) + yIndex + (row << shiftTS);
          C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
          C[C_index] = alpha * rC[col][row] + beta * C[C_index];
        }
      }
    }
  });
#undef TILESIZE
#undef MICROTILESIZE
  return HCBLAS_SUCCEEDS;
}

/*
* SGEMM - NoTransA case - Row major Access
* SUBMICROTILE Implementation
* TILESIZE = 16 MICROTILESIZE = 2
*/

hcblasStatus gemm_NoTransA_rMajor_batch_MICRO_TS16XMTS2(hc::accelerator_view accl_view,
						        float *A, long aOffset, long A_batchOffset,
						        float *B, long bOffset, long B_batchOffset,
						        float *C, long cOffset, long C_batchOffset,
						        int M, int N, int K, int lda, int ldb, int ldc,
						        float alpha, float beta, int batchSize) {
#define TILESIZE 16
#define MICROTILESIZE 2
  int M_ = hc::fast_math::fmax(1, (M / MICROTILESIZE) + 1);
  int N_ = hc::fast_math::fmax(1, (N / MICROTILESIZE) + 1); 
  hc::extent<3> grdExt(batchSize, (M_  + (TILESIZE - 1)) & ~(TILESIZE - 1), (N_ + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int elt = tidx.tile[0];
    float rC[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
    float rA[1][MICROTILESIZE];
    float rB[1][MICROTILESIZE];
    tile_static float lA[TILESIZE * TILESIZE * MICROTILESIZE];
    tile_static float lB[TILESIZE * TILESIZE * MICROTILESIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[2];
    int idx = tidx.local[1];
    int idy = tidx.local[2];
    int idt = TILESIZE * idy + idx;
    int idxT = idt % TILESIZE;
    int idyT = idt / TILESIZE;
    int block_k = 0;

    do {
      tidx.barrier.wait();

      for(int sec = 0; sec < MICROTILESIZE; ++sec) {
        if(gidy * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < N && block_k * TILESIZE + idyT < K) {
          lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = B[bOffset + B_batchOffset * elt + (gidy * TILESIZE * MICROTILESIZE + idxT + sec * TILESIZE) * ldb + idyT + block_k * TILESIZE];
        } else {
          lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
        }

        if(gidx * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < M && block_k * TILESIZE + idyT < K) {
          lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = A[aOffset + A_batchOffset * elt + (gidx * TILESIZE * MICROTILESIZE + idxT + sec * TILESIZE) * lda +  idyT + block_k * TILESIZE];
        } else {
          lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
        }
      }

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < TILESIZE; ++iter) {
        MTS;
      }

      tidx.barrier.wait();
    } while (++block_k < (((K + TILESIZE - 1) & ~(TILESIZE - 1)) / TILESIZE));

    int xIndex = (gidx * TILESIZE * MICROTILESIZE + idx) * ldc;
    int yIndex = (gidy * TILESIZE * MICROTILESIZE + idy);

    for( int row = 0; row < MICROTILESIZE; row++) {
      for( int col = 0; col < MICROTILESIZE ; col++) {
        if((xIndex / ldc) + (TILESIZE * col) < M && (yIndex) + (TILESIZE * row) < N) {
          long C_index = cOffset + C_batchOffset * elt + (xIndex + (TILESIZE * col) * N) + yIndex + (TILESIZE * row);
          C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
          C[C_index] = alpha * rC[col][row] + beta * C[C_index];
        }
      }
    }
  });
#undef TILESIZE
#undef MICROTILESIZE
  return HCBLAS_SUCCEEDS;
}
/*
* SGEMM - NoTransB case - Row major Access
* STEP with Non Bank Conflict Implementation
* TILESIZE = 16 STEPSIZE = 16
*/

hcblasStatus gemm_NoTransB_rMajor_batch_STEP_NBK_TS16XSS16(hc::accelerator_view accl_view,
						           float *A, long aOffset, long A_batchOffset,
						           float *B, long bOffset, long B_batchOffset,
						           float *C, long cOffset, long C_batchOffset,
						           int M, int N, int K, int lda, int ldb, int ldc,
						           float alpha, float beta, int batchSize) {
#define TILESIZE 16
#define STEPSIZE 16
  hc::extent<3> grdExt(batchSize, (M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int tilemulshift = (int)hc::fast_math::log2(TILESIZE);
    int shiftfactor = hc::fast_math::log2(STEPSIZE);
    int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftfactor;
    int elt = tidx.tile[0];
    float rC[1][1] = {{0.0}};
    float rA[1][STEPTILERATIO];
    float rB[1][STEPTILERATIO];
    tile_static float lA[STEPTILEPROD + STEPSIZE];
    tile_static float lB[STEPTILEPROD + STEPSIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[2];
    int idx = tidx.local[1];
    int idy = tidx.local[2];
    int idt = (idy << tilemulshift) + idx;
    int idyT = idt >> tilemulshift;
    int idxT = idt & (TILESIZE - 1);
    int gidyOffset = gidy << tilemulshift;
    int gidxOffset = gidx << tilemulshift;
    int idyTOffset = idyT * BANKTILESIZE;
    int i = 0;

    do {
      tidx.barrier.wait();
      int iOffset = i << shiftfactor;

      for(int sec = 0; sec < STEPTILERATIO; ++sec) {
        int secOffset  = sec << tilemulshift;
        int secStartPt = (sec << tilemulshift) * BANKTILESIZE;
        int localIdx = secStartPt + idxT + idyTOffset;
        int kIndex = iOffset + idyT + secOffset;
        // Initialize the local memory with zero
        lB[localIdx] = 0;
        lA[localIdx] = 0;

        if(gidyOffset + idxT < N && kIndex < K) {
          lB[localIdx] = B[bOffset + B_batchOffset * elt + gidyOffset + idxT + kIndex * ldb];
        }

        if(gidxOffset + idxT < M && kIndex < K) {
          lA[localIdx] = A[aOffset + A_batchOffset * elt + gidxOffset + idxT + kIndex * lda];
        }
      }

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < TILESIZE; ++iter) {
        MS1x1_NOBANK(BANKTILESIZE);
      }

      i++;
    } while (--block_k > 0);

    tidx.barrier.wait();
    int crow = (gidxOffset + idx) * ldc;
    int ccolprod = (gidyOffset + idy);

    if(crow / ldc < M && ccolprod < N) {
      long C_index = cOffset + C_batchOffset * elt + crow + ccolprod;
      C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
      C[C_index] = alpha * rC[0][0] + beta * C[C_index];
    }
  });
#undef TILESIZE
#undef STEPSIZE
  return HCBLAS_SUCCEEDS;
}

/*
* SGEMM - NoTransB case - Row major Access
* SUBMICROTILE with Non Bank Conflict Implementation
* TILESIZE = 16 MICROTILESIZE = 2
*/

hcblasStatus gemm_NoTransB_rMajor_batch_MICRO_NBK_TS16XMTS2(hc::accelerator_view accl_view,
							    float *A, long aOffset, long A_batchOffset,
							    float *B, long bOffset, long B_batchOffset,
							    float *C, long cOffset, long C_batchOffset,
							    int M, int N, int K, int lda, int ldb, int ldc,
							    float alpha, float beta, int batchSize) {
#define TILESIZE 16
#define MICROTILESIZE 2
  int M_ = hc::fast_math::fmax(1, (M / MICROTILESIZE + 1));
  int N_ = hc::fast_math::fmax(1, (N / MICROTILESIZE + 1)); 
  hc::extent<3> grdExt(batchSize, (M_  + (TILESIZE - 1)) & ~(TILESIZE - 1), (N_ + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int shiftTS = hc::fast_math::log2(TILESIZE);
    int elt = tidx.tile[0];
    float rC[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
    float rA[1][MICROTILESIZE];
    float rB[1][MICROTILESIZE];
    tile_static float lA[TOTMICROTILEPROD + TILESIZE];
    tile_static float lB[TOTMICROTILEPROD + TILESIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[2];
    int idx = tidx.local[1];
    int idy = tidx.local[2];
    int idt = ( idy << shiftTS) + idx;
    int idxT = idt & ( TILESIZE - 1);
    int idyT = idt >> shiftTS;
    int block_k = 0;

    do {
      int colIndex = ( block_k << shiftTS ) + idyT;
      int lIndex = (idyT * BANKMICROTILESIZE) + idxT;
      tidx.barrier.wait();

      for(int sec = 0; sec < MICROTILESIZE; ++sec) {
        int secVal = sec << shiftTS;
        int BrowIndex = (gidy * MICROTILEPROD) + idxT + secVal;
        int ArowIndex = (gidx * MICROTILEPROD) + idxT + secVal;

        if( BrowIndex < N && colIndex < K) {
          lB[ lIndex + secVal] = B[bOffset + B_batchOffset * elt + BrowIndex + colIndex * ldb];
        } else {
          lB[lIndex + secVal] = 0;
        }

        if(ArowIndex < M && colIndex < K) {
          lA[lIndex + secVal] = A[aOffset + A_batchOffset * elt + ArowIndex + colIndex * lda];
        } else {
          lA[lIndex + secVal] = 0;
        }
      }

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < TILESIZE; ++iter) {
        MTS_NOBANK;
      }

      tidx.barrier.wait();
    } while (++block_k < (((K + TILESIZE - 1) & ~(TILESIZE - 1)) >> shiftTS));

    int xIndex = ((gidx * MICROTILEPROD) + idx) * ldc;
    int yIndex = ((gidy * MICROTILEPROD) + idy);

    for( int row = 0; row < MICROTILESIZE; row++) {
      for( int col = 0; col < MICROTILESIZE ; col++) {
        if((xIndex / ldc) + (col << shiftTS) < M && (yIndex) + (row << shiftTS) < N) {
          long C_index = cOffset + C_batchOffset * elt + (xIndex + ((col << shiftTS) * N)) + yIndex + (row << shiftTS);
          C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
          C[C_index] = alpha * rC[col][row] + beta * C[C_index];
        }
      }
    }
  });
#undef TILESIZE
#undef MICROTILESIZE
  return HCBLAS_SUCCEEDS;
}

/*
* SGEMM - NoTransB case - Row major Access
* STEP Implementation
* TILESIZE = 8 STEPSIZE = 8
*/

hcblasStatus gemm_NoTransB_rMajor_batch_STEP_TS8XSS8(hc::accelerator_view accl_view,
						     float *A, long aOffset, long A_batchOffset,
						     float *B, long bOffset, long B_batchOffset,
						     float *C, long cOffset, long C_batchOffset,
						     int M, int N, int K, int lda, int ldb, int ldc,
						     float alpha, float beta, int batchSize) {
#define TILESIZE 8
#define STEPSIZE 8
  hc::extent<3> grdExt(batchSize, (M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int shiftFactor = hc::fast_math::log2(STEPSIZE);
    int elt = tidx.tile[0];
    float rC[1][1] = {{(float)0}};
    float rA[1][STEPSIZE / TILESIZE];
    float rB[1][STEPSIZE / TILESIZE];
    tile_static float lA[TILESIZE + TILESIZE * STEPSIZE];
    tile_static float lB[TILESIZE + TILESIZE * STEPSIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[2];
    int idx = tidx.local[1];
    int idy = tidx.local[2];
    int idt = TILESIZE * idy + idx;
    int idxT = idt % TILESIZE;
    int idyT = idt / TILESIZE;
    int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftFactor;
    int i = 0;

    do {
      tidx.barrier.wait();

      for(int sec = 0; sec < STEPSIZE / TILESIZE; ++sec) {
        if(gidy * TILESIZE + idxT < N && i * STEPSIZE + idyT + (sec * TILESIZE) < K) {
          lB[(idyT + (sec * TILESIZE)) * TILESIZE + idxT] = B[bOffset + B_batchOffset * elt + gidy * TILESIZE + idxT + (idyT + (sec * TILESIZE)) * ldb + i * (ldb << shiftFactor)];
        } else {
          lB[(idyT + (sec * TILESIZE )) * TILESIZE + idxT] = 0;
        }

        if(gidx * TILESIZE + idxT < M && i * STEPSIZE + idyT + (sec * TILESIZE) < K) {
          lA[(idyT + (sec * TILESIZE)) * TILESIZE + idxT] = A[aOffset + A_batchOffset * elt + gidx * TILESIZE + idxT + (idyT + (sec * TILESIZE)) * lda + i * (lda << shiftFactor)];
        } else {
          lA[(idyT + (sec * TILESIZE)) * TILESIZE + idxT] = 0;
        }
      }

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;
      int offset = TILESIZE;

      for (int iter = 0; iter < TILESIZE; ++iter) {
        MS1x1(offset, offset);
      }

      i++;
    } while (--block_k > 0);

    tidx.barrier.wait();

    if(gidx * TILESIZE + idx < M && gidy * TILESIZE + idy < N) {
      long C_index = cOffset + C_batchOffset * elt + (gidx * TILESIZE + idx) * ldc + (gidy * TILESIZE + idy);
      C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
      C[C_index] = alpha * rC[0][0] + beta * C[C_index];
    }
  });
#undef TILESIZE
#undef STEPSIZE
  return HCBLAS_SUCCEEDS;
}

/*
* SGEMM - NoTransB case - Row major Access
* STEP Implementation
* TILESIZE = 16 STEPSIZE = 16
*/

hcblasStatus gemm_NoTransB_rMajor_batch_STEP_TS16XSS16(hc::accelerator_view accl_view,
						       float *A, long aOffset, long A_batchOffset,
						       float *B, long bOffset, long B_batchOffset,
						       float *C, long cOffset, long C_batchOffset,
						       int M, int N, int K, int lda, int ldb, int ldc,
						       float alpha, float beta, int batchSize) {
#define TILESIZE 16
#define STEPSIZE 16
  hc::extent<3> grdExt(batchSize, (M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int shiftFactor = hc::fast_math::log2(STEPSIZE);
    int elt = tidx.tile[0];
    float rC[1][1] = {{(float)0}};
    float rA[1][STEPSIZE / TILESIZE];
    float rB[1][STEPSIZE / TILESIZE];
    tile_static float lA[TILESIZE + TILESIZE * STEPSIZE];
    tile_static float lB[TILESIZE + TILESIZE * STEPSIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[2];
    int idx = tidx.local[1];
    int idy = tidx.local[2];
    int idt = TILESIZE * idy + idx;
    int idxT = idt % TILESIZE;
    int idyT = idt / TILESIZE;
    int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftFactor;
    int i = 0;

    do {
      tidx.barrier.wait();

      for(int sec = 0; sec < STEPSIZE / TILESIZE; ++sec) {
        if(gidy * TILESIZE + idxT < N && i * STEPSIZE + idyT + (sec * TILESIZE) < K) {
          lB[(idyT + (sec * TILESIZE)) * TILESIZE + idxT] = B[bOffset + B_batchOffset * elt + gidy * TILESIZE + idxT + (idyT + (sec * TILESIZE)) * ldb + i * (ldb << shiftFactor)];
        } else {
          lB[(idyT + (sec * TILESIZE )) * TILESIZE + idxT] = 0;
        }

        if(gidx * TILESIZE + idxT < M && i * STEPSIZE + idyT + (sec * TILESIZE) < K) {
          lA[(idyT + (sec * TILESIZE)) * TILESIZE + idxT] = A[aOffset + A_batchOffset * elt + gidx * TILESIZE + idxT + (idyT + (sec * TILESIZE)) * lda + i * (lda << shiftFactor)];
        } else {
          lA[(idyT + (sec * TILESIZE)) * TILESIZE + idxT] = 0;
        }
      }

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;
      int offset = TILESIZE;

      for (int iter = 0; iter < TILESIZE; ++iter) {
        MS1x1(offset, offset);
      }

      i++;
    } while (--block_k > 0);

    tidx.barrier.wait();

    if(gidx * TILESIZE + idx < M && gidy * TILESIZE + idy < N) {
      long C_index = cOffset + C_batchOffset * elt + (gidx * TILESIZE + idx) * ldc + (gidy * TILESIZE + idy);
      C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
      C[C_index] = alpha * rC[0][0] + beta * C[C_index];
    }
  });
#undef TILESIZE
#undef STEPSIZE
  return HCBLAS_SUCCEEDS;
}

/*
* SGEMM - NoTransB case - Row major Access
* SUBMICORTILE Implementation
* TILESIZE = 16 STEPSIZE = 2
*/

hcblasStatus gemm_NoTransB_rMajor_batch_MICRO_TS16XMTS2(hc::accelerator_view accl_view,
						        float *A, long aOffset, long A_batchOffset,
						        float *B, long bOffset, long B_batchOffset,
						        float *C, long cOffset, long C_batchOffset,
						        int M, int N, int K, int lda, int ldb, int ldc,
						        float alpha, float beta, int batchSize) {
#define TILESIZE 16
#define MICROTILESIZE 2
  int M_ = hc::fast_math::fmax(1, (M / MICROTILESIZE + 1));
  int N_ = hc::fast_math::fmax(1, (N / MICROTILESIZE + 1)); 
  hc::extent<3> grdExt(batchSize, (M_  + (TILESIZE - 1)) & ~(TILESIZE - 1), (N_ + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int elt = tidx.tile[0];
    float rC[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
    float rA[1][MICROTILESIZE];
    float rB[1][MICROTILESIZE];
    tile_static float lA[TILESIZE * TILESIZE * MICROTILESIZE];
    tile_static float lB[TILESIZE * TILESIZE * MICROTILESIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[2];
    int idx = tidx.local[1];
    int idy = tidx.local[2];
    int idt = TILESIZE * idy + idx;
    int idxT = idt % TILESIZE;
    int idyT = idt / TILESIZE;
    int block_k = 0;

    do {
      tidx.barrier.wait();

      for(int sec = 0; sec < MICROTILESIZE; ++sec) {
        if(gidy * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < N && block_k * TILESIZE + idyT < K) {
          lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = B[bOffset + B_batchOffset * elt + (gidy * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE) + idyT * ldb + block_k * (ldb * TILESIZE)];
        } else {
          lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
        }

        if(gidx * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < M && block_k * TILESIZE + idyT < K) {
          lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = A[aOffset + A_batchOffset * elt + (gidx * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE) +  idyT * lda + block_k * (lda * TILESIZE)];
        } else {
          lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
        }
      }

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < TILESIZE; ++iter) {
        MTS;
      }

      tidx.barrier.wait();
    } while (++block_k < (((K + TILESIZE - 1) & ~(TILESIZE - 1)) / TILESIZE));

    int xIndex = (gidx * TILESIZE * MICROTILESIZE + idx) * ldc;
    int yIndex = (gidy * TILESIZE * MICROTILESIZE + idy);

    for( int row = 0; row < MICROTILESIZE; row++) {
      for( int col = 0; col < MICROTILESIZE ; col++) {
        if((xIndex / ldc) + (TILESIZE * col) < M && (yIndex) + (TILESIZE * row) < N) {
          long C_index = cOffset + C_batchOffset * elt + (xIndex + (TILESIZE * col) * N) + yIndex + (TILESIZE * row);
          C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
          C[C_index] = alpha * rC[col][row] + beta * C[C_index];
        }
      }
    }
  });
#undef TILESIZE
#undef MICROTILESIZE
  return HCBLAS_SUCCEEDS;
}

/*
* SGEMM - NoTransB case - Row major Access
* STEP with Non Bank Concflict Implementation
* TILESIZE = 8 STEPSIZE = 8
*/

hcblasStatus gemm_NoTransB_rMajor_batch_STEP_NBK_TS8XSS8(hc::accelerator_view accl_view,
						         float *A, long aOffset, long A_batchOffset,
						         float *B, long bOffset, long B_batchOffset,
						         float *C, long cOffset, long C_batchOffset,
						         int M, int N, int K, int lda, int ldb, int ldc,
						         float alpha, float beta, int batchSize) {
#define TILESIZE 8
#define STEPSIZE 8
  hc::extent<3> grdExt(batchSize, (M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int tilemulshift = (int)hc::fast_math::log2(TILESIZE);
    int shiftfactor = hc::fast_math::log2(STEPSIZE);
    int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftfactor;
    int elt = tidx.tile[0];
    float rC[1][1] = {{0.0}};
    float rA[1][STEPTILERATIO];
    float rB[1][STEPTILERATIO];
    tile_static float lA[STEPTILEPROD + STEPSIZE];
    tile_static float lB[STEPTILEPROD + STEPSIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[2];
    int idx = tidx.local[1];
    int idy = tidx.local[2];
    int idt = (idy << tilemulshift) + idx;
    int idyT = idt >> tilemulshift;
    int idxT = idt & (TILESIZE - 1);
    int gidyOffset = gidy << tilemulshift;
    int gidxOffset = gidx << tilemulshift;
    int idyTOffset = idyT * BANKTILESIZE;
    int i = 0;

    do {
      tidx.barrier.wait();
      int iOffset = i << shiftfactor;

      for(int sec = 0; sec < STEPTILERATIO; ++sec) {
        int secOffset  = sec << tilemulshift;
        int secStartPt = (sec << tilemulshift) * BANKTILESIZE;
        int localIdx = secStartPt + idxT + idyTOffset;
        int kIndex = iOffset + idyT + secOffset;
        // Initialize the local memory with zero
        lB[localIdx] = 0;
        lA[localIdx] = 0;

        if(gidyOffset + idxT < N && kIndex < K) {
          lB[localIdx] = B[bOffset + B_batchOffset * elt + gidyOffset + idxT + kIndex * ldb];
        }

        if(gidxOffset + idxT < M && kIndex < K) {
          lA[localIdx] = A[aOffset + A_batchOffset * elt + gidxOffset + idxT + kIndex * lda];
        }
      }

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < TILESIZE; ++iter) {
        MS1x1_NOBANK(BANKTILESIZE);
      }

      i++;
    } while (--block_k > 0);

    tidx.barrier.wait();
    int crow = (gidxOffset + idx) * ldc;
    int ccolprod = (gidyOffset + idy);

    if(crow / ldc < M && ccolprod < N) {
      long C_index = cOffset + C_batchOffset * elt + crow + ccolprod;
      C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
      C[C_index] = alpha * rC[0][0] + beta * C[C_index];
    }
  });
#undef TILESIZE
#undef STEPSIZE
  return HCBLAS_SUCCEEDS;
}

/*
* SGEMM - TransAB case - Row major Access
* MICRO with Non Bank Conflict Implementation
* TILESIZE = 16 MICROTILESIZE = 2
*/

hcblasStatus gemm_TransAB_rMajor_batch_MICRO_NBK_TS16XMTS2(hc::accelerator_view accl_view,
						           float *A, long aOffset, long A_batchOffset,
						           float *B, long bOffset, long B_batchOffset,
						           float *C, long cOffset, long C_batchOffset,
						           int M, int N, int K, int lda, int ldb, int ldc,
						           float alpha, float beta, int batchSize) {
#define TILESIZE 16
#define MICROTILESIZE 2
  int M_ = hc::fast_math::fmax(1, (M / MICROTILESIZE + 1));
  int N_ = hc::fast_math::fmax(1, (N / MICROTILESIZE + 1)); 
  hc::extent<3> grdExt(batchSize, (M_  + (TILESIZE - 1)) & ~(TILESIZE - 1), (N_ + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int shiftTS = hc::fast_math::log2(TILESIZE);
    float rC[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
    float rA[1][MICROTILESIZE];
    float rB[1][MICROTILESIZE];
    tile_static float lA[TOTMICROTILEPROD + TILESIZE];
    tile_static float lB[TOTMICROTILEPROD + TILESIZE];
    int elt = tidx.tile[0];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[2];
    int idx = tidx.local[1];
    int idy = tidx.local[2];
    int idt =  ( idy << shiftTS ) + idx;
    int idxT = idt & ( TILESIZE - 1);
    int idyT = idt >> shiftTS;
    int block_k = 0;

    do {
      int colIndex = ( block_k << shiftTS ) + idyT;
      int lIndex = (idyT * BANKMICROTILESIZE) + idxT;
      tidx.barrier.wait();

      for(int sec = 0; sec < MICROTILESIZE; ++sec) {
        int secVal = sec << shiftTS;
        int BrowIndex =  (gidy * MICROTILEPROD) + idxT + secVal;
        int ArowIndex = (gidx * MICROTILEPROD) + idxT + secVal;

        if( BrowIndex < N && colIndex < K) {
          lB[ lIndex + secVal] = B[bOffset + B_batchOffset * elt + BrowIndex * ldb + colIndex];
        } else {
          lB[lIndex + secVal] = 0;
        }

        if( ArowIndex < M && colIndex < K) {
          lA[ lIndex + secVal] = A[aOffset + A_batchOffset * elt + ArowIndex +  colIndex * lda];
        } else {
          lA[ lIndex + secVal] = 0;
        }
      }

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < TILESIZE; ++iter) {
        MTS_NOBANK;
      }

      tidx.barrier.wait();
    } while (++block_k < (((K + TILESIZE - 1) & ~(TILESIZE - 1)) / TILESIZE));

    int xIndex = ((gidx  * MICROTILEPROD) + idx) * ldc;
    int yIndex = (gidy * MICROTILEPROD) + idy;

    for( int row = 0; row < MICROTILESIZE; row++) {
      for( int col = 0; col < MICROTILESIZE ; col++) {
        if((xIndex / ldc) + (col << shiftTS) < M && yIndex  + (row << shiftTS) < N) {
          long C_index = cOffset + C_batchOffset * elt + (xIndex + (col << shiftTS) * ldc) + yIndex + (row << shiftTS);
          C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
          C[C_index] = alpha * rC[col][row] + beta * C[C_index];
        }
      }
    }
  });
#undef TILESIZE
#undef MICROTILESIZE
  return HCBLAS_SUCCEEDS;
}
/*
* SGEMM - TransAB case - Row major Access
* STEP with Non Bank Conflict Implementation
* TILESIZE = 8 STEPSIZE = 8
*/

hcblasStatus gemm_TransAB_rMajor_batch_STEP_NBK_TS8XSS8(hc::accelerator_view accl_view,
						        float *A, long aOffset, long A_batchOffset,
						        float *B, long bOffset, long B_batchOffset,
						        float *C, long cOffset, long C_batchOffset,
						        int M, int N, int K, int lda, int ldb, int ldc,
						        float alpha, float beta, int batchSize) {
#define TILESIZE 8
#define STEPSIZE 8
  hc::extent<3> grdExt(batchSize, (M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int shiftFactor = hc::fast_math::log2(STEPSIZE);
    int elt = tidx.tile[0];
    float rC[1][1] = {{0.0}};
    float rA[1][STEPTILERATIO];
    float rB[1][STEPTILERATIO];
    tile_static float lA[STEPTILEPROD + STEPSIZE];//8*8+8
    tile_static float lB[STEPTILEPROD + STEPSIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[2];
    int idx = tidx.local[1];
    int idy = tidx.local[2];
    int idt = TILESIZE * idy + idx;
    int idxT = idt % TILESIZE;
    int idyT = idt / TILESIZE;
    int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftFactor;
    int i = 0;

    do {
      tidx.barrier.wait();

      // Load Sections of A and B into respective shared memory slots
      for (int sec = 0; sec < STEPSIZE / TILESIZE; ++sec) {
        // Load Section 'sec' from global memory B onto shared lB
        if(gidy * TILESIZE + idyT  < N && (idxT + i * STEPSIZE + (TILESIZE * sec)) < K) {
          lB[idyT * BANKTILESIZE + idxT + (BANKNUMTILEELMTS * sec)] = B[bOffset + B_batchOffset * elt + (gidy * TILESIZE + idyT) * ldb + idxT + i * STEPSIZE + (TILESIZE * sec)];
        } else {
          lB[idyT * BANKTILESIZE + idxT + (BANKNUMTILEELMTS * sec)] = 0;
        }

        // Load Section 'sec' from global memory A onto shared lA
        if(gidx * TILESIZE + idxT < M && (i * STEPSIZE + idyT + (TILESIZE * sec)) < K) {
          lA[idxT * BANKTILESIZE + idyT + (BANKNUMTILEELMTS * sec)] = A[aOffset  + A_batchOffset * elt + gidx * TILESIZE + idxT + idyT * lda + i * (lda << shiftFactor) + (TILESIZE * sec) * lda];
        } else {
          lA[idxT * BANKTILESIZE + idyT + (BANKNUMTILEELMTS * sec)] = 0;
        }
      }

      tidx.barrier.wait();
      int offA = idx * BANKTILESIZE;
      int offB = idy * BANKTILESIZE;

      for (int iter = 0; iter < TILESIZE; ++iter) {
        MS1x1_NOBANK(1);
      }

      i++;
    } while (--block_k > 0);

    tidx.barrier.wait();

    if(gidx * TILESIZE + idx < M && gidy * TILESIZE + idy < N) {
      long C_index = cOffset + C_batchOffset * elt + (gidx * TILESIZE + idx) * ldc + (gidy * TILESIZE + idy);
      C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
      C[C_index] = alpha * rC[0][0] + beta * C[C_index];
    }
  });
#undef TILESIZE
#undef STEPSIZE
  return HCBLAS_SUCCEEDS;
}
/*
* SGEMM - TransAB case - Row major Access
* STEP with Non Bank Conflict Implementation
* TILESIZE = 16 STEPSIZE = 16
*/

hcblasStatus gemm_TransAB_rMajor_batch_STEP_NBK_TS16XSS16(hc::accelerator_view accl_view,
						          float *A, long aOffset, long A_batchOffset,
						          float *B, long bOffset, long B_batchOffset,
						          float *C, long cOffset, long C_batchOffset,
						          int M, int N, int K, int lda, int ldb, int ldc,
						          float alpha, float beta, int batchSize) {
#define TILESIZE 16
#define STEPSIZE 16
  hc::extent<3> grdExt(batchSize, (M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int shiftFactor = hc::fast_math::log2(STEPSIZE);
    int elt = tidx.tile[0];
    float rC[1][1] = {{0.0}};
    float rA[1][STEPTILERATIO];
    float rB[1][STEPTILERATIO];
    tile_static float lA[STEPTILEPROD + STEPSIZE];//8*8+8
    tile_static float lB[STEPTILEPROD + STEPSIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[2];
    int idx = tidx.local[1];
    int idy = tidx.local[2];
    int idt = TILESIZE * idy + idx;
    int idxT = idt % TILESIZE;
    int idyT = idt / TILESIZE;
    int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftFactor;
    int i = 0;

    do {
      tidx.barrier.wait();

      // Load Sections of A and B into respective shared memory slots
      for (int sec = 0; sec < STEPSIZE / TILESIZE; ++sec) {
        // Load Section 'sec' from global memory B onto shared lB
        if(gidy * TILESIZE + idyT  < N && (idxT + i * STEPSIZE + (TILESIZE * sec)) < K) {
          lB[idyT * BANKTILESIZE + idxT + (BANKNUMTILEELMTS * sec)] = B[bOffset + B_batchOffset * elt + (gidy * TILESIZE + idyT) * ldb + idxT + i * STEPSIZE + (TILESIZE * sec)];
        } else {
          lB[idyT * BANKTILESIZE + idxT + (BANKNUMTILEELMTS * sec)] = 0;
        }

        // Load Section 'sec' from global memory A onto shared lA
        if(gidx * TILESIZE + idxT < M && (i * STEPSIZE + idyT + (TILESIZE * sec)) < K) {
          lA[idxT * BANKTILESIZE + idyT + (BANKNUMTILEELMTS * sec)] = A[aOffset  + A_batchOffset * elt + gidx * TILESIZE + idxT + idyT * lda + i * (lda << shiftFactor) + (TILESIZE * sec) * lda];
        } else {
          lA[idxT * BANKTILESIZE + idyT + (BANKNUMTILEELMTS * sec)] = 0;
        }
      }

      tidx.barrier.wait();
      int offA = idx * BANKTILESIZE;
      int offB = idy * BANKTILESIZE;

      for (int iter = 0; iter < TILESIZE; ++iter) {
        MS1x1_NOBANK(1);
      }

      i++;
    } while (--block_k > 0);

    tidx.barrier.wait();

    if(gidx * TILESIZE + idx < M && gidy * TILESIZE + idy < N) {
      long C_index = cOffset + C_batchOffset * elt + (gidx * TILESIZE + idx) * ldc + (gidy * TILESIZE + idy);
      C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
      C[C_index] = alpha * rC[0][0] + beta * C[C_index];
    }
  });
#undef TILESIZE
#undef STEPSIZE
  return HCBLAS_SUCCEEDS;
}

/*
* SGEMM - TransAB case - Row major Access
* SUBMICROTILE with Non Bank Concflict Implmentation
* TILESIZE = 16 MICROITLESIZE = 2
*/

hcblasStatus gemm_TransAB_rMajor_batch_MICRO_TS16XMTS2(hc::accelerator_view accl_view,
						       float *A, long aOffset, long A_batchOffset,
						       float *B, long bOffset, long B_batchOffset,
						       float *C, long cOffset, long C_batchOffset,
						       int M, int N, int K, int lda, int ldb, int ldc,
						       float alpha, float beta, int batchSize) {
#define TILESIZE 16
#define MICROTILESIZE 2
  int M_ = hc::fast_math::fmax(1, (M / MICROTILESIZE + 1));
  int N_ = hc::fast_math::fmax(1, (N / MICROTILESIZE + 1)); 
  hc::extent<3> grdExt(batchSize, (M_  + (TILESIZE - 1)) & ~(TILESIZE - 1), (N_ + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int elt = tidx.tile[0];
    float rC[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
    float rA[1][MICROTILESIZE];
    float rB[1][MICROTILESIZE];
    tile_static float lA[TILESIZE * TILESIZE * MICROTILESIZE];
    tile_static float lB[TILESIZE * TILESIZE * MICROTILESIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[2];
    int idx = tidx.local[1];
    int idy = tidx.local[2];
    int idt = TILESIZE * idy + idx;
    int idxT = idt % TILESIZE;
    int idyT = idt / TILESIZE;
    int block_k = 0;

    do {
      tidx.barrier.wait();

      for(int sec = 0; sec < MICROTILESIZE; ++sec) {
        if(gidy * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < N && block_k * TILESIZE + idyT < K) {
          lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = B[bOffset + B_batchOffset * elt + (gidy * TILESIZE * MICROTILESIZE + idxT + sec * TILESIZE) * ldb + idyT + block_k * TILESIZE];
        } else {
          lB[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
        }

        if(gidx * TILESIZE * MICROTILESIZE + idxT + (sec * TILESIZE) < M && block_k * TILESIZE + idyT < K) {
          lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = A[aOffset + A_batchOffset * elt + (gidx * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE) +  idyT * lda + block_k * (lda * TILESIZE)];
        } else {
          lA[(idyT * TILESIZE * MICROTILESIZE) + idxT + (sec * TILESIZE)] = 0;
        }
      }

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < TILESIZE; ++iter) {
        MTS;
      }

      tidx.barrier.wait();
    } while (++block_k < (((K + TILESIZE - 1) & ~(TILESIZE - 1)) / TILESIZE));

    int xIndex = (gidx * TILESIZE * MICROTILESIZE + idx) * ldc;
    int yIndex = (gidy * TILESIZE * MICROTILESIZE + idy);

    for( int row = 0; row < MICROTILESIZE; row++) {
      for( int col = 0; col < MICROTILESIZE ; col++) {
        if((xIndex / ldc) + (TILESIZE * col) < M && (yIndex) + (TILESIZE * row) < N) {
          long C_index = cOffset + C_batchOffset * elt + (xIndex + (TILESIZE * col) * N) + yIndex + (TILESIZE * row);
          C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
          C[C_index] = alpha * rC[col][row] + beta * C[C_index];
        }
      }
    }
  });
#undef TILESIZE
#undef MICROTILESIZE
  return HCBLAS_SUCCEEDS;
}

/*
* SGEMM - TransAB case - Row major Access
* STEP Implementation
* TILESIZE = 8 STEPSIZE = 8
*/

hcblasStatus gemm_TransAB_rMajor_batch_STEP_TS8XSS8(hc::accelerator_view accl_view,
						    float *A, long aOffset, long A_batchOffset,
						    float *B, long bOffset, long B_batchOffset,
						    float *C, long cOffset, long C_batchOffset,
						    int M, int N, int K, int lda, int ldb, int ldc,
						    float alpha, float beta, int batchSize) {
#define TILESIZE 8
#define STEPSIZE 8
  hc::extent<3> grdExt(batchSize, (M + (TILESIZE - 1)) & ~(TILESIZE - 1), (N + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int shiftFactor = hc::fast_math::log2(STEPSIZE);
    int elt = tidx.tile[0];
    float rC[1][1];
    float rA[1][STEPSIZE / TILESIZE];
    float rB[1][STEPSIZE / TILESIZE];
    tile_static float lA[TILESIZE + TILESIZE * STEPSIZE];//8*8+8
    tile_static float lB[TILESIZE + TILESIZE * STEPSIZE];
    rC[0][0] = 0;
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[2];
    int idx = tidx.local[1];
    int idy = tidx.local[2];
    int idt = TILESIZE * idy + idx;
    int idxT = idt % TILESIZE;
    int idyT = idt / TILESIZE;
    int block_k = ((K + (STEPSIZE - 1)) & ~(STEPSIZE - 1)) >> shiftFactor;
    int i = 0;

    do {
      tidx.barrier.wait();

      // Load Sections of A and B into respective shared memory slots
      for (int sec = 0; sec < STEPSIZE / TILESIZE; ++sec) {
        // Load Section 'sec' from global memory B onto shared lB
        if(gidy * TILESIZE + idxT  < N && (idyT + i * STEPSIZE + (TILESIZE * sec)) < K) {
          lB[idxT * TILESIZE + idyT + (TILESIZE * TILESIZE * sec)] = B[bOffset + B_batchOffset * elt + (gidy * TILESIZE + idxT) * ldb + idyT + i * STEPSIZE + (TILESIZE * sec)];
        } else {
          lB[idxT * TILESIZE + idyT + (TILESIZE * TILESIZE * sec)] = 0;
        }

        // Load Section 'sec' from global memory A onto shared lA
        if(gidx * TILESIZE + idxT < M && (i * STEPSIZE + idyT + (TILESIZE * sec)) < K) {
          lA[idxT * TILESIZE + idyT + (TILESIZE * TILESIZE * sec)] = A[aOffset  + A_batchOffset * elt + gidx * TILESIZE + idxT + idyT * lda + i * (lda << shiftFactor) + (TILESIZE * sec) * lda];
        } else {
          lA[idxT * TILESIZE + idyT + (TILESIZE * TILESIZE * sec)] = 0;
        }
      }

      tidx.barrier.wait();
      int offA = idx * TILESIZE;
      int offB = idy * TILESIZE;
      int offset = 1;

      for (int iter = 0; iter < TILESIZE; ++iter) {
        MS1x1(offset, offset);
      }

      i++;
    } while (--block_k > 0);

    tidx.barrier.wait();

    if(gidx * TILESIZE + idx < M && gidy * TILESIZE + idy < N) {
      long C_index = cOffset + C_batchOffset * elt + (gidx * TILESIZE + idx) * ldc + (gidy * TILESIZE + idy);
      C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
      C[C_index] = alpha * rC[0][0] + beta * C[C_index];
    }
  });
#undef TILESIZE
#undef STEPSIZE
  return HCBLAS_SUCCEEDS;
}

hcblasStatus gemm_TransAB_rMajor_batch_largeM(hc::accelerator_view accl_view,
					      float *A, long aOffset, long A_batchOffset,
					      float *B, long bOffset, long B_batchOffset,
					      float *C, long cOffset, long C_batchOffset,
					      int M, int N, int K, int lda, int ldb, int ldc,
					      float alpha, float beta, int batchSize) {
#define MICROTILESIZE_A 2
#define MICROTILESIZE_B 1
#define TILESIZE_A 32
#define TILESIZE_B 8
  int M_ = hc::fast_math::fmax(1, (M / MICROTILESIZE_A + 1));
  int N_ = hc::fast_math::fmax(1, (N / MICROTILESIZE_B + 1));
  hc::extent<3> grdExt(batchSize, (M_ + (TILESIZE_A - 1)) & ~(TILESIZE_A - 1), (N_ + (TILESIZE_B - 1)) & ~(TILESIZE_B - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, TILESIZE_A, TILESIZE_B);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int elt = tidx.tile[0];
    float rC[MICROTILESIZE_A][MICROTILESIZE_B] = {{0}};
    float rA[1][MICROTILESIZE_A];
    float rB[1][MICROTILESIZE_B];
    tile_static float lA[TILESIZE_A * TILESIZE_A * MICROTILESIZE_A];
    tile_static float lB[TILESIZE_A * TILESIZE_B * MICROTILESIZE_B];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[2];
    int idx = tidx.local[1];
    int idy = tidx.local[2];
    int idt = TILESIZE_A * idy + idx;
    int idxT = idt % TILESIZE_A;
    int idyT = idt / TILESIZE_A;
    int block_k = 0;

    do {
      for (int sec = 0; sec < MICROTILESIZE_B; ++sec) {
        if (gidy * TILESIZE_B * MICROTILESIZE_B + idyT + (sec * TILESIZE_B) < N && block_k * TILESIZE_A + idxT < K) {
          lB[(idxT * TILESIZE_B * MICROTILESIZE_B) + idyT + (sec * TILESIZE_B)] = B[bOffset + B_batchOffset * elt + (gidy * TILESIZE_B * MICROTILESIZE_B + idyT + sec * TILESIZE_B) * ldb + idxT + block_k * TILESIZE_A];
        } else {
          lB[(idxT * TILESIZE_B * MICROTILESIZE_B) + idyT + (sec * TILESIZE_B)] = 0;
        }
      }

      for (int sec = 0; sec < MICROTILESIZE_A; ++sec) {
        for (int iter = 0; iter < TILESIZE_A / TILESIZE_B; iter++) {
          if (gidx * TILESIZE_A * MICROTILESIZE_A + idxT + (sec * TILESIZE_A) < M && block_k * TILESIZE_A + iter * TILESIZE_B + idyT < K) {
            lA[((idyT + iter * TILESIZE_B) * TILESIZE_A * MICROTILESIZE_A) + idxT + (sec * TILESIZE_A)] = A[aOffset + A_batchOffset * elt + (gidx * TILESIZE_A * MICROTILESIZE_A) + idxT + (sec * TILESIZE_A) +  (idyT + iter * TILESIZE_B + block_k * TILESIZE_A) * lda];
          } else {
            lA[((idyT + iter * TILESIZE_B) * TILESIZE_A * MICROTILESIZE_A) + idxT + (sec * TILESIZE_A)] = 0;
          }
        }
      }

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < TILESIZE_A; ++iter) {
        MTS_AB;
      }

      tidx.barrier.wait();
    } while (++block_k < (((K + TILESIZE_A - 1) & ~(TILESIZE_A - 1)) / TILESIZE_A));

    int xIndex = gidx * TILESIZE_A * MICROTILESIZE_A + idx;
    int yIndex = gidy * TILESIZE_B * MICROTILESIZE_B + idy;

    for (int col = 0; col < MICROTILESIZE_A; col++) {
      for (int row = 0; row < MICROTILESIZE_B ; row++) {
        if (xIndex + (TILESIZE_A * col) < M && (yIndex) + (TILESIZE_B * row) < N) {
          long C_index = cOffset + C_batchOffset * elt + (xIndex + TILESIZE_A * col) * ldc + yIndex + TILESIZE_B * row;
          C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
          C[C_index] = alpha * rC[col][row] + beta * C[C_index];
        }
      }
    }
  });
#undef TILESIZE_A
#undef TILESIZE_B
#undef MICROTILESIZE_A
#undef MICROTILESIZE_B
  return HCBLAS_SUCCEEDS;
}

hcblasStatus gemm_NoTransB_rMajor_batch_largeM(hc::accelerator_view accl_view,
					       float *A, long aOffset, long A_batchOffset,
					       float *B, long bOffset, long B_batchOffset,
					       float *C, long cOffset, long C_batchOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       float alpha, float beta, int batchSize) {
#define MICROTILESIZE_A 2
#define MICROTILESIZE_B 2
#define TILESIZE_A 32
#define TILESIZE_B 8
  int M_ = hc::fast_math::fmax(1, (M / MICROTILESIZE_A + 1));
  int N_ = hc::fast_math::fmax(1, (N / MICROTILESIZE_B + 1));
  hc::extent<3> grdExt(batchSize, (M_ + (TILESIZE_A - 1)) & ~(TILESIZE_A - 1), (N_ + (TILESIZE_B - 1)) & ~(TILESIZE_B - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, TILESIZE_A, TILESIZE_B);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    float rC[MICROTILESIZE_A][MICROTILESIZE_B] = {{0}};
    float rA[1][MICROTILESIZE_A];
    float rB[1][MICROTILESIZE_B];
    tile_static float lA[TILESIZE_A * TILESIZE_B * MICROTILESIZE_A];
    tile_static float lB[TILESIZE_B * TILESIZE_B * MICROTILESIZE_B];
    int elt = tidx.tile[0];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[2];
    int idx = tidx.local[1];
    int idy = tidx.local[2];
    int idt = TILESIZE_A * idy + idx;
    int idxT = idt % TILESIZE_A;
    int idyT = idt / TILESIZE_A;
    int block_k = 0;

    do {
      tidx.barrier.wait();

      for (int sec = 0; sec < MICROTILESIZE_B; ++sec) {
        if (idxT % 4 == 0) {
          if (gidy * TILESIZE_B * MICROTILESIZE_B + idxT / 4 + (sec * TILESIZE_B) < N && block_k * TILESIZE_B + idyT < K) {
            lB[(idyT * TILESIZE_B * MICROTILESIZE_B) + idxT / 4 + (sec * TILESIZE_B)] = B[bOffset + B_batchOffset * elt + (gidy * TILESIZE_B * MICROTILESIZE_B) + idxT / 4 + (sec * TILESIZE_B) + (idyT + block_k * TILESIZE_B) * ldb];
          } else {
            lB[(idyT * TILESIZE_B * MICROTILESIZE_B) + idxT / 4 + (sec * TILESIZE_B)] = 0;
          }
        }
      }

      tidx.barrier.wait();

      for (int sec = 0; sec < MICROTILESIZE_A; ++sec) {
        if (gidx * TILESIZE_A * MICROTILESIZE_A + idxT + (sec * TILESIZE_A) < M && block_k * TILESIZE_B + idyT < K) {
          lA[(idyT * TILESIZE_A * MICROTILESIZE_A) + idxT + (sec * TILESIZE_A)] = A[aOffset + A_batchOffset * elt + (gidx * TILESIZE_A * MICROTILESIZE_A) + idxT + (sec * TILESIZE_A) +  (idyT + block_k * TILESIZE_B) * lda];
        } else {
          lA[(idyT * TILESIZE_A * MICROTILESIZE_A) + idxT + (sec * TILESIZE_A)] = 0;
        }
      }

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < TILESIZE_B; ++iter) {
        MTS_AB
      }

      tidx.barrier.wait();
    } while (++block_k < (((K + TILESIZE_B - 1) & ~(TILESIZE_B - 1)) / TILESIZE_B));

    int xIndex = (gidx * TILESIZE_A * MICROTILESIZE_A + idx) * ldc;
    int yIndex = gidy * TILESIZE_B * MICROTILESIZE_B + idy;

    for (int row = 0; row < MICROTILESIZE_B; row++) {
      for (int col = 0; col < MICROTILESIZE_A; col++) {
        if ((xIndex / ldc) + (TILESIZE_A * col) < M && yIndex + (TILESIZE_B * row) < N) {
          long C_index = cOffset + C_batchOffset * elt + (xIndex + (TILESIZE_A * col) * ldc) + yIndex + (TILESIZE_B * row);
          C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
          C[C_index] = alpha * rC[col][row] + beta * C[C_index];
        }
      }
    }
  });
#undef TILESIZE_A
#undef TILESIZE_B
#undef MICROTILESIZE_A
#undef MICROTILESIZE_B
  return HCBLAS_SUCCEEDS;
}

hcblasStatus gemm_NoTransA_rMajor_batch_largeM(hc::accelerator_view accl_view,
					       float *A, long aOffset, long A_batchOffset,
					       float *B, long bOffset, long B_batchOffset,
					       float *C, long cOffset, long C_batchOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       float alpha, float beta, int batchSize) {
#define MICROTILESIZE_A 2
#define MICROTILESIZE_B 1
#define TILESIZE_A 32
#define TILESIZE_B 8
  int M_ = hc::fast_math::fmax(1, (M / MICROTILESIZE_A + 1));
  int N_ = hc::fast_math::fmax(1, (N / MICROTILESIZE_B + 1));
  hc::extent<3> grdExt(batchSize, (M_ + (TILESIZE_A - 1)) & ~(TILESIZE_A - 1), (N_ + (TILESIZE_B - 1)) & ~(TILESIZE_B - 1));
  hc::tiled_extent<3> t_ext = grdExt.tile(1, TILESIZE_A, TILESIZE_B);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<3>& tidx) __attribute__((hc, cpu)) {
    int elt = tidx.tile[0];
    float rC[MICROTILESIZE_A][MICROTILESIZE_B] = {{0}};
    float rA[1][MICROTILESIZE_A];
    float rB[1][MICROTILESIZE_B];
    tile_static float lA[TILESIZE_A * TILESIZE_A * MICROTILESIZE_A];
    tile_static float lB[TILESIZE_A * TILESIZE_B * MICROTILESIZE_B];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[2];
    int idx = tidx.local[1];
    int idy = tidx.local[2];
    int idt = TILESIZE_A * idy + idx;
    int idxT = idt % TILESIZE_A;
    int idyT = idt / TILESIZE_A;
    int block_k = 0;

    do {
      for (int sec = 0; sec < MICROTILESIZE_B; ++sec) {
        if (gidy * TILESIZE_B * MICROTILESIZE_B + idyT + (sec * TILESIZE_B) < N && block_k * TILESIZE_A + idxT < K) {
          lB[(idxT * TILESIZE_B * MICROTILESIZE_B) + idyT + (sec * TILESIZE_B)] = B[bOffset + B_batchOffset * elt + (gidy * TILESIZE_B * MICROTILESIZE_B + idyT + sec * TILESIZE_B) * ldb + idxT + block_k * TILESIZE_A];
        } else {
          lB[(idxT * TILESIZE_B * MICROTILESIZE_B) + idyT + (sec * TILESIZE_B)] = 0;
        }
      }

      for (int sec = 0; sec < MICROTILESIZE_A; ++sec) {
        for (int iter = 0; iter < TILESIZE_A / TILESIZE_B; iter++) {
          if (gidx * TILESIZE_A * MICROTILESIZE_A + idxT + (sec * TILESIZE_A) < M && block_k * TILESIZE_A + iter * TILESIZE_B + idyT < K) {
            lA[((idyT + iter * TILESIZE_B) * TILESIZE_A * MICROTILESIZE_A) + idxT + (sec * TILESIZE_A)] = A[aOffset + A_batchOffset * elt + ((gidx * TILESIZE_A * MICROTILESIZE_A) + idxT + (sec * TILESIZE_A)) * lda +  (idyT + iter * TILESIZE_B + block_k * TILESIZE_A) ];
          } else {
            lA[((idyT + iter * TILESIZE_B) * TILESIZE_A * MICROTILESIZE_A) + idxT + (sec * TILESIZE_A)] = 0;
          }
        }
      }

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < TILESIZE_A; ++iter) {
        MTS_AB;
      }

      tidx.barrier.wait();
    } while (++block_k < (((K + TILESIZE_A - 1) & ~(TILESIZE_A - 1)) / TILESIZE_A));

    int xIndex = gidx * TILESIZE_A * MICROTILESIZE_A + idx;
    int yIndex = gidy * TILESIZE_B * MICROTILESIZE_B + idy;

    for (int row = 0; row < MICROTILESIZE_B ; row++) {
      for (int col = 0; col < MICROTILESIZE_A; col++) {
        if (xIndex + (TILESIZE_A * col) < M && (yIndex) + (TILESIZE_B * row) < N) {
          long C_index = cOffset + C_batchOffset * elt + (xIndex + TILESIZE_A * col) * ldc + yIndex + TILESIZE_B * row;
          C[C_index] = (isnan(C[C_index]) || isinf(C[C_index])) ? 0 : C[C_index];
          C[C_index] = alpha * rC[col][row] + beta * C[C_index];
        }
      }
    }
  });
#undef TILESIZE_A
#undef TILESIZE_B
#undef MICROTILESIZE_A
#undef MICROTILESIZE_B
  return HCBLAS_SUCCEEDS;
}


/*  TOP LEVEL FUNCITONS */
hcblasStatus gemm_NoTransAB_rMajor(hc::accelerator_view accl_view,
                                   float *A, long aOffset, long A_batchOffset,
                                   float *B, long bOffset, long B_batchOffset,
                                   float *C, long cOffset, long C_batchOffset,
                                   int M, int N, int K, int lda, int ldb, int ldc,
                                   float alpha, float beta, int batchSize) {
  if ((M < 600 && N < 600 && K < 10) || (M < 1800 && N < 600 && K < 600)) {
    return gemm_NoTransAB_rMajor_batch_STEP_NBK_TS8XSS8(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
  } else if ((M < 600 && N < 600 && K < 1800) || (M < 1800 && ((N < 600 && K < 1800) || (N < 1800 && K < 10)))) {
    return gemm_NoTransAB_rMajor_batch_STEP_NBK_TS16XSS16(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
  } else {
    return gemm_NoTransAB_rMajor_batch_MICRO_TS16XMTS2(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
  }
}

hcblasStatus gemm_NoTransA_rMajor(hc::accelerator_view accl_view,
                                  float *A, long aOffset, long A_batchOffset,
                                  float *B, long bOffset, long B_batchOffset,
                                  float *C, long cOffset, long C_batchOffset,
                                  int M, int N, int K, int lda, int ldb, int ldc,
                                  float alpha, float beta, int batchSize) {
  if (M > 10000 && N < 500) {
    return gemm_NoTransA_rMajor_batch_largeM(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
  } else if((M < 6000 && N < 600 && K < 10) || (M < 1800 && N < 80 &&  K > 1800 && K < 6000)) {
    return gemm_NoTransA_rMajor_batch_STEP_TS8XSS8(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
  } else if  ((M < 600 && N < 600 && K < 6000) || ( M > 1800 && M < 6000 && (K < 600 || (K > 1800 && K < 10000)) && N < 10 ) || (M < 10 && N < 600 && K < 1800 ) || (M < 600 && N < 1800 && K < 10 )) {
    return gemm_NoTransA_rMajor_batch_STEP_NBK_TS16XSS16(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
  } else if ((M > 1800 && M < 6000 && N > 100 && N < 600 && (K < 600  ||  (K < 6000 && K > 1800))) || ( M < 1800 && N < 600 && K < 10) || (M > 1800 && M < 6000 && K > 1800 &&  K < 6000 && N < 300 && M == K)) {
    return gemm_NoTransA_rMajor_batch_MICRO_NBK_TS16XMTS2(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
  } else if ((M == K && M < 10000 && N < 200 ) || (M < 600 && N < 1800 && K < 600 ) || ( M < 1800 && N < 100 && K < 1800) || (M > 600 && M < 6000 && K > 1800 &&  K < 10000 && N < 300 && M < K)) {
    return gemm_NoTransA_rMajor_batch_MICRO_TS16XMTS2(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
  } else {
    return gemm_NoTransA_rMajor_batch_MICRO_NBK_TS16XMTS2(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
  }
}


hcblasStatus gemm_NoTransB_rMajor(hc::accelerator_view accl_view,
                                  float *A, long aOffset, long A_batchOffset,
                                  float *B, long bOffset, long B_batchOffset,
                                  float *C, long cOffset, long C_batchOffset,
                                  int M, int N, int K, int lda, int ldb, int ldc,
                                  float alpha, float beta, int batchSize) {
  if (M > 10000 && N < 500) {
    return gemm_NoTransB_rMajor_batch_largeM(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
  } else if( M > 1800 && M < 6000 && N > 600 && N < 1800 && K < 600 ) {
    return gemm_NoTransB_rMajor_batch_MICRO_NBK_TS16XMTS2(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
  } else if (M > 600 && M < 1800 && N < 600 && K < 10) {
    return gemm_NoTransB_rMajor_batch_STEP_TS8XSS8(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
  } else if (M > 1800 && M < 6000 && N > 1800 && N < 6000 && K < 10) {
    return gemm_NoTransB_rMajor_batch_MICRO_TS16XMTS2(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
  } else if  ( (M < 600 && N < 600 && K < 6000) || ( M > 1800 && M < 6000 && K < 1800 && N < 10 ) || (M < 10 && N < 1800 && K > 1800 && K < 6000 )) {
    return gemm_NoTransB_rMajor_batch_STEP_TS16XSS16(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
  } else if (( M < 1800 && K < 600 && N < 10 ) || (M < 10 && N < 600 && K < 1800 ) || (M < 600 && N < 1800 && K < 10 )) {
    return gemm_NoTransB_rMajor_batch_STEP_NBK_TS16XSS16(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
  } else {
    return gemm_NoTransB_rMajor_batch_MICRO_TS16XMTS2(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
  }
}

hcblasStatus gemm_TransAB_rMajor(hc::accelerator_view accl_view,
                                 float *A, long aOffset, long A_batchOffset,
                                 float *B, long bOffset, long B_batchOffset,
                                 float *C, long cOffset, long C_batchOffset,
                                 int M, int N, int K, int lda, int ldb, int ldc,
                                 float alpha, float beta, int batchSize) {
  if (M > 10000 && N < 500) {
    return gemm_TransAB_rMajor_batch_largeM(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
  } else if ( M > 600 && M < 1800 && N < 200 && K > 600 && K < 1800) {
    return gemm_TransAB_rMajor_batch_MICRO_TS16XMTS2(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
  } else if ((( M > 600 && M < 1800 && N < 600 ) || (M < 50 && N < 1800)) && (K < 10)) {
    return gemm_TransAB_rMajor_batch_STEP_TS8XSS8(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
  } else if ((M < 600 && N < 600 && K < 6000) || (M > 1800 && M < 10000 && K > 600 && K < 10000 && N < 10) || (M < 10 && N > 600 && N < 1800 && K < 6000 )) {
    return gemm_TransAB_rMajor_batch_STEP_NBK_TS16XSS16(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
  } else if  ( (((M > 1800 && M < 6000 && M == K) || ( M > 1800 && M < 10000 && K > 1800 &&  K < 10000))  && N < 200) || (M < 10000 && N < 1800 && K < 10 ) || (M > 1800 && M < 6000 && N < 600 && K < 200)) {
    return gemm_TransAB_rMajor_batch_MICRO_NBK_TS16XMTS2(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
  } else if(M > 6000 && M < 10000 && N < 600 && K < 10) {
    return gemm_TransAB_rMajor_batch_STEP_TS8XSS8(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
  } else {
    return gemm_TransAB_rMajor_batch_MICRO_NBK_TS16XMTS2(accl_view, A, aOffset, A_batchOffset, B, bOffset, B_batchOffset, C, cOffset, C_batchOffset, M, N, K, lda, ldb, ldc, alpha, beta, batchSize);
  }
}
