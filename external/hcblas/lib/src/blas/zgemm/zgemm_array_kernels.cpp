#include "zgemm_array_kernels.h"
#include "hc_math.hpp"
using namespace hc::fast_math;
hcblasStatus zgemm_NoTransAB_loopunroll(hc::accelerator_view accl_view,
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
    float CValue = 0, CValue1 = 0;
    int Row = tidx.tile[0] * TILE_DIM + tidx.local[0];
    int Col = tidx.tile[1] * TILE_DIM + tidx.local[1];
    tile_static float Asreal[TILE_DIM][TILE_DIM];
    tile_static float Asimg[TILE_DIM][TILE_DIM];
    tile_static float Bsreal[TILE_DIM][TILE_DIM];
    tile_static float Bsimg[TILE_DIM][TILE_DIM];
    float tempReal = 0.0;
    float tempImg = 0.0;
    float CReal = 0.0;
    float CImg = 0.0;

    for (int k = 0; k < ((K + (TILE_DIM - 1)) / TILE_DIM ) ; k++) {
      if (k * TILE_DIM + tidx.local[1] < K && Row < N) {
        Bsreal[tidx.local[0]][tidx.local[1]] = B[bOffset + Row * ldb + (k * TILE_DIM + tidx.local[1])].x;
        Bsimg[tidx.local[0]][tidx.local[1]] = B[bOffset + Row * ldb + (k * TILE_DIM + tidx.local[1])].y;
      } else {
        Bsreal[tidx.local[0]][tidx.local[1]] = 0.0;
        Bsimg[tidx.local[0]][tidx.local[1]] = 0.0;
      }

      if (k * TILE_DIM + tidx.local[0] < K && Col < M) {
        Asreal[tidx.local[0]][tidx.local[1]] = A[aOffset + (k * TILE_DIM + tidx.local[0]) * lda + Col].x;
        Asimg[tidx.local[0]][tidx.local[1]] = A[aOffset + (k * TILE_DIM + tidx.local[0]) * lda + Col].y;
      } else {
        Asreal[tidx.local[0]][tidx.local[1]] = 0.0;
        Asimg[tidx.local[0]][tidx.local[1]] = 0.0;
      }

      tidx.barrier.wait();
      // Unrolled Matrix Mul operation
      CValue += ((Bsreal[tidx.local[0]][0] * Asreal[0][tidx.local[1]]) - (Bsimg[tidx.local[0]][0] * Asimg[0][tidx.local[1]]) +
                 (Bsreal[tidx.local[0]][1] * Asreal[1][tidx.local[1]]) - (Bsimg[tidx.local[0]][1] * Asimg[1][tidx.local[1]]) +
                 (Bsreal[tidx.local[0]][2] * Asreal[2][tidx.local[1]]) - (Bsimg[tidx.local[0]][2] * Asimg[2][tidx.local[1]]) +
                 (Bsreal[tidx.local[0]][3] * Asreal[3][tidx.local[1]]) - (Bsimg[tidx.local[0]][3] * Asimg[3][tidx.local[1]]) +
                 (Bsreal[tidx.local[0]][4] * Asreal[4][tidx.local[1]]) - (Bsimg[tidx.local[0]][4] * Asimg[4][tidx.local[1]]) +
                 (Bsreal[tidx.local[0]][5] * Asreal[5][tidx.local[1]]) - (Bsimg[tidx.local[0]][5] * Asimg[5][tidx.local[1]]) +
                 (Bsreal[tidx.local[0]][6] * Asreal[6][tidx.local[1]]) - (Bsimg[tidx.local[0]][6] * Asimg[6][tidx.local[1]]) +
                 (Bsreal[tidx.local[0]][7] * Asreal[7][tidx.local[1]]) - (Bsimg[tidx.local[0]][7] * Asimg[7][tidx.local[1]]) +
                 (Bsreal[tidx.local[0]][8] * Asreal[8][tidx.local[1]]) - (Bsimg[tidx.local[0]][8] * Asimg[8][tidx.local[1]]) +
                 (Bsreal[tidx.local[0]][9] * Asreal[9][tidx.local[1]]) - (Bsimg[tidx.local[0]][9] * Asimg[9][tidx.local[1]]) +
                 (Bsreal[tidx.local[0]][10] * Asreal[10][tidx.local[1]]) - (Bsimg[tidx.local[0]][10] * Asimg[10][tidx.local[1]]) +
                 (Bsreal[tidx.local[0]][11] * Asreal[11][tidx.local[1]]) - (Bsimg[tidx.local[0]][11] * Asimg[11][tidx.local[1]]) +
                 (Bsreal[tidx.local[0]][12] * Asreal[12][tidx.local[1]]) - (Bsimg[tidx.local[0]][12] * Asimg[12][tidx.local[1]]) +
                 (Bsreal[tidx.local[0]][13] * Asreal[13][tidx.local[1]]) - (Bsimg[tidx.local[0]][13] * Asimg[13][tidx.local[1]]) +
                 (Bsreal[tidx.local[0]][14] * Asreal[14][tidx.local[1]]) - (Bsimg[tidx.local[0]][14] * Asimg[14][tidx.local[1]]) +
                 (Bsreal[tidx.local[0]][15] * Asreal[15][tidx.local[1]]) - (Bsimg[tidx.local[0]][15] * Asimg[15][tidx.local[1]]));
      CValue1 += ((Bsreal[tidx.local[0]][0] * Asimg[0][tidx.local[1]]) + (Bsimg[tidx.local[0]][0] * Asreal[0][tidx.local[1]]) +
                  (Bsreal[tidx.local[0]][1] * Asimg[1][tidx.local[1]]) + (Bsimg[tidx.local[0]][1] * Asreal[1][tidx.local[1]]) +
                  (Bsreal[tidx.local[0]][2] * Asimg[2][tidx.local[1]]) + (Bsimg[tidx.local[0]][2] * Asreal[2][tidx.local[1]]) +
                  (Bsreal[tidx.local[0]][3] * Asimg[3][tidx.local[1]]) + (Bsimg[tidx.local[0]][3] * Asreal[3][tidx.local[1]]) +
                  (Bsreal[tidx.local[0]][4] * Asimg[4][tidx.local[1]]) + (Bsimg[tidx.local[0]][4] * Asreal[4][tidx.local[1]]) +
                  (Bsreal[tidx.local[0]][5] * Asimg[5][tidx.local[1]]) + (Bsimg[tidx.local[0]][5] * Asreal[5][tidx.local[1]]) +
                  (Bsreal[tidx.local[0]][6] * Asimg[6][tidx.local[1]]) + (Bsimg[tidx.local[0]][6] * Asreal[6][tidx.local[1]]) +
                  (Bsreal[tidx.local[0]][7] * Asimg[7][tidx.local[1]]) + (Bsimg[tidx.local[0]][7] * Asreal[7][tidx.local[1]]) +
                  (Bsreal[tidx.local[0]][8] * Asimg[8][tidx.local[1]]) + (Bsimg[tidx.local[0]][8] * Asreal[8][tidx.local[1]]) +
                  (Bsreal[tidx.local[0]][9] * Asimg[9][tidx.local[1]]) + (Bsimg[tidx.local[0]][9] * Asreal[9][tidx.local[1]]) +
                  (Bsreal[tidx.local[0]][10] * Asimg[10][tidx.local[1]]) + (Bsimg[tidx.local[0]][10] * Asreal[10][tidx.local[1]]) +
                  (Bsreal[tidx.local[0]][11] * Asimg[11][tidx.local[1]]) + (Bsimg[tidx.local[0]][11] * Asreal[11][tidx.local[1]]) +
                  (Bsreal[tidx.local[0]][12] * Asimg[12][tidx.local[1]]) + (Bsimg[tidx.local[0]][12] * Asreal[12][tidx.local[1]]) +
                  (Bsreal[tidx.local[0]][13] * Asimg[13][tidx.local[1]]) + (Bsimg[tidx.local[0]][13] * Asreal[13][tidx.local[1]]) +
                  (Bsreal[tidx.local[0]][14] * Asimg[14][tidx.local[1]]) + (Bsimg[tidx.local[0]][14] * Asreal[14][tidx.local[1]]) +
                  (Bsreal[tidx.local[0]][15] * Asimg[15][tidx.local[1]]) + (Bsimg[tidx.local[0]][15] * Asreal[15][tidx.local[1]]));
      tidx.barrier.wait();
    }

    if (Row < N && Col < M) {
      CReal = C[cOffset + (tidx.global[0] * ldc) + tidx.global[1]].x;
      CImg = C[cOffset + (tidx.global[0] * ldc) + tidx.global[1]].y;
      CReal = (isnan(CReal) || isinf(CReal)) ? 0 : CReal;
      CImg = (isnan(CImg) || isinf(CImg)) ? 0 : CImg;
      tempReal = ((CReal * beta.x) - (CImg * beta.y));
      tempImg = ((CReal * beta.y) + (CImg * beta.x));
      C[cOffset + (tidx.global[0] * ldc) + tidx.global[1]].x = tempReal + ((CValue * alpha.x) - (CValue1 * alpha.y));
      C[cOffset + (tidx.global[0] * ldc) + tidx.global[1]].y = tempImg + ((CValue * alpha.y) + (CValue1 * alpha.x));
    }
  });
#undef THREADS
#undef TILE_DIM
  return HCBLAS_SUCCEEDS;
}


hcblasStatus zgemm_NoTransAB_MICRO_TS16XMTS2(hc::accelerator_view accl_view,
					     double_2 *A, long aOffset,
					     double_2 *B, long bOffset,
					     double_2 *C, long cOffset,
					     int M, int N, int K, int lda, int ldb, int ldc,
					     double_2 alpha, double_2 beta) {
#define TILESIZE 16
#define MICROTILESIZE 1
  int M_ = hc::fast_math::fmax(1, (M / MICROTILESIZE));
  int N_ = hc::fast_math::fmax(1, (N / MICROTILESIZE));
  hc::extent<2> grdExt((N_ + (TILESIZE - 1)) & ~(TILESIZE - 1), (M_ + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<2> t_ext = grdExt.tile(TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {
    int shiftTS = hc::fast_math::log2(TILESIZE);
    int shiftMTP = hc::fast_math::log2(MICROTILEPROD);
    float rCreal[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
    float rCimg[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
    float rAreal[1][MICROTILESIZE] = {{(float)0}};
    float rAimg[1][MICROTILESIZE] = {{(float)0}};
    float rBreal[1][MICROTILESIZE] = {{(float)0}};
    float rBimg[1][MICROTILESIZE] = {{(float)0}};
    tile_static float lAreal[TOTMICROTILEPROD + TILESIZE];
    tile_static float lAimg[TOTMICROTILEPROD + TILESIZE];
    tile_static float lBreal[TOTMICROTILEPROD + TILESIZE];
    tile_static float lBimg[TOTMICROTILEPROD + TILESIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int idt =  ( idy * TILESIZE ) + idx;
    int idxT = idt & (TILESIZE - 1);
    int idyT = idt / TILESIZE;
    int block_k = 0;
    float tempReal = 0.0;
    float tempImg = 0.0;
    float CReal = 0.0;
    float CImg = 0.0;

    do {
      int colIndex = ( block_k * TILESIZE ) + idyT;
      int lIndex = (idyT * BANKMICROTILESIZE) + idxT;
      tidx.barrier.wait();

      for(int sec = 0; sec < MICROTILESIZE; ++sec) {
        int secVal = sec * TILESIZE;
        int BrowIndex =  (gidy * MICROTILEPROD) + idxT + secVal;
        int ArowIndex = (gidx * MICROTILEPROD) + idxT + secVal;

        if( BrowIndex < N && colIndex < K) {
          lBreal[lIndex + secVal] = B[bOffset + BrowIndex * ldb + colIndex].x;
          lBimg[lIndex + secVal] = B[bOffset + BrowIndex * ldb + colIndex].y;
        } else {
          lBreal[lIndex + secVal] = 0;
          lBimg[lIndex + secVal] = 0;
        }

        if( ArowIndex < M && colIndex < K) {
          lAreal[lIndex + secVal] = A[aOffset + ArowIndex +  colIndex * lda].x;
          lAimg[lIndex + secVal] = A[aOffset + ArowIndex +  colIndex * lda].y;
        } else {
          lAreal[lIndex + secVal] = 0;
          lAimg[lIndex + secVal] = 0;
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

    int xIndex = (gidx << shiftMTP) + idx;
    int yIndex = ((gidy << shiftMTP) + idy) * ldc;

    for( int row = 0; row < MICROTILESIZE; row++) {
      for( int col = 0; col < MICROTILESIZE ; col++) {
        if(xIndex + (col << shiftTS) < M && (yIndex / ldc) + (row << shiftTS) < N) {
          CReal = C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row << shiftTS) * ldc].x;
          CImg = C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row * TILESIZE) * ldc].y;
          CReal = (isnan(CReal) || isinf(CReal)) ? 0 : CReal;
          CImg = (isnan(CImg) || isinf(CImg)) ? 0 : CImg;
          tempReal = ((CReal * beta.x) - (CImg * beta.y));
          tempImg  = ((CReal * beta.y) + (CImg * beta.x));
          C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row << shiftTS) * ldc].x = tempReal + ((rCreal[col][row] * alpha.x) - (rCimg[col][row] * alpha.y));
          C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row * TILESIZE) * ldc].y = tempImg + ((rCreal[col][row] * alpha.y) + (rCimg[col][row] * alpha.x));
        }
      }
    }
  });
#undef TILESIZE
#undef MICROTILESIZE
  return HCBLAS_SUCCEEDS;
}

hcblasStatus zgemm_NoTransAB_STEP_TS8XSS8(hc::accelerator_view accl_view,
				          double_2 *A, long aOffset,
				          double_2 *B, long bOffset,
				          double_2 *C, long cOffset,
				          int M, int N, int K, int lda, int ldb, int ldc,
				          double_2 alpha, double_2 beta)

{
#define TILESIZE 8
#define STEPSIZE 8
  hc::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<2> t_ext = grdExt.tile(TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {
    int shiftFactor = hc::fast_math::log2(STEPSIZE);
    float rCreal[1][1];
    float rAreal[1][STEPSIZE / TILESIZE];
    float rBreal[1][STEPSIZE / TILESIZE];
    float rCimg[1][1];
    float rAimg[1][STEPSIZE / TILESIZE];
    float rBimg[1][STEPSIZE / TILESIZE];
    tile_static float lAreal[TILESIZE * STEPSIZE];//8*8+8
    tile_static float lBreal[TILESIZE * STEPSIZE];
    tile_static float lAimg[TILESIZE * STEPSIZE];//8*8+8
    tile_static float lBimg[TILESIZE * STEPSIZE];
    rCreal[0][0] = 0;
    rCimg[0][0] = 0;
    float tempReal = 0.0;
    float tempImg = 0.0;
    float CReal = 0.0;
    float CImg = 0.0;
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
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
          lBreal[idxT * TILESIZE + idyT + (TILESIZE * TILESIZE * sec)] = B[bOffset + (gidy * TILESIZE + idxT) * ldb + idyT + i * STEPSIZE + (TILESIZE * sec)].x;
          lBimg[idxT * TILESIZE + idyT + (TILESIZE * TILESIZE * sec)] = B[bOffset + (gidy * TILESIZE + idxT) * ldb + idyT + i * STEPSIZE + (TILESIZE * sec)].y;
        } else {
          lBreal[idxT * TILESIZE + idyT + (TILESIZE * TILESIZE * sec)] = 0;
          lBimg[idxT * TILESIZE + idyT + (TILESIZE * TILESIZE * sec)] = 0;
        }

        // Load Section 'sec' from global memory A onto shared lA
        if(gidx * TILESIZE + idxT < M && (i * STEPSIZE + idyT + (TILESIZE * sec)) < K) {
          lAreal[idxT * TILESIZE + idyT + (TILESIZE * TILESIZE * sec)] = A[aOffset  + gidx * TILESIZE + idxT + idyT * lda + i * (lda << shiftFactor) + (TILESIZE * sec) * lda].x;
          lAimg[idxT * TILESIZE + idyT + (TILESIZE * TILESIZE * sec)] = A[aOffset  + gidx * TILESIZE + idxT + idyT * lda + i * (lda << shiftFactor) + (TILESIZE * sec) * lda].y;
        } else {
          lAreal[idxT * TILESIZE + idyT + (TILESIZE * TILESIZE * sec)] = 0;
          lAimg[idxT * TILESIZE + idyT + (TILESIZE * TILESIZE * sec)] = 0;
        }
      }

      tidx.barrier.wait();
      int offA = idx * TILESIZE;
      int offB = idy * TILESIZE;
      int offset = 1;

      for (int iter = 0; iter < TILESIZE; ++iter) {
        MS1x1(offset);
      }

      i++;
    } while (--block_k > 0);

    tidx.barrier.wait();

    if(gidx * TILESIZE + idx < M && gidy * TILESIZE + idy < N) {
      CReal = C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy) * ldc].x;
      CImg = C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy) * ldc].y;
      CReal = (isnan(CReal) || isinf(CReal)) ? 0 : CReal;
      CImg = (isnan(CImg) || isinf(CImg)) ? 0 : CImg;
      tempReal = ((CReal * beta.x) - (CImg * beta.y));
      tempImg  = ((CReal * beta.y) + (CImg * beta.x));
      C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy)*ldc].x = tempReal + ((rCreal[0][0] * alpha.x) - (rCimg[0][0] * alpha.y));
      C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy)*ldc].y = tempImg  + ((rCreal[0][0] * alpha.y) + (rCimg[0][0] * alpha.x));
    }
  });
#undef TILESIZE
#undef STEPSIZE
  return HCBLAS_SUCCEEDS;
}


hcblasStatus zgemm_NoTransAB_MICRO_TS8XMTS2(hc::accelerator_view accl_view,
					    double_2 *A, long aOffset,
					    double_2 *B, long bOffset,
					    double_2 *C, long cOffset,
					    int M, int N, int K, int lda, int ldb, int ldc,
					    double_2 alpha, double_2 beta) {
#define TILESIZE 8
#define MICROTILESIZE 1
  int M_ = hc::fast_math::fmax(1, (M / MICROTILESIZE));
  int N_ = hc::fast_math::fmax(1, (N / MICROTILESIZE));
  hc::extent<2> grdExt(( N_ + (TILESIZE - 1)) & ~(TILESIZE - 1), ( M_ + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<2> t_ext = grdExt.tile(TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {
    int shiftTS = hc::fast_math::log2(TILESIZE);
    int shiftMTP = hc::fast_math::log2(MICROTILEPROD);
    float rCreal[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
    float rCimg[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
    float rAreal[1][MICROTILESIZE] = {{(float)0}};
    float rAimg[1][MICROTILESIZE] = {{(float)0}};
    float rBreal[1][MICROTILESIZE] = {{(float)0}};
    float rBimg[1][MICROTILESIZE] = {{(float)0}};
    tile_static float lAreal[TOTMICROTILEPROD + TILESIZE];
    tile_static float lAimg[TOTMICROTILEPROD + TILESIZE];
    tile_static float lBreal[TOTMICROTILEPROD + TILESIZE];
    tile_static float lBimg[TOTMICROTILEPROD + TILESIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int idt =  ( idy * TILESIZE ) + idx;
    int idxT = idt & (TILESIZE - 1);
    int idyT = idt / TILESIZE;
    int block_k = 0;
    float tempReal = 0.0;
    float tempImg = 0.0;
    float CReal = 0.0;
    float CImg = 0.0;

    do {
      int colIndex = ( block_k * TILESIZE ) + idyT;
      int lIndex = (idyT * BANKMICROTILESIZE) + idxT;
      tidx.barrier.wait();

      for(int sec = 0; sec < MICROTILESIZE; ++sec) {
        int secVal = sec * TILESIZE;
        int BrowIndex =  (gidy * MICROTILEPROD) + idxT + secVal;
        int ArowIndex = (gidx * MICROTILEPROD) + idxT + secVal;

        if( BrowIndex < N && colIndex < K) {
          lBreal[lIndex + secVal] = B[bOffset + BrowIndex * ldb + colIndex].x;
          lBimg[lIndex + secVal] = B[bOffset + BrowIndex * ldb + colIndex].y;
        } else {
          lBreal[lIndex + secVal] = 0;
          lBimg[lIndex + secVal] = 0;
        }

        if( ArowIndex < M && colIndex < K) {
          lAreal[lIndex + secVal] = A[aOffset + ArowIndex +  colIndex * lda].x;
          lAimg[lIndex + secVal] = A[aOffset + ArowIndex +  colIndex * lda].y;
        } else {
          lAreal[lIndex + secVal] = 0;
          lAimg[lIndex + secVal] = 0;
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

    int xIndex = (gidx << shiftMTP) + idx;
    int yIndex = ((gidy << shiftMTP) + idy) * ldc;

    for( int row = 0; row < MICROTILESIZE; row++) {
      for( int col = 0; col < MICROTILESIZE ; col++) {
        if(xIndex + (col << shiftTS) < M && (yIndex / ldc) + (row << shiftTS) < N) {
          CReal = C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row << shiftTS) * ldc].x;
          CImg = C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row * TILESIZE) * ldc].y;
          CReal = (isnan(CReal) || isinf(CReal)) ? 0 : CReal;
          CImg = (isnan(CImg) || isinf(CImg)) ? 0 : CImg;
          tempReal = ((CReal * beta.x) - (CImg * beta.y));
          tempImg  = ((CReal * beta.y) + (CImg * beta.x));
          C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row << shiftTS) * ldc].x = tempReal + ((rCreal[col][row] * alpha.x) - (rCimg[col][row] * alpha.y));
          C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row * TILESIZE) * ldc].y = tempImg + ((rCreal[col][row] * alpha.y) + (rCimg[col][row] * alpha.x));
        }
      }
    }
  });
#undef TILESIZE
#undef MICROTILESIZE
  return HCBLAS_SUCCEEDS;
}

hcblasStatus zgemm_NoTransA_MICRO_TS16XMTS2(hc::accelerator_view accl_view,
					    double_2 *A, long aOffset,
					    double_2 *B, long bOffset,
					    double_2 *C, long cOffset,
					    int M, int N, int K, int lda, int ldb, int ldc,
					    double_2 alpha, double_2 beta) {
#define TILESIZE 16
#define MICROTILESIZE 1
  int M_ = hc::fast_math::fmax(1, (M / MICROTILESIZE));
  int N_ = hc::fast_math::fmax(1, (N / MICROTILESIZE));
  hc::extent<2> grdExt((N_  + (TILESIZE - 1)) & ~(TILESIZE - 1), (M_ + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<2> t_ext = grdExt.tile(TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {
    int shiftTS = hc::fast_math::log2(TILESIZE);
    int shiftMTP = hc::fast_math::log2(MICROTILEPROD);
    float rCreal[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
    float rCimg[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
    float rAreal[1][MICROTILESIZE] = {{(float)0}};
    float rAimg[1][MICROTILESIZE] = {{(float)0}};
    float rBreal[1][MICROTILESIZE] = {{(float)0}};
    float rBimg[1][MICROTILESIZE] = {{(float)0}};
    tile_static float lAreal[TOTMICROTILEPROD + TILESIZE];
    tile_static float lAimg[TOTMICROTILEPROD + TILESIZE];
    tile_static float lBreal[TOTMICROTILEPROD + TILESIZE];
    tile_static float lBimg[TOTMICROTILEPROD + TILESIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int idt = ( idy << shiftTS) + idx;
    int idxT = idt & ( TILESIZE - 1);
    int idyT = idt >> shiftTS;
    int block_k = 0;
    float tempReal = 0.0;
    float tempImg = 0.0;
    float CReal = 0.0;
    float CImg = 0.0;

    do {
      int colIndex = ( block_k << shiftTS ) + idyT;
      int lIndex = (idyT * BANKMICROTILESIZE) + idxT;
      tidx.barrier.wait();

      for(int sec = 0; sec < MICROTILESIZE; ++sec) {
        int secVal = sec << shiftTS;
        int BrowIndex = (gidy << shiftMTP) + idxT + secVal;
        int ArowIndex = (gidx << shiftMTP) + idxT + secVal;

        if( BrowIndex < N && colIndex < K) {
          lBreal[lIndex + secVal] = B[bOffset + BrowIndex + colIndex * ldb].x;
          lBimg[lIndex + secVal] = B[bOffset + BrowIndex + colIndex * ldb].y;
        } else {
          lBreal[lIndex + secVal] = 0;
          lBimg[lIndex + secVal] = 0;
        }

        if(ArowIndex < M && colIndex < K) {
          lAreal[lIndex + secVal] = A[aOffset + ArowIndex + colIndex * lda].x;
          lAimg[lIndex + secVal] = A[aOffset + ArowIndex + colIndex * lda].y;
        } else {
          lAreal[lIndex + secVal] = 0;
          lAimg[lIndex + secVal] = 0;
        }
      }

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < TILESIZE; ++iter) {
        MTS;
      }

      tidx.barrier.wait();
    } while (++block_k < (((K + TILESIZE - 1) & ~(TILESIZE - 1)) >> shiftTS));

    int xIndex = (gidx << shiftMTP) + idx;
    int yIndex = ((gidy << shiftMTP) + idy) * ldc;

    for( int row = 0; row < MICROTILESIZE; row++) {
      for( int col = 0; col < MICROTILESIZE ; col++) {
        if(xIndex + (col << shiftTS) < M && (yIndex / ldc) + (row << shiftTS) < N) {
          CReal = C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row << shiftTS) * ldc].x;
          CImg = C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row * TILESIZE) * ldc].y;
          CReal = (isnan(CReal) || isinf(CReal)) ? 0 : CReal;
          CImg = (isnan(CImg) || isinf(CImg)) ? 0 : CImg;
          tempReal = ((CReal * beta.x) - (CImg * beta.y));
          tempImg  = ((CReal * beta.y) + (CImg * beta.x));
          C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row << shiftTS) * ldc].x = tempReal + ((rCreal[col][row] * alpha.x) - (rCimg[col][row] * alpha.y));
          C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row * TILESIZE) * ldc].y = tempImg + ((rCreal[col][row] * alpha.y) + (rCimg[col][row] * alpha.x));
        }
      }
    }
  });
#undef TILESIZE
#undef MICROTILESIZE
  return HCBLAS_SUCCEEDS;
}

hcblasStatus zgemm_NoTransB_STEP_TS8XSS8(hc::accelerator_view accl_view,
				         double_2 *A, long aOffset,
				         double_2 *B, long bOffset,
				         double_2 *C, long cOffset,
				         int M, int N, int K, int lda, int ldb, int ldc,
				         double_2 alpha, double_2 beta) {
#define TILESIZE 8
#define STEPSIZE 8
  hc::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<2> t_ext = grdExt.tile(TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {
    int shiftFactor = hc::fast_math::log2(TILESIZE);
    float rCreal[1][1];
    float rAreal[1][1];
    float rBreal[1][1];
    float rCimg[1][1];
    float rAimg[1][1];
    float rBimg[1][1];
    tile_static float lAreal[TILESIZE *  TILESIZE];//8*8+8
    tile_static float lBreal[TILESIZE * TILESIZE];
    tile_static float lAimg[TILESIZE * TILESIZE];//8*8+8
    tile_static float lBimg[TILESIZE * TILESIZE];
    rCreal[0][0] = 0;
    rCimg[0][0] = 0;
    float tempReal = 0.0;
    float tempImg = 0.0;
    float CReal = 0.0;
    float CImg = 0.0;
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int idt = TILESIZE * idy + idx;
    int idxT = idt % TILESIZE;
    int idyT = idt / TILESIZE;
    int block_k = ( (K + (TILESIZE - 1)) & ~(TILESIZE - 1)) >> shiftFactor;
    int i = 0;

    do {
      //barrier(CLK_LOCAL_MEM_FENCE);
      tidx.barrier.wait();

      if(gidy * TILESIZE + idxT < N && i * TILESIZE + idyT < K) {
        lBreal[idyT + idxT * TILESIZE] = B[bOffset + (gidy * TILESIZE + idxT) * ldb + idyT + i * TILESIZE].x;
        lBimg[idyT + idxT * TILESIZE] = B[bOffset + (gidy * TILESIZE + idxT) * ldb + idyT + i * TILESIZE].y;
      } else {
        lBreal[idyT + idxT * TILESIZE] = 0;
        lBimg[idyT + idxT * TILESIZE] = 0;
      }

      if(gidx * TILESIZE + idxT < M && i * TILESIZE + idyT < K) {
        lAreal[idyT + idxT * TILESIZE] = A[aOffset  + (gidx * TILESIZE + idxT) * lda + idyT + i * TILESIZE].x;
        lAimg[idyT + idxT * TILESIZE] = A[aOffset  + (gidx * TILESIZE + idxT) * lda + idyT + i * TILESIZE].y;
      } else {
        lAreal[idyT + idxT * TILESIZE] = 0;
        lAimg[idyT + idxT * TILESIZE] = 0;
      }

      tidx.barrier.wait();
      //barrier(CLK_LOCAL_MEM_FENCE);
      int offA = idx * TILESIZE;
      int offB = idy * TILESIZE;
      int offset = 1;

      for(int iter = 0; iter < TILESIZE; ++iter) {
        M1x1(offset);
      }

      i++;
    } while (--block_k > 0);

    tidx.barrier.wait();

    if(gidx * TILESIZE + idx < M && gidy * TILESIZE + idy < N) {
      CReal = C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy) * ldc].x;
      CImg = C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy) * ldc].y;
      CReal = (isnan(CReal) || isinf(CReal)) ? 0 : CReal;
      CImg = (isnan(CImg) || isinf(CImg)) ? 0 : CImg;
      tempReal = ((CReal * beta.x) - (CImg * beta.y));
      tempImg  = ((CReal * beta.y) + (CImg * beta.x));
      C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy)*ldc].x = tempReal + ((rCreal[0][0] * alpha.x) - (rCimg[0][0] * alpha.y));
      C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy)*ldc].y = tempImg  + ((rCreal[0][0] * alpha.y) + (rCimg[0][0] * alpha.x));
    }
  });
#undef TILESIZE
#undef STEPSIZE
  return HCBLAS_SUCCEEDS;
}

hcblasStatus zgemm_NoTransB_MICRO_TS16XMTS2(hc::accelerator_view accl_view,
				            double_2 *A, long aOffset,
				            double_2 *B, long bOffset,
					    double_2 *C, long cOffset,
					    int M, int N, int K, int lda, int ldb, int ldc,
					    double_2 alpha, double_2 beta) {
#define TILESIZE 16
#define MICROTILESIZE 1
  int M_ = hc::fast_math::fmax(1, (M / MICROTILESIZE));
  int N_ = hc::fast_math::fmax(1, (N / MICROTILESIZE));
  hc::extent<2> grdExt((N_ + (TILESIZE - 1)) & ~(TILESIZE - 1), (M_ + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<2> t_ext = grdExt.tile(TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {
    int shiftTS = hc::fast_math::log2(TILESIZE);
    int shiftMTP = hc::fast_math::log2(MICROTILEPROD);
    float rCreal[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
    float rCimg[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
    float rAreal[1][MICROTILESIZE] = {{(float)0}};
    float rAimg[1][MICROTILESIZE] = {{(float)0}};
    float rBreal[1][MICROTILESIZE] = {{(float)0}};
    float rBimg[1][MICROTILESIZE] = {{(float)0}};
    tile_static float lAreal[TOTMICROTILEPROD + TILESIZE];
    tile_static float lAimg[TOTMICROTILEPROD + TILESIZE];
    tile_static float lBreal[TOTMICROTILEPROD + TILESIZE];
    tile_static float lBimg[TOTMICROTILEPROD + TILESIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int idt = ( idy << shiftTS ) + idx;
    int idxT = idt % TILESIZE ;
    int idyT = idt / TILESIZE;
    int block_k = 0;
    float tempReal = 0.0;
    float tempImg = 0.0;
    float CReal = 0.0;
    float CImg = 0.0;

    do {
      int colIndex = ( block_k << shiftTS ) + idyT;
      int lIndex = (idyT * BANKMICROTILESIZE) + idxT;
      tidx.barrier.wait();

      for(int sec = 0; sec < MICROTILESIZE; ++sec) {
        int secVal = sec << shiftTS;
        int BrowIndex = ( gidy << shiftMTP) + idxT + secVal;
        int ArowIndex = ( gidx << shiftMTP) + idxT + secVal;
        tidx.barrier.wait();

        if( BrowIndex < N && colIndex < K) {
          lBreal[lIndex + secVal] = B[ bOffset + BrowIndex * ldb + colIndex].x;
          lBimg[lIndex + secVal] = B[ bOffset + BrowIndex * ldb + colIndex].y;
        } else {
          lBreal[lIndex + secVal] = 0;
          lBimg[lIndex + secVal] = 0;
        }

        if( ArowIndex < M && colIndex < K) {
          lAreal[lIndex + secVal] = A[aOffset + ArowIndex * lda +  colIndex].x;
          lAimg[lIndex + secVal] = A[aOffset + ArowIndex * lda +  colIndex].y;
        } else {
          lAreal[lIndex + secVal] = 0;
          lAimg[lIndex + secVal] = 0;
        }
      }

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < TILESIZE; ++iter) {
        MTS;
      }

      tidx.barrier.wait();
    } while (++block_k < (((K + TILESIZE - 1) & ~(TILESIZE - 1)) >> shiftTS));

    int xIndex = (gidx << shiftMTP) + idx;
    int yIndex = ((gidy << shiftMTP) + idy) * ldc;

    for( int row = 0; row < MICROTILESIZE; row++) {
      for( int col = 0; col < MICROTILESIZE ; col++) {
        if(xIndex + (col << shiftTS) < M && (yIndex / ldc) + (row << shiftTS) < N) {
          CReal = C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row << shiftTS) * ldc].x;
          CImg = C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row * TILESIZE) * ldc].y;
          CReal = (isnan(CReal) || isinf(CReal)) ? 0 : CReal;
          CImg = (isnan(CImg) || isinf(CImg)) ? 0 : CImg;
          tempReal = ((CReal * beta.x) - (CImg * beta.y));
          tempImg  = ((CReal * beta.y) + (CImg * beta.x));
          C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row << shiftTS) * ldc].x = tempReal + ((rCreal[col][row] * alpha.x) - (rCimg[col][row] * alpha.y));
          C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row * TILESIZE) * ldc].y = tempImg + ((rCreal[col][row] * alpha.y) + (rCimg[col][row] * alpha.x));
        }
      }
    }
  });
#undef TILESIZE
#undef MICROTILESIZE
  return HCBLAS_SUCCEEDS;
}

hcblasStatus zgemm_NoTransB_loopunroll(hc::accelerator_view accl_view,
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
    float CValue = 0, CValue1 = 0;
    int Row = tidx.global[0];
    int Col = tidx.global[1];
    tile_static float Asreal[TILE_DIM][TILE_DIM];
    tile_static float Asimg[TILE_DIM][TILE_DIM];
    tile_static float Bsreal[TILE_DIM][TILE_DIM];
    tile_static float Bsimg[TILE_DIM][TILE_DIM];
    float tempReal = 0.0;
    float tempImg = 0.0;
    float CReal = 0.0;
    float CImg = 0.0;

    for (int k = 0; k < ((K + (TILE_DIM - 1)) & ~(TILE_DIM - 1)) ; k += TILE_DIM) {
      if (k + tidx.local[1] < K && Row < N) {
        Bsreal[tidx.local[0]][tidx.local[1]] = B[bOffset + Row * ldb + k + tidx.local[1]].x;
        Bsimg[tidx.local[0]][tidx.local[1]] = B[bOffset + Row * ldb + k + tidx.local[1]].y;
      } else {
        Bsreal[tidx.local[0]][tidx.local[1]] = 0.0;
        Bsimg[tidx.local[0]][tidx.local[1]] = 0.0;
      }

      if (k + tidx.local[1] < K && (tidx.tile[1] * TILE_DIM + tidx.local[0]) < M) {
        Asreal[tidx.local[0]][tidx.local[1]] = A[aOffset + ((tidx.tile[1] * TILE_DIM + tidx.local[0]) * lda) + k + tidx.local[1]].x;
        Asimg[tidx.local[0]][tidx.local[1]] = A[aOffset + ((tidx.tile[1] * TILE_DIM + tidx.local[0]) * lda) + k + tidx.local[1]].y;
      } else {
        Asreal[tidx.local[0]][tidx.local[1]] = 0.0;
        Asimg[tidx.local[0]][tidx.local[1]] = 0.0;
      }

      tidx.barrier.wait();
      // Unrolled Matrix Mul operation
      CValue += ((Bsreal[tidx.local[0]][0] * Asreal[tidx.local[1]][0]) - (Bsimg[tidx.local[0]][0] * Asimg[tidx.local[1]][0]) +
                 (Bsreal[tidx.local[0]][1] * Asreal[tidx.local[1]][1]) - (Bsimg[tidx.local[0]][1] * Asimg[tidx.local[1]][1]) +
                 (Bsreal[tidx.local[0]][2] * Asreal[tidx.local[1]][2]) - (Bsimg[tidx.local[0]][2] * Asimg[tidx.local[1]][2]) +
                 (Bsreal[tidx.local[0]][3] * Asreal[tidx.local[1]][3]) - (Bsimg[tidx.local[0]][3] * Asimg[tidx.local[1]][3]) +
                 (Bsreal[tidx.local[0]][4] * Asreal[tidx.local[1]][4]) - (Bsimg[tidx.local[0]][4] * Asimg[tidx.local[1]][4]) +
                 (Bsreal[tidx.local[0]][5] * Asreal[tidx.local[1]][5]) - (Bsimg[tidx.local[0]][5] * Asimg[tidx.local[1]][5]) +
                 (Bsreal[tidx.local[0]][6] * Asreal[tidx.local[1]][6]) - (Bsimg[tidx.local[0]][6] * Asimg[tidx.local[1]][6]) +
                 (Bsreal[tidx.local[0]][7] * Asreal[tidx.local[1]][7]) - (Bsimg[tidx.local[0]][7] * Asimg[tidx.local[1]][7]) +
                 (Bsreal[tidx.local[0]][8] * Asreal[tidx.local[1]][8]) - (Bsimg[tidx.local[0]][8] * Asimg[tidx.local[1]][8]) +
                 (Bsreal[tidx.local[0]][9] * Asreal[tidx.local[1]][9]) - (Bsimg[tidx.local[0]][9] * Asimg[tidx.local[1]][9]) +
                 (Bsreal[tidx.local[0]][10] * Asreal[tidx.local[1]][10]) - (Bsimg[tidx.local[0]][10] * Asimg[tidx.local[1]][10]) +
                 (Bsreal[tidx.local[0]][11] * Asreal[tidx.local[1]][11]) - (Bsimg[tidx.local[0]][11] * Asimg[tidx.local[1]][11]) +
                 (Bsreal[tidx.local[0]][12] * Asreal[tidx.local[1]][12]) - (Bsimg[tidx.local[0]][12] * Asimg[tidx.local[1]][12]) +
                 (Bsreal[tidx.local[0]][13] * Asreal[tidx.local[1]][13]) - (Bsimg[tidx.local[0]][13] * Asimg[tidx.local[1]][13]) +
                 (Bsreal[tidx.local[0]][14] * Asreal[tidx.local[1]][14]) - (Bsimg[tidx.local[0]][14] * Asimg[tidx.local[1]][14]) +
                 (Bsreal[tidx.local[0]][15] * Asreal[tidx.local[1]][15]) - (Bsimg[tidx.local[0]][15] * Asimg[tidx.local[1]][15]));
      CValue1 += ((Bsreal[tidx.local[0]][0] * Asimg[tidx.local[1]][0]) + (Bsimg[tidx.local[0]][0] * Asreal[tidx.local[1]][0]) +
                  (Bsreal[tidx.local[0]][1] * Asimg[tidx.local[1]][1]) + (Bsimg[tidx.local[0]][1] * Asreal[tidx.local[1]][1]) +
                  (Bsreal[tidx.local[0]][2] * Asimg[tidx.local[1]][2]) + (Bsimg[tidx.local[0]][2] * Asreal[tidx.local[1]][2]) +
                  (Bsreal[tidx.local[0]][3] * Asimg[tidx.local[1]][3]) + (Bsimg[tidx.local[0]][3] * Asreal[tidx.local[1]][3]) +
                  (Bsreal[tidx.local[0]][4] * Asimg[tidx.local[1]][4]) + (Bsimg[tidx.local[0]][4] * Asreal[tidx.local[1]][4]) +
                  (Bsreal[tidx.local[0]][5] * Asimg[tidx.local[1]][5]) + (Bsimg[tidx.local[0]][5] * Asreal[tidx.local[1]][5]) +
                  (Bsreal[tidx.local[0]][6] * Asimg[tidx.local[1]][6]) + (Bsimg[tidx.local[0]][6] * Asreal[tidx.local[1]][6]) +
                  (Bsreal[tidx.local[0]][7] * Asimg[tidx.local[1]][7]) + (Bsimg[tidx.local[0]][7] * Asreal[tidx.local[1]][7]) +
                  (Bsreal[tidx.local[0]][8] * Asimg[tidx.local[1]][8]) + (Bsimg[tidx.local[0]][8] * Asreal[tidx.local[1]][8]) +
                  (Bsreal[tidx.local[0]][9] * Asimg[tidx.local[1]][9]) + (Bsimg[tidx.local[0]][9] * Asreal[tidx.local[1]][9]) +
                  (Bsreal[tidx.local[0]][10] * Asimg[tidx.local[1]][10]) + (Bsimg[tidx.local[0]][10] * Asreal[tidx.local[1]][10]) +
                  (Bsreal[tidx.local[0]][11] * Asimg[tidx.local[1]][11]) + (Bsimg[tidx.local[0]][11] * Asreal[tidx.local[1]][11]) +
                  (Bsreal[tidx.local[0]][12] * Asimg[tidx.local[1]][12]) + (Bsimg[tidx.local[0]][12] * Asreal[tidx.local[1]][12]) +
                  (Bsreal[tidx.local[0]][13] * Asimg[tidx.local[1]][13]) + (Bsimg[tidx.local[0]][13] * Asreal[tidx.local[1]][13]) +
                  (Bsreal[tidx.local[0]][14] * Asimg[tidx.local[1]][14]) + (Bsimg[tidx.local[0]][14] * Asreal[tidx.local[1]][14]) +
                  (Bsreal[tidx.local[0]][15] * Asimg[tidx.local[1]][15]) + (Bsimg[tidx.local[0]][15] * Asreal[tidx.local[1]][15]));
      tidx.barrier.wait();
    }

    if (Row < N && Col < M) {
      CReal = C[cOffset + (tidx.global[0] * ldc) + tidx.global[1]].x;
      CImg = C[cOffset + (tidx.global[0] * ldc) + tidx.global[1]].y;
      CReal = (isnan(CReal) || isinf(CReal)) ? 0 : CReal;
      CImg = (isnan(CImg) || isinf(CImg)) ? 0 : CImg;
      tempReal = ((CReal * beta.x) - (CImg * beta.y));
      tempImg = ((CReal * beta.y) + (CImg * beta.x));
      C[cOffset + (tidx.global[0] * ldc) + tidx.global[1]].x = tempReal + ((CValue * alpha.x) - (CValue1 * alpha.y));
      C[cOffset + (tidx.global[0] * ldc) + tidx.global[1]].y = tempImg + ((CValue * alpha.y) + (CValue1 * alpha.x));
    }
  });
#undef THREADS
#undef TILE_DIM
  return HCBLAS_SUCCEEDS;
}

hcblasStatus zgemm_TransAB_STEP_TS8XSS8(hc::accelerator_view accl_view,
                                        double_2 *A, long aOffset,
                                        double_2 *B, long bOffset,
                                        double_2 *C, long cOffset,
                                        int M, int N, int K, int lda, int ldb, int ldc,
                                        double_2 alpha, double_2 beta) {
#define TILESIZE 8
#define STEPSIZE 8
  hc::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<2> t_ext = grdExt.tile(TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {
    int shiftFactor = hc::fast_math::log2(TILESIZE);
    float rCreal[1][1];
    float rAreal[1][1];
    float rBreal[1][1];
    float rCimg[1][1];
    float rAimg[1][1];
    float rBimg[1][1];
    tile_static float lAreal[TILESIZE * TILESIZE];
    tile_static float lBreal[TILESIZE * TILESIZE];
    tile_static float lAimg[TILESIZE * TILESIZE];
    tile_static float lBimg[TILESIZE * TILESIZE];
    rCreal[0][0] = 0;
    rCimg[0][0] = 0;
    float tempReal = 0.0;
    float tempImg = 0.0;
    float CReal = 0.0;
    float CImg = 0.0;
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int idt = TILESIZE * idy + idx;
    int idxT = idt % TILESIZE;
    int idyT = idt / TILESIZE;
    int block_k = ((K + (TILESIZE - 1)) & ~(TILESIZE - 1)) >> shiftFactor;
    int i = 0;

    do {
      tidx.barrier.wait();

      if(gidy * TILESIZE + idxT < N && i * TILESIZE + idyT < K) {
        lBreal[idyT * TILESIZE + idxT] = B[bOffset + gidy * TILESIZE + idxT + idyT * ldb + i * (ldb << shiftFactor)].x;
        lBimg[idyT * TILESIZE + idxT] = B[bOffset + gidy * TILESIZE + idxT + idyT * ldb + i * (ldb << shiftFactor)].y;
      } else {
        lBreal[idyT * TILESIZE + idxT] = 0;
        lBimg[idyT * TILESIZE + idxT] = 0;
      }

      if(gidx * TILESIZE + idxT < M && i * TILESIZE + idyT < K) {
        lAreal[idyT * TILESIZE + idxT] = A[aOffset  + (gidx * TILESIZE + idxT) * lda + idyT + i * TILESIZE].x;
        lAimg[idyT * TILESIZE + idxT] = A[aOffset  + (gidx * TILESIZE + idxT) * lda + idyT + i * TILESIZE].y;
      } else {
        lAreal[idyT * TILESIZE + idxT] = 0;
        lAimg[idyT * TILESIZE + idxT] = 0;
      }

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;

      for(int iter = 0; iter < TILESIZE; iter++) {
        M1x1(TILESIZE);
      }

      i++;
    } while (--block_k > 0);

    tidx.barrier.wait();

    if(gidx * TILESIZE + idx < M && gidy * TILESIZE + idy < N) {
      CReal = C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy) * ldc].x;
      CImg = C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy) * ldc].y;
      CReal = (isnan(CReal) || isinf(CReal)) ? 0 : CReal;
      CImg = (isnan(CImg) || isinf(CImg)) ? 0 : CImg;
      tempReal = ((CReal * beta.x) - (CImg * beta.y));
      tempImg  = ((CReal * beta.y) + (CImg * beta.x));
      C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy)*ldc].x = tempReal + ((rCreal[0][0] * alpha.x) - (rCimg[0][0] * alpha.y));
      C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy)*ldc].y = tempImg  + ((rCreal[0][0] * alpha.y) + (rCimg[0][0] * alpha.x));
    }
  });
#undef TILESIZE
#undef STEPSIZE
  return HCBLAS_SUCCEEDS;
}

hcblasStatus zgemm_TransAB_STEP_TS16XSS16(hc::accelerator_view accl_view,
				          double_2 *A, long aOffset,
				          double_2 *B, long bOffset,
				          double_2 *C, long cOffset,
				          int M, int N, int K, int lda, int ldb, int ldc,
				          double_2 alpha, double_2 beta) {
#define TILESIZE 16
#define STEPSIZE 16
  hc::extent<2> grdExt((N + (TILESIZE - 1)) & ~(TILESIZE - 1), (M + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<2> t_ext = grdExt.tile(TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {
    int shiftFactor = hc::fast_math::log2(TILESIZE);
    float rCreal[1][1];
    float rAreal[1][1];
    float rBreal[1][1];
    float rCimg[1][1];
    float rAimg[1][1];
    float rBimg[1][1];
    tile_static float lAreal[TILESIZE * TILESIZE];
    tile_static float lBreal[TILESIZE * TILESIZE];
    tile_static float lAimg[TILESIZE * TILESIZE];
    tile_static float lBimg[TILESIZE * TILESIZE];
    rCreal[0][0] = 0;
    rCimg[0][0] = 0;
    float tempReal = 0.0;
    float tempImg = 0.0;
    float CReal = 0.0;
    float CImg = 0.0;
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int idt = TILESIZE * idy + idx;
    int idxT = idt % TILESIZE;
    int idyT = idt / TILESIZE;
    int block_k = ((K + (TILESIZE - 1)) & ~(TILESIZE - 1)) >> shiftFactor;
    int i = 0;

    do {
      tidx.barrier.wait();

      if(gidy * TILESIZE + idxT < N && i * TILESIZE + idyT < K) {
        lBreal[idyT * TILESIZE + idxT] = B[bOffset + gidy * TILESIZE + idxT + idyT * ldb + i * (ldb << shiftFactor)].x;
        lBimg[idyT * TILESIZE + idxT] = B[bOffset + gidy * TILESIZE + idxT + idyT * ldb + i * (ldb << shiftFactor)].y;
      } else {
        lBreal[idyT * TILESIZE + idxT] = 0;
        lBimg[idyT * TILESIZE + idxT] = 0;
      }

      if(gidx * TILESIZE + idxT < M && i * TILESIZE + idyT < K) {
        lAreal[idyT * TILESIZE + idxT] = A[aOffset  + (gidx * TILESIZE + idxT) * lda + idyT + i * TILESIZE].x;
        lAimg[idyT * TILESIZE + idxT] = A[aOffset  + (gidx * TILESIZE + idxT) * lda + idyT + i * TILESIZE].y;
      } else {
        lAreal[idyT * TILESIZE + idxT] = 0;
        lAimg[idyT * TILESIZE + idxT] = 0;
      }

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;

      for(int iter = 0; iter < TILESIZE; iter++) {
        M1x1(TILESIZE);
      }

      i++;
    } while (--block_k > 0);

    tidx.barrier.wait();

    if(gidx * TILESIZE + idx < M && gidy * TILESIZE + idy < N) {
      CReal = C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy) * ldc].x;
      CImg = C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy) * ldc].y;
      CReal = (isnan(CReal) || isinf(CReal)) ? 0 : CReal;
      CImg = (isnan(CImg) || isinf(CImg)) ? 0 : CImg;
      tempReal = ((CReal * beta.x) - (CImg * beta.y));
      tempImg  = ((CReal * beta.y) + (CImg * beta.x));
      C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy)*ldc].x = tempReal + ((rCreal[0][0] * alpha.x) - (rCimg[0][0] * alpha.y));
      C[cOffset + gidx * TILESIZE + idx + (gidy * TILESIZE + idy)*ldc].y = tempImg  + ((rCreal[0][0] * alpha.y) + (rCimg[0][0] * alpha.x));
    }
  });
#undef TILESIZE
#undef STEPSIZE
  return HCBLAS_SUCCEEDS;
}

hcblasStatus zgemm_TransAB_MICRO_TS16XMTS2(hc::accelerator_view accl_view,
				           double_2 *A, long aOffset,
				           double_2 *B, long bOffset,
				           double_2 *C, long cOffset,
				           int M, int N, int K, int lda, int ldb, int ldc,
			  	           double_2 alpha, double_2 beta) {
#define TILESIZE 16
#define MICROTILESIZE 1
  int M_ = hc::fast_math::fmax(1, (M / MICROTILESIZE));
  int N_ = hc::fast_math::fmax(1, (N / MICROTILESIZE));
  hc::extent<2> grdExt((N_ + (TILESIZE - 1)) & ~(TILESIZE - 1), (M_ + (TILESIZE - 1)) & ~(TILESIZE - 1));
  hc::tiled_extent<2> t_ext = grdExt.tile(TILESIZE, TILESIZE);
  hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {
    int shiftTS = hc::fast_math::log2(TILESIZE);
    int shiftMTP = hc::fast_math::log2(MICROTILEPROD);
    float rCreal[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
    float rCimg[MICROTILESIZE][MICROTILESIZE] = {{(float)0}};
    float rAreal[1][MICROTILESIZE] = {{(float)0}};
    float rAimg[1][MICROTILESIZE] = {{(float)0}};
    float rBreal[1][MICROTILESIZE] = {{(float)0}};
    float rBimg[1][MICROTILESIZE] = {{(float)0}};
    tile_static float lAreal[TOTMICROTILEPROD + TILESIZE];
    tile_static float lAimg[TOTMICROTILEPROD + TILESIZE];
    tile_static float lBreal[TOTMICROTILEPROD + TILESIZE];
    tile_static float lBimg[TOTMICROTILEPROD + TILESIZE];
    int gidx = tidx.tile[1];
    int gidy = tidx.tile[0];
    int idx = tidx.local[1];
    int idy = tidx.local[0];
    int idt = ( idy << shiftTS) + idx;
    int idxT = idt & ( TILESIZE - 1);
    int idyT = idt >> shiftTS;
    int block_k = 0;
    float tempReal = 0.0;
    float tempImg = 0.0;
    float CReal = 0.0;
    float CImg = 0.0;

    do {
      int colIndex = ( block_k << shiftTS ) + idyT;
      int lIndex = (idyT * BANKMICROTILESIZE) + idxT;
      tidx.barrier.wait();

      for(int sec = 0; sec < MICROTILESIZE; ++sec) {
        int secVal = sec << shiftTS;
        int BrowIndex = ( gidy << shiftMTP) + idxT + secVal;
        int ArowIndex = ( gidx << shiftMTP) + idxT + secVal;

        if( BrowIndex < N && colIndex < K) {
          lBreal[lIndex + secVal] = B[bOffset + BrowIndex + colIndex * ldb].x;
          lBimg[lIndex + secVal] = B[bOffset + BrowIndex + colIndex * ldb].y;
        } else {
          lBreal[lIndex + secVal ] = 0;
          lBimg[lIndex + secVal ] = 0;
        }

        if( ArowIndex < M && colIndex < K) {
          lAreal[lIndex + secVal] = A[aOffset + ArowIndex * lda + colIndex].x;
          lAimg[lIndex + secVal] = A[aOffset + ArowIndex * lda + colIndex].y;
        } else {
          lAreal[lIndex + secVal] = 0;
          lAimg[lIndex + secVal] = 0;
        }
      }

      tidx.barrier.wait();
      int offA = idx;
      int offB = idy;

      for (int iter = 0; iter < TILESIZE; ++iter) {
        MTS;
      }

      tidx.barrier.wait();
    } while (++block_k < (((K + TILESIZE - 1) & ~(TILESIZE - 1)) >> shiftTS));

    int xIndex = (gidx << shiftMTP) + idx;
    int yIndex = ((gidy << shiftMTP) + idy) * ldc;

    for( int row = 0; row < MICROTILESIZE; row++) {
      for( int col = 0; col < MICROTILESIZE ; col++) {
        if(xIndex + (col << shiftTS) < M && (yIndex / ldc) + (row << shiftTS) < N) {
          CReal = C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row << shiftTS) * ldc].x;
          CImg = C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row * TILESIZE) * ldc].y;
          CReal = (isnan(CReal) || isinf(CReal)) ? 0 : CReal;
          CImg = (isnan(CImg) || isinf(CImg)) ? 0 : CImg;
          tempReal = ((CReal * beta.x) - (CImg * beta.y));
          tempImg  = ((CReal * beta.y) + (CImg * beta.x));
          C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row << shiftTS) * ldc].x = tempReal + ((rCreal[col][row] * alpha.x) - (rCimg[col][row] * alpha.y));
          C[cOffset + (xIndex + (col << shiftTS)) + yIndex + (row * TILESIZE) * ldc].y = tempImg + ((rCreal[col][row] * alpha.y) + (rCimg[col][row] * alpha.x));
        }
      }
    }
  });
#undef TILESIZE
#undef MICROTILESIZE
  return HCBLAS_SUCCEEDS;
}

hcblasStatus zgemm_NoTransAB(hc::accelerator_view accl_view,
                             double_2 *A, long aOffset,
                             double_2 *B, long bOffset,
                             double_2 *C, long cOffset,
                             int M, int N, int K, int lda, int ldb, int ldc,
                             double_2 alpha, double_2 beta) {
  if (M < 600 && N < 600 && K >= 600 && K < 1800) {
    return zgemm_NoTransAB_loopunroll(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
  } else if (M >= 600 && M < 6000 && N < 600 && K < 1800) {
    return zgemm_NoTransAB_MICRO_TS16XMTS2(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
  } else if (M >= 1800 && M < 6000 && N < 10 && K >= 600 && K < 6000) {
    return zgemm_NoTransAB_STEP_TS8XSS8(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
  } else if (M >= 6000 && M < 10000 && N < 10 && K >= 1800 && K < 6000) {
    return zgemm_NoTransAB_MICRO_TS8XMTS2(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
  } else {
    return zgemm_NoTransAB_MICRO_TS16XMTS2(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
  }
}

hcblasStatus zgemm_NoTransA(hc::accelerator_view accl_view,
                            double_2 *A, long aOffset,
                            double_2 *B, long bOffset,
                            double_2 *C, long cOffset,
                            int M, int N, int K, int lda, int ldb, int ldc,
                            double_2 alpha, double_2 beta) {
  /*if ((M < 600 && N >= 600 && N < 1800 && K < 600)||(M >= 600 && M < 6000 && N <6000 && K < 600))
  { */
  return zgemm_NoTransA_MICRO_TS16XMTS2(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
  //}
}

hcblasStatus zgemm_NoTransB(hc::accelerator_view accl_view,
                            double_2 *A, long aOffset,
                            double_2 *B, long bOffset,
                            double_2 *C, long cOffset,
                            int M, int N, int K, int lda, int ldb, int ldc,
                            double_2 alpha, double_2 beta) {
  if ((M >= 10 && M < 6000 && N < 600 && K < 10) || (  M >= 600 && M < 1800 && N < 10 && K >= 1800 && K < 6000) || ( M < 600 && N < 600 && K > 1800 && K < 6000)) {
    return zgemm_NoTransB_STEP_TS8XSS8(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
  } else if (M >= 600 && M < 6000 && N < 600 && K < 600) {
    return zgemm_NoTransB_MICRO_TS16XMTS2(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
  } else if (M > 1800 && M < 6000 && N < 600 && K >= 1800 && K < 10000 ) {
    return zgemm_NoTransB_loopunroll(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
  } else {
    return zgemm_NoTransB_MICRO_TS16XMTS2(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
  }
}

hcblasStatus zgemm_TransAB(hc::accelerator_view accl_view,
                           double_2 *A, long aOffset,
                           double_2 *B, long bOffset,
                           double_2 *C, long cOffset,
                           int M, int N, int K, int lda, int ldb, int ldc,
                           double_2 alpha, double_2 beta) {
  if ((M < 600 && N < 600 && K < 10) || (M < 1800 && N < 600 && K < 600)) {
    return zgemm_TransAB_STEP_TS8XSS8(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
  } else if((M < 600 && N < 600 && K < 1800) || (M < 1800 && ((N < 600 && K < 1800) || (N < 1800 && K < 10)))) {
    return zgemm_TransAB_STEP_TS16XSS16(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
  } else { /*if (M < 1800 && N < 1800 && K < 1800)*/
    return zgemm_TransAB_MICRO_TS16XMTS2(accl_view, A, aOffset, B, bOffset, C, cOffset, M, N, K, lda, ldb, ldc, alpha, beta);
  }
}

