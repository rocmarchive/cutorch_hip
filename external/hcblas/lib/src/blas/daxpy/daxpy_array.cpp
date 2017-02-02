#include "hcblaslib.h"
#include <hc.hpp>
#include "hc_math.hpp"
using namespace hc::fast_math;
using namespace hc;
#define BLOCK_SIZE 256

void axpy_HC(hc::accelerator_view accl_view,
             long n, double alpha,
             const double *X, long xOffset, long incx,
             double *Y, long yOffset, long incy) {
  if(n <= 102400) {
    long size = (n + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1);
    hc::extent<1> compute_domain(size);
    hc::parallel_for_each(accl_view, compute_domain.tile(BLOCK_SIZE), [ = ] (hc::tiled_index<1>& tidx) __attribute__((hc, cpu)) {
      if(tidx.global[0] < n) {
        long Y_index = yOffset + tidx.global[0];
        Y[Y_index] = (isnan(Y[Y_index]) || isinf(Y[Y_index])) ? 0 : Y[Y_index];
        Y[Y_index] += X[xOffset + tidx.global[0]] * alpha;
      }
    });
  } else {
    int step_sz;

    if (n <= 409600) {
      step_sz = 3;
    } else if (n <= 921600) {
      step_sz = 5;
    } else if ( n <= 1939526) {
      step_sz = 7;
    } else {
      step_sz = 15;
    }

    long size = (n / step_sz + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1);
    long nBlocks = size / BLOCK_SIZE;
    hc::extent<1> compute_domain(size);
    hc::parallel_for_each(accl_view, compute_domain.tile(BLOCK_SIZE), [ = ] (hc::tiled_index<1>& tidx) __attribute__((hc, cpu)) {
      if(tidx.tile[0] != nBlocks - 1) {
        for(int iter = 0; iter < step_sz; iter++) {
          long Y_index = yOffset + tidx.tile[0] * 256 * step_sz + tidx.local[0] + iter * 256;
          Y[Y_index] = (isnan(Y[Y_index]) || isinf(Y[Y_index])) ? 0 : Y[Y_index];
          Y[Y_index] += X[xOffset + tidx.tile[0] * 256 * step_sz + tidx.local[0] + iter * 256] * alpha;
        }
      } else {
        int n_iter = ((n - ((nBlocks - 1) * 256 * step_sz)) / 256) + 1;

        for(int iter = 0; iter < n_iter; iter++) {
          if (((nBlocks - 1) * 256 * step_sz + iter * 256 + tidx.local[0]) < n) {
            long Y_index = yOffset + tidx.tile[0] * 256 * step_sz + tidx.local[0] + iter * 256;
            Y[Y_index] = (isnan(Y[Y_index]) || isinf(Y[Y_index])) ? 0 : Y[Y_index];
            Y[Y_index] += X[xOffset + tidx.tile[0] * 256 * step_sz + tidx.local[0] + iter * 256] * alpha;
          }
        }
      }
    });
  }
}

void axpy_HC(hc::accelerator_view accl_view,
             long n, double alpha,
             const double *X, long xOffset, long incx,
             double *Y, long yOffset, long incy,
             long X_batchOffset, long Y_batchOffset, int batchSize) {
  if(n <= 102400) {
    long size = (n + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1);
    hc::extent<2> compute_domain(batchSize, size);
    hc::parallel_for_each(accl_view, compute_domain.tile(1, BLOCK_SIZE), [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {
      int elt = tidx.tile[0];

      if(tidx.global[1] < n) {
        long Y_index = yOffset + Y_batchOffset * elt + tidx.global[1];
        Y[Y_index] = (isnan(Y[Y_index]) || isinf(Y[Y_index])) ? 0 : Y[Y_index];
        Y[Y_index] += X[xOffset + X_batchOffset * elt + tidx.global[1]] * alpha;
      }
    });
  } else {
    int step_sz;

    if (n <= 409600) {
      step_sz = 3;
    } else if (n <= 921600) {
      step_sz = 5;
    } else if ( n <= 1939526) {
      step_sz = 7;
    } else {
      step_sz = 15;
    }

    long size = (n / step_sz + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1);
    long nBlocks = size / BLOCK_SIZE;
    hc::extent<2> compute_domain(batchSize, size);
    hc::parallel_for_each(accl_view, compute_domain.tile(1, BLOCK_SIZE), [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {
      int elt = tidx.tile[0];

      if(tidx.tile[1] != nBlocks - 1) {
        for(int iter = 0; iter < step_sz; iter++) {
          long Y_index = yOffset +  Y_batchOffset * elt + tidx.tile[1] * 256 * step_sz + tidx.local[1] + iter * 256;
          Y[Y_index] = (isnan(Y[Y_index]) || isinf(Y[Y_index])) ? 0 : Y[Y_index];
          Y[Y_index] += X[xOffset +  X_batchOffset * elt + tidx.tile[1] * 256 * step_sz + tidx.local[1] + iter * 256] * alpha;
        }
      } else {
        int n_iter = ((n - ((nBlocks - 1) * 256 * step_sz)) / 256) + 1;

        for(int iter = 0; iter < n_iter; iter++) {
          if (((nBlocks - 1) * 256 * step_sz + iter * 256 + tidx.local[1]) < n) {
            long Y_index = yOffset +  Y_batchOffset * elt + tidx.tile[1] * 256 * step_sz + tidx.local[1] + iter * 256;
            Y[Y_index] = (isnan(Y[Y_index]) || isinf(Y[Y_index])) ? 0 : Y[Y_index];
            Y[Y_index] += X[xOffset +  X_batchOffset * elt + tidx.tile[1] * 256 * step_sz + tidx.local[1] + iter * 256] * alpha;
          }
        }
      }
    });
  }
}

/* SAXPY - Type I : Inputs and outputs are device pointers */
hcblasStatus Hcblaslibrary :: hcblas_daxpy(hc::accelerator_view accl_view,
				           const int N, const double &alpha,
				           const double *X, const int incX,
				           double *Y, const int incY,
				           const long xOffset, const long yOffset)

{
  /*Check the conditions*/
  if ( X == NULL || Y == NULL || N <= 0 || incX <= 0 || incY <= 0 ) {
    return HCBLAS_INVALID;
  }

  if ( alpha == 0) {
    return HCBLAS_SUCCEEDS;
  }

  axpy_HC(accl_view, N, alpha, X, xOffset, incX, Y, yOffset, incY);
  return HCBLAS_SUCCEEDS;
}

/* SAXPY - Type II : Inputs and outputs are device pointers with batch processing */
hcblasStatus  Hcblaslibrary :: hcblas_daxpy(hc::accelerator_view accl_view,
					    const int N, const double &alpha,
					    const double *X, const int incX, const long X_batchOffset,
					    double *Y, const int incY, const long Y_batchOffset,
					    const long xOffset, const long yOffset, const int batchSize)

{
  /*Check the conditions*/
  if ( X == NULL || Y == NULL || N <= 0 || incX <= 0 || incY <= 0 ) {
    return HCBLAS_INVALID;
  }

  if ( alpha == 0) {
    return HCBLAS_SUCCEEDS;
  }

  axpy_HC(accl_view, N, alpha, X, xOffset, incX, Y, yOffset, incY, X_batchOffset, Y_batchOffset, batchSize);
  return HCBLAS_SUCCEEDS;
}



