#include "hcblaslib.h"
#include <hc.hpp>
#include "hc_math.hpp"
using namespace hc::fast_math;

using namespace hc;
#define BLOCK_SIZE 8

void scopy_HC(hc::accelerator_view accl_view, long n,
              const float *X, long incx, long xOffset,
              float *Y, long incy, long yOffset) {
  long size = (n + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1);
  hc::extent<1> compute_domain(size);
  hc::parallel_for_each(accl_view, compute_domain.tile(BLOCK_SIZE), [ = ] (hc::tiled_index<1>& tidx) __attribute__((hc, cpu)) {
    if(tidx.global[0] < n) {
      long Y_index = yOffset + tidx.global[0];
      Y[Y_index] = (isnan(Y[Y_index]) || isinf(Y[Y_index])) ? 0 : Y[Y_index];
      Y[Y_index] = X[xOffset + tidx.global[0]];
    }
  });
}

void scopy_HC(hc::accelerator_view accl_view, long n,
              const float *X, long incx, long xOffset,
              float *Y, long incy, long yOffset,
              long X_batchOffset, long Y_batchOffset, int batchSize) {
  long size = (n + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1);
  hc::extent<2> compute_domain(batchSize, size);
  hc::parallel_for_each(accl_view, compute_domain.tile(1, BLOCK_SIZE), [ = ] (hc::tiled_index<2>& tidx) __attribute__((hc, cpu)) {
    int elt = tidx.tile[0];

    if(tidx.global[1] < n) {
      long Y_index = yOffset + Y_batchOffset * elt + tidx.global[1];
      Y[Y_index] = (isnan(Y[Y_index]) || isinf(Y[Y_index])) ? 0 : Y[Y_index];
      Y[Y_index] = X[xOffset + X_batchOffset * elt + tidx.global[1]];
    }
  });
}

// SCOPY Call Type I: Inputs and outputs are HCC float array containers
hcblasStatus Hcblaslibrary :: hcblas_scopy(hc::accelerator_view accl_view, const int N,
				           const float *X, const int incX, const long xOffset,
				           float *Y, const int incY, const long yOffset) {
  /*Check the conditions*/
  if ( X == NULL || Y == NULL || N <= 0 || incX <= 0 || incY <= 0 ) {
    return HCBLAS_INVALID;
  }

  scopy_HC(accl_view, N, X, incX, xOffset, Y, incY, yOffset);
  return HCBLAS_SUCCEEDS;
}

// SCOPY Type II - Overloaded function with arguments related to batch processing
hcblasStatus Hcblaslibrary :: hcblas_scopy(hc::accelerator_view accl_view, const int N,
				           const float *X, const int incX, const long xOffset,
				           float *Y, const int incY, const long yOffset,
				           const long X_batchOffset, const long Y_batchOffset, const int batchSize) {
  /*Check the conditions*/
  if ( X == NULL || Y == NULL || N <= 0 || incX <= 0 || incY <= 0 ) {
    return HCBLAS_INVALID;
  }

  scopy_HC(accl_view, N, X, incX, xOffset, Y, incY, yOffset, X_batchOffset, Y_batchOffset, batchSize);
  return HCBLAS_SUCCEEDS;
}

