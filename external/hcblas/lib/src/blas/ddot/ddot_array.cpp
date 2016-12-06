#include "hcblaslib.h"
#include <hc.hpp>
#include <hc_math.hpp>
#include <hc_am.hpp>
#define TILE_SIZE 256
using namespace hc;
using namespace hc::fast_math;
double ddot_HC(hc::accelerator_view accl_view, long n,
              const double *xView, long incx, long xOffset,
              const double *yView, long incy, long yOffset, double out) {
  out = 0.0;
  // runtime sizes
  unsigned int tile_count = (n + TILE_SIZE - 1) / TILE_SIZE;
  // simultaneous live threads
  const unsigned int thread_count = tile_count * TILE_SIZE;
  // global buffer (return type)
  hc::accelerator accl = accl_view.get_accelerator();
  double* dev_global_buffer = (double *) hc::am_alloc(sizeof(double) * tile_count, accl, 0);
  // configuration
  hc::extent<1> extent(thread_count);
  hc::parallel_for_each(accl_view, 
    extent.tile(TILE_SIZE),
  [ = ] (hc::tiled_index<1>& tid) __attribute__((hc, cpu)) {
    // shared tile buffer
    tile_static double local_buffer[TILE_SIZE];
    // indexes
    int idx = tid.global[0];
    // this threads's shared memory pointer
    double& smem = local_buffer[ tid.local[0] ];
    // initialize local buffer
    smem = 0.0f;

    // fold data into local buffer
    while (idx < n) {
      // reduction of smem and X[idx] with results stored in smem
      smem += xView[xOffset + hc::index<1>(idx)[0]] * yView[yOffset + hc::index<1>(idx)[0]];
      // next chunk
      idx += thread_count;
    }

    // synchronize
    tid.barrier.wait_with_tile_static_memory_fence();
    // reduce all values in this tile
    unsigned int local = tid.local[0];
    double* mem = &smem;

    // unrolled for performance
    if (local < 128) {
      mem[0] = mem[0] + mem[128];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <  64) {
      mem[0] = mem[0] + mem[ 64];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <  32) {
      mem[0] = mem[0] + mem[ 32];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <  16) {
      mem[0] = mem[0] + mem[ 16];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   8) {
      mem[0] = mem[0] + mem[  8];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   4) {
      mem[0] = mem[0] + mem[  4];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   2) {
      mem[0] = mem[0] + mem[  2];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   1) {
      mem[0] = mem[0] + mem[  1];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    // only 1 thread per tile does the inter tile communication
    if (tid.local[0] == 0) {
      // write to global buffer in this tiles
      dev_global_buffer[ tid.tile[0] ] = smem;
    }
  }).wait();

  // create host buffer
  double* host_global_buffer = (double* )malloc(sizeof(double) * tile_count);
  // Copy device contents back to host
  accl_view.copy(dev_global_buffer, host_global_buffer, sizeof(double) * tile_count);

  // 2nd pass reduction
  for(int i = 0; i < tile_count; i++) {
    out = (isnan(out) || isinf(out)) ? 0 : out;
    out += host_global_buffer[ i ] ;
  }

  //free up resources
  free(host_global_buffer);
  hc::am_free(dev_global_buffer);
  return out;
}

double ddot_HC(hc::accelerator_view accl_view, long n,
              const double *xView, long incx, long xOffset,
              const double *yView, long incy, long yOffset, double out,
              const long X_batchOffset, const long Y_batchOffset, const int batchSize)

{
  out = 0.0;
  // runtime sizes
  unsigned int tile_count = (n + TILE_SIZE - 1) / TILE_SIZE;
  // simultaneous live threads
  const unsigned int thread_count = tile_count * TILE_SIZE;
  // global buffer (return type)
  hc::accelerator accl = accl_view.get_accelerator();
  double* dev_global_buffer = (double *) hc::am_alloc(sizeof(double) * batchSize * tile_count, accl, 0);
  // configuration
  hc::extent<2> extent(batchSize, thread_count);
  hc::parallel_for_each(accl_view, 
    extent.tile(1, TILE_SIZE),
  [ = ] (hc::tiled_index<2>& tid) __attribute__((hc, cpu)) {
    // shared tile buffer
    tile_static double local_buffer[TILE_SIZE];
    // indexes
    int elt = tid.tile[0];
    int idx = tid.global[1];
    // this threads's shared memory pointer
    double& smem = local_buffer[ tid.local[1] ];
    // initialize local buffer
    smem = 0.0f;

    // fold data into local buffer
    while (idx < n) {
      // reduction of smem and X[idx] with results stored in smem
      smem += xView[xOffset + X_batchOffset * elt + hc::index<1>(idx)[0]] * yView[yOffset + Y_batchOffset * elt + hc::index<1>(idx)[0]];
      // next chunk
      idx += thread_count;
    }

    // synchronize
    tid.barrier.wait_with_tile_static_memory_fence();
    // reduce all values in this tile
    unsigned int local = tid.local[1];
    double* mem = &smem;

    // unrolled for performance
    if (local < 128) {
      mem[0] = mem[0] + mem[128];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <  64) {
      mem[0] = mem[0] + mem[ 64];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <  32) {
      mem[0] = mem[0] + mem[ 32];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <  16) {
      mem[0] = mem[0] + mem[ 16];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   8) {
      mem[0] = mem[0] + mem[  8];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   4) {
      mem[0] = mem[0] + mem[  4];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   2) {
      mem[0] = mem[0] + mem[  2];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    if (local <   1) {
      mem[0] = mem[0] + mem[  1];
    }

    tid.barrier.wait_with_tile_static_memory_fence();

    // only 1 thread per tile does the inter tile communication
    if (tid.local[1] == 0) {
      // write to global buffer in this tiles
      dev_global_buffer[ elt * tile_count + tid.tile[1] ] = smem;
    }
  }).wait();

  // create host buffer
  double* host_global_buffer = (double* )malloc(sizeof(double) * batchSize * tile_count);
  // Copy device contents back to host
  accl_view.copy(dev_global_buffer, host_global_buffer, sizeof(double) * batchSize * tile_count);

// 2nd pass reduction
  for(int i = 0; i < tile_count * batchSize; i++) {
    out = (isnan(out) || isinf(out)) ? 0 : out;
    out += host_global_buffer[ i ] ;
  }

  //free up resources
  free(host_global_buffer);
  hc::am_free(dev_global_buffer);

  return out;
}

// DDOT Call Type I: Inputs and outputs are HCC double array containers
hcblasStatus Hcblaslibrary :: hcblas_ddot(hc::accelerator_view accl_view, const int N,
				          const double *X, const int incX, const long xOffset,
				          const double *Y, const int incY, const long yOffset, double &dot)

{
  /*Check the conditions*/
  if (X == NULL || Y == NULL || N <= 0 || incX <= 0 || incY <= 0 ) {
    return HCBLAS_INVALID;
  }

  dot = ddot_HC(accl_view, N, X, incX, xOffset, Y, incY, yOffset, dot);
  return HCBLAS_SUCCEEDS;
}

// DDOT Type II - Overloaded function with arguments related to batch processing
hcblasStatus Hcblaslibrary :: hcblas_ddot(hc::accelerator_view accl_view, const int N,
				          const double *X, const int incX, const long xOffset,
				          const double *Y, const int incY, const long yOffset, double &dot,
				          const long X_batchOffset, const long Y_batchOffset, const int batchSize) {
  /*Check the conditions*/
  if ( X == NULL || Y == NULL || N <= 0 || incX <= 0 || incY <= 0 ) {
    return HCBLAS_INVALID;
  }

  dot = ddot_HC(accl_view, N, X, incX, xOffset, Y, incY, yOffset, dot, X_batchOffset, Y_batchOffset, batchSize);
  return HCBLAS_SUCCEEDS;
}

