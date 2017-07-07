#ifndef THC_TENSOR_RANDOM_CUH
#define THC_TENSOR_RANDOM_CUH

#include "THCNumerics.cuh"
#include "THCReduceApplyUtils.cuh"
#include "THCTensorMathReduce.cuh"

#ifdef CURAND_PATH
#include <curand_kernel.h>
#else
  #include <hip/hip_hcc.h>
  #include "hiprng.h"
#endif
#define MAX_NUM_BLOCKS 64
#define BLOCK_SIZE 256
/* Separate kernel because curand_log_normal gets extra parameters. */

#ifdef CURAND_PATH
template <typename T>
__global__ void generateLogNormal(curandStateMtgp32 *state, int size, T *result, double mean, double stddev)
{
  int idx = hipBlockIdx_x * BLOCK_SIZE + hipThreadIdx_x;
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {
    float x = curand_log_normal(&state[hipBlockIdx_x], mean, stddev);
    if (i < size) {
      result[i] = ScalarConvert<float, T>::to(x);
    }
  }
}

template <>
__global__ void generateLogNormal<double>(curandStateMtgp32 *state, int size, double *result, double mean, double stddev)
{
  int idx = hipBlockIdx_x * BLOCK_SIZE + hipThreadIdx_x;
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {
    double x = curand_log_normal_double(&state[hipBlockIdx_x], mean, stddev);
    if (i < size) {
      result[i] = x;
    }
  }
}
#endif

#undef MAX_NUM_BLOCKS
#undef BLOCK_SIZE

// Normalizes the L1 norm of every row to 1; used by multinomial
template <typename T>
__global__ void renormRowsL1(T* dist, long rows, long cols) {
#ifdef CURAND_PATH
  extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
#else
  // TODO: Check the vulnerability of this change
  extern __shared__ unsigned char my_smem[];
#endif
  T *smem = reinterpret_cast<T *>(my_smem);

  for (long row = hipBlockIdx_x; row < rows; row += hipGridDim_x) {
    T sum = ScalarConvert<int, T>::to(0);
    for (long col = hipThreadIdx_x; col < cols; col += hipBlockDim_x) {
      sum = THCNumerics<T>::add(sum, dist[row * cols + col]);
    }

    sum = reduceBlock(smem, hipBlockDim_x, sum, ReduceAdd<T, T>(), ScalarConvert<int, T>::to(0));
    if (hipThreadIdx_x == 0) {
      smem[0] = sum;
    }
    __syncthreads();

    sum = smem[0];
    if (THCNumerics<T>::gt(sum, ScalarConvert<int, T>::to(0))) {
      for (long col = hipThreadIdx_x; col < cols; col += hipBlockDim_x) {
        dist[row * cols + col] = THCNumerics<T>::div(dist[row * cols + col], sum);
      }
    }
  }
}

template <typename T>
__device__ int binarySearchForMultinomial(T* dist,
                                          int size,
                                          T val) {
  int start = 0;
  int end = size;

  while (end - start > 0) {
    int mid = start + (end - start) / 2;

    T midVal = dist[mid];
    if (THCNumerics<T>::lt(midVal, val)) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }

  if (start == size) {
    // No probability mass or precision problems; just return the
    // first element
    start = 0;
  }

  T curVal = dist[start];
  while(start >= 1 && THCNumerics<T>::eq(dist[start - 1], curVal)) start--;

  return start;
}

template <typename T, typename AccT>
__global__ void
sampleMultinomialOnce(long* dest,
                      long distributions,
                      int categories,
                      T* sampled,
                      T* dist) {
#ifdef CURAND_PATH 
  extern __shared__ __align__(sizeof(AccT)) unsigned char my_smem[];
#else
  // TODO: Check vulnerability of this change
  extern __shared__  unsigned char my_smem[];
#endif
  __shared__ bool found;

  // Shared Memory hold blockdim.x T for holding the cumulative sum,
  // hipBlockDim_x AccT for normalizing the probabilities,
  T *smem = reinterpret_cast<T *>(my_smem);
  AccT *asmem = reinterpret_cast<AccT *>(&my_smem[hipBlockDim_x * sizeof(T)]);

  AccT accZero = ScalarConvert<int, AccT>::to(0);
  T zero = ScalarConvert<int, T>::to(0);

  for (long curDist = hipBlockIdx_x;
       curDist < distributions; curDist += hipGridDim_x) {
    // Each block handles one distribution
    // First pass, find the total sum of the distribution
    AccT sum = accZero;
    for (int cat = hipThreadIdx_x; cat < categories; cat += hipBlockDim_x) {
      sum = THCNumerics<AccT>::add(
        sum,
        ScalarConvert<T, AccT>::to(dist[curDist * categories + cat]));
    }

    // hipThreadIdx_x == 0 has the sum value from this
    sum = reduceBlock(asmem, hipBlockDim_x, sum, ReduceAdd<AccT, AccT>(), accZero);

    // Broadcast sum and sample value
    if (hipThreadIdx_x == 0) {
      // Make sure the sum of our distribution didn't overflow
     #ifdef CUDA_PATH
      assert(!isinf(sum));
     #endif

      asmem[0] = sum;
      smem[0] = sampled[curDist];
    }
    __syncthreads();

    sum = asmem[0];
    T sample = smem[0];
    __syncthreads();

    if (THCNumerics<AccT>::eq(sum,  accZero) || THCNumerics<T>::eq(sample, zero)) {
      // Choose the first element
      if (hipThreadIdx_x == 0) {
        dest[curDist] = TH_INDEX_BASE;
      }

      continue;
    }

    int chunks = THCCeilDiv(categories, (int) hipBlockDim_x);
    T prevHighProb = zero;
    found = false;

    for (int chunk = 0; chunk < chunks && !found; ++chunk) {
      // All threads in bounds load a value
      int cat = chunk * hipBlockDim_x + hipThreadIdx_x;

      AccT val =
        cat < categories ?
          THCNumerics<AccT>::div(
              ScalarConvert<T, AccT>::to(dist[curDist * categories + cat]),
              sum) :
          accZero;

      smem[hipThreadIdx_x] = ScalarConvert<AccT, T>::to(val);
      __syncthreads();

      // Perform an inclusive prefix sum of the shared memory contents
      for (int offset = 1; offset < hipBlockDim_x; offset *= 2) {
        T val = zero;

        if (hipThreadIdx_x >= offset) {
          val = THCNumerics<T>::add(smem[hipThreadIdx_x - offset], smem[hipThreadIdx_x]);
        }

        __syncthreads();
        if (hipThreadIdx_x >= offset) {
          smem[hipThreadIdx_x] = val;
        }
        __syncthreads();
      }

      // Each thread will check to see if the sample falls in its
      // bucket
      T curBucket = THCNumerics<T>::add(smem[hipThreadIdx_x], prevHighProb);
      T prevBucket =
        hipThreadIdx_x == 0 ? prevHighProb :
        THCNumerics<T>::add(smem[hipThreadIdx_x - 1], prevHighProb);
      bool inBucket =
        (cat < categories) &&
        (!THCNumerics<T>::gt(sample, curBucket)) &&
        (THCNumerics<T>::gt(sample, prevBucket));

      if (inBucket) {
        // We're done; we have the sample
        // Torch indices are 1-based
        dest[curDist] = cat + TH_INDEX_BASE;
        found = true;
      }

      // Store the previous scan's high value for future use
      prevHighProb = THCNumerics<T>::add(prevHighProb, smem[hipBlockDim_x - 1]);

      __syncthreads();
    }

    if (hipThreadIdx_x == 0 && !found) {
      // This should address a rare bug where we don't select a valid index. This likely occurs when
      // due to floating point arithmetic rounding errors, our cumulative sum does not add up to 1, but
      // and our uniform sample is greater than this value. In this case we likely have unitialized memory
      // in dest[curDist]. So basically we will loop through the distribution and pick the largest index
      // where the distribution is non-zero. This is obviously terribly inefficient, but due to the
      // rarity in which this occurs, this should not be an issue.
      for (int cat = categories - 1; cat >= 0; --cat) {
        if (THCNumerics<T>::gt(dist[curDist * categories + cat], zero)) {
          dest[curDist] = cat + TH_INDEX_BASE;
          break;
        }
      }
    }
  }
}

#ifdef CURAND_PATH
template <typename T>
__global__ void
sampleMultinomialWithReplacement(curandStateMtgp32* state,
                                 int totalSamples,
                                 long* dest,
                                 long distributions,
                                 int categories,
                                 T* normDistPrefixSum) {
#else
template <typename T>
__global__ void
sampleMultinomialWithReplacement(hiprngStateMtgp32* state,
                                 int totalSamples,
                                 long* dest,
                                 long distributions,
                                 int categories,
                                 T* normDistPrefixSum) {

#endif
  // At the moment, each warp computes one sample value in the binary
  // search due to divergence. It seems possible to compute multiple
  // values and limit divergence though later on. However, no matter
  // what, all block threads must participate in the curand_uniform
  // call to update the generator state.

  // The block determines the distribution for which we generate a point
  for (long curDist = hipBlockIdx_x;
       curDist < distributions;
       curDist += hipGridDim_x) {
    for (int sampleBase = 0;
         sampleBase < totalSamples; sampleBase += hipBlockDim_y) {
      // The warp determines the sample
      int sample = sampleBase + hipThreadIdx_y;

      // All threads participate in this
      #ifdef CURAND_PATH
        T r = ScalarConvert<float, T>::to(curand_uniform(&state[hipBlockIdx_x]));
      #else
        T r = ScalarConvert<float, T>::to(hcrng_uniform((&state[hipBlockIdx_x]), hipThreadIdx_x));
      #endif

      if (hipThreadIdx_x == 0 && sample < totalSamples) {
        // Find the bucket that a uniform sample lies in
        int choice = binarySearchForMultinomial<T>(
          normDistPrefixSum + curDist * categories,
          categories,
          r);

        // Torch indices are 1-based
        dest[curDist * totalSamples + sample] = choice + TH_INDEX_BASE;
      }
    }
  }
}

template <typename T>
#ifdef CURAND_PATH
__global__ void
sampleMultinomialWithoutReplacement(curandStateMtgp32* state,
                                    int totalSamples,
                                    int sample,
                                    long* dest,
                                    long distributions,
                                    int categories,
                                    T* origDist,
                                    T* normDistPrefixSum) {
#else
__global__ void
sampleMultinomialWithoutReplacement(hiprngStateMtgp32* state,
                                    int totalSamples,
                                    int sample,
                                    long* dest,
                                    long distributions,
                                    int categories,
                                    T* origDist,
                                    T* normDistPrefixSum) {
#endif
  // At the moment, each warp computes one sample value in the binary
  // search due to divergence. It seems possible to compute multiple
  // values and limit divergence though later on. However, no matter
  // what, all block threads must participate in the curand_uniform
  // call to update the generator state.

  // The block and warp determines the distribution for which we
  // generate a point
  for (long curDistBase = hipBlockIdx_x * hipBlockDim_y;
       curDistBase < distributions;
       curDistBase += hipGridDim_x * hipBlockDim_y) {
    // The warp determines the distribution
    long curDist = curDistBase + hipThreadIdx_y;

#ifdef CURAND_PATH
    // All threads must participate in this
    T r = ScalarConvert<float, T>::to(curand_uniform(&state[hipBlockIdx_x]));
#else
    T r = ScalarConvert<float, T>::to(hcrng_uniform(&state[hipBlockIdx_x], hipThreadIdx_x));
#endif
    if (hipThreadIdx_x == 0 && curDist < distributions) {
      // Find the bucket that a uniform sample lies in
      int choice = binarySearchForMultinomial<T>(
        normDistPrefixSum + curDist * categories,
        categories,
        r);

      // Torch indices are 1-based
      dest[curDist * totalSamples + sample] = choice + TH_INDEX_BASE;

      // Without replacement, so update the original probability so it
      // is not considered a second time
      origDist[curDist * categories + choice] = ScalarConvert<int, T>::to(0);
    }
  }
}

#endif // THC_TENSOR_RANDOM_CUH
