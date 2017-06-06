#include "THCSleep.h"

__global__ void spin_kernel(long long cycles)
{
  // see concurrentKernels CUDA sampl
#ifdef __NVCC__
  long long start_clock = clock64();
#else
  long long start_clock = clock();
#endif
  long long clock_offset = 0;
  while (clock_offset < cycles)
  {
#ifdef __NVCC__
    clock_offset = clock64() - start_clock;
#else
    clock_offset = clock() - start_clock;
#endif
  }
}

THC_API void THC_sleep(THCState* state, long long cycles)
{
  dim3 grid(1);
  dim3 block(1);
  hipLaunchKernelGGL(spin_kernel, grid, block, 0, THCState_getCurrentStream(state), cycles);
  THCudaCheck(hipGetLastError());
}
