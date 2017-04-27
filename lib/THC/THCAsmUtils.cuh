#ifndef THC_ASM_UTILS_INC
#define THC_ASM_UTILS_INC

// Collection of direct PTX functions

__device__ __forceinline__
unsigned int getBitfield(unsigned int val, int pos, int len) {
  unsigned int ret;
#ifdef CUDA_PATH
  asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
#endif
  return ret;
}

__device__ __forceinline__
unsigned int setBitfield(unsigned int val, unsigned int toInsert, int pos, int len) {
  unsigned int ret;
#ifdef CUDA_PATH
  asm("bfi.b32 %0, %1, %2, %3, %4;" :
      "=r"(ret) : "r"(toInsert), "r"(val), "r"(pos), "r"(len));
#endif
  return ret;
}

__device__ __forceinline__ int getLaneId() {
  int laneId;
#ifdef CUDA_PATH
  asm("mov.s32 %0, %laneid;" : "=r"(laneId) );
#endif
  return laneId;
}

__device__ __forceinline__ unsigned getLaneMaskLt() {
  unsigned mask;
#ifdef CUDA_PATH
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
#endif
  return mask;
}

__device__ __forceinline__ unsigned getLaneMaskLe() {
  unsigned mask;
#ifdef CUDA_PATH
  asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
#endif
  return mask;
}

__device__ __forceinline__ unsigned getLaneMaskGt() {
  unsigned mask;
#ifdef CUDA_PATH
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
#endif
  return mask;
}

__device__ __forceinline__ unsigned getLaneMaskGe() {
  unsigned mask;
#ifdef CUDA_PATH
  asm("mov.u32 %0, %%lanemask_ge;" : "=r"(mask));
#endif
  return mask;
}

#endif // THC_ASM_UTILS_INC
