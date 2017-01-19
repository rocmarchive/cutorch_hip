#ifndef THC_ASM_UTILS_INC
#define THC_ASM_UTILS_INC

// Collection of direct PTX functions
#ifdef CUDA_PATH

// TODO: these map straightforwardly to AMD intrinsincs, which we should use.
//       In general, we should provide a generic function which we subsequently
//       overload for the NVIDIA PTX / AMD intrinsinc case.
__device__ __forceinline__
static
unsigned int getBitfield(unsigned int val, int pos, int len) {
  unsigned int ret;
  asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
  return ret;
}

__device__ __forceinline__
static
unsigned int setBitfield(unsigned int val, unsigned int toInsert, int pos, int len) {
  unsigned int ret;
  asm("bfi.b32 %0, %1, %2, %3, %4;" :
      "=r"(ret) : "r"(toInsert), "r"(val), "r"(pos), "r"(len));
  return ret;
}

__device__ __forceinline__
static
int getLaneId() {
  int laneId;
  asm("mov.s32 %0, %laneid;" : "=r"(laneId) );
  return laneId;
}

__device__ __forceinline__
static
unsigned getLaneMaskLt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__
static
unsigned getLaneMaskLe() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__
static
unsigned getLaneMaskGt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__
static
unsigned getLaneMaskGe() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_ge;" : "=r"(mask));
  return mask;
}
#endif

#endif // THC_ASM_UTILS_INC
