#ifndef THC_ASM_UTILS_INC
#define THC_ASM_UTILS_INC

// Collection of direct PTX functions
#if defined(__HIP_PLATFORM_HCC__)
  #include <climits>
  #include <cstdint>
#endif

#if defined(__HIP_PLATFORM_HCC__)
  __device__
  inline
  unsigned int getBitfield(unsigned int val, int pos, int len)
  {
    pos &= 0x1f;
    len &= 0x1f;

    unsigned int m = (1u << len) - 1u;
    m <<= pos;
    return val & m;
  }
#else
  __device__ __forceinline__
  unsigned int getBitfield(unsigned int val, int pos, int len)
  {
    unsigned int ret;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
    return ret;
  }
#endif

#if defined(__HIP_PLATFORM_HCC__)
  __device__
  inline
  unsigned int setBitfield(
    unsigned int val, unsigned int toInsert, int pos, int len)
  {
    pos &= 0x1f;
    len &= 0x1f;

    unsigned int m = (1u << len) - 1u;
    toInsert &= m;
    toInsert <<= pos;
    m <<= pos;

    return (val & ~m) | toInsert;
  }
#else
  __device__ __forceinline__
  unsigned int setBitfield(
      unsigned int val, unsigned int toInsert, int pos, int len)
  {
    unsigned int ret;
    asm("bfi.b32 %0, %1, %2, %3, %4;" :
        "=r"(ret) : "r"(toInsert), "r"(val), "r"(pos), "r"(len));
    return ret;
  }
#endif

__device__ __forceinline__
int getLaneId()
{
  #if defined(__HIP_PLATFORM_HCC__)
    return hc::__lane_id();
  #else
    int laneId;
    asm("mov.s32 %0, %laneid;" : "=r"(laneId) );
    return laneId;
  #endif
}

#if defined(__HIP_PLATFORM_HCC__)
  __device__
  inline
  std::uint64_t getLaneMaskLt()
  {
    std::uint64_t m = (1ull << getLaneId()) - 1ull;
    return m;
  }
#else
  __device__ __forceinline__
  unsigned int getLaneMaskLt()
  {
    unsigned int mask;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
    return mask;
  }
#endif

#if defined(__HIP_PLATFORM_HCC__)
  __device__
  inline
  std::uint64_t getLaneMaskLe()
  {
    std::uint64_t m = (1ull << (getLaneId() + 1ull)) - 1ull;
    return m;
  }
#else
  __device__ __forceinline__
  unsigned int getLaneMaskLe()
  {
    unsigned int mask;
    asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
    return mask;
  }
#endif

#if defined(__HIP_PLATFORM_HCC__)
  __device__
  inline
  std::uint64_t getLaneMaskGt()
  {
    std::uint64_t m = getLaneMaskLe();
    return m ? ~m : m;
  }
#else
  __device__ __forceinline__
  unsigned int getLaneMaskGt()
  {
    unsigned int mask;
    asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
    return mask;
  }
#endif

#if defined(__HIP_PLATFORM_HCC__)
  __device__
  inline
  std::uint64_t getLaneMaskGe()
  {
    std::uint64_t m = getLaneMaskLt();
    return ~m;
  }
#else
  __device__ __forceinline__
  unsigned int getLaneMaskGe()
  {
    unsigned int mask;
    asm("mov.u32 %0, %%lanemask_ge;" : "=r"(mask));
    return mask;
  }
#endif

#endif // THC_ASM_UTILS_INC
