#include "THCAllocator.h"

static void *THCudaHostAllocator_malloc(void* ctx, ptrdiff_t size) {
  void* ptr;

  if (size < 0) THError("Invalid memory size: %ld", size);

  if (size == 0) return NULL;

  THCudaCheck(hipHostMalloc(&ptr, size));

  return ptr;
}

static void THCudaHostAllocator_free(void* ctx, void* ptr) {
  if (!ptr) return;

  THCudaCheck(hipHostFree(ptr));
}

THAllocator THCudaHostAllocator = {
  &THCudaHostAllocator_malloc,
  NULL,
  &THCudaHostAllocator_free
};

static hipError_t THCIpcAllocator_malloc(void* ctx, void** devPtr, size_t size, hipStream_t stream)
{
  THError("THCIpcAllocator.malloc() not supported");
  return hipSuccess;
}

static hipError_t THCIpcAllocator_free(void* ctx, void* devPtr)
{
  return hipIpcCloseMemHandle(devPtr);
}

THCDeviceAllocator THCIpcAllocator = {
  &THCIpcAllocator_malloc,
  NULL,
  &THCIpcAllocator_free,
  NULL,
  NULL
};

static void *THCUVAAllocator_alloc(void* ctx, ptrdiff_t size) {
  if (size < 0) THError("Invalid memory size: %ld", size);

  if (size == 0) return NULL;

  // See J.1.1 of the CUDA_C_Programming_Guide.pdf for UVA and coherence rules
  // on various compute capabilities.
  void* ptr;
  // TODO: HIP_EQUIVALENT
  //THCudaCheck(hipMallocManaged(&ptr, size, hipMemAttachGlobal));
  return ptr;
}

static void THCUVAAllocator_free(void* ctx, void* ptr) {
  if (!ptr) return;
  THCudaCheck(hipFree(ptr));
}

THAllocator THCUVAAllocator = {
  &THCUVAAllocator_alloc,
  NULL,
  &THCUVAAllocator_free
};
