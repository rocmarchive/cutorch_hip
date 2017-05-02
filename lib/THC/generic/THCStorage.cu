#ifndef THC_GENERIC_FILE
#include "hip/hip_runtime.h"
#define THC_GENERIC_FILE "generic/THCStorage.cu"
#else

#ifdef __HCC__
#include <bolt/amp/fill.h>
#include <bolt/amp/iterator/ubiquitous_iterator.h>
#endif

void THCStorage_(fill)(THCState *state, THCStorage *self, real value)
{
#ifdef THRUST_PATH
  thrust::device_ptr<real> self_data(self->data);
  thrust::fill(
    self_data, self_data+self->size, value);
#else
    bolt::amp::fill(
        bolt::amp::make_ubiquitous_iterator(self->data),
        bolt::amp::make_ubiquitous_iterator(self->data) + self->size,
        value);
#endif
}

void THCStorage_(resize)(THCState *state, THCStorage *self, ptrdiff_t size)
{
  THArgCheck(size >= 0, 2, "invalid size");
  THAssert(self->allocator != NULL);
  int device;
  THCudaCheck(hipGetDevice(&device));

  if(!(self->flag & TH_STORAGE_RESIZABLE))
    THError("Trying to resize storage that is not resizable");

  if (self->allocator->realloc) {
    THCHeapUpdate(state, (size - self->size) * sizeof(real));
    hipError_t err = (*self->allocator->realloc)(
      self->allocatorContext,
      (void**)&(self->data),
      self->size * sizeof(real),
      size * sizeof(real), THCState_getCurrentStream(state));
    if (err != hipSuccess) {
      THCHeapUpdate(state, (self->size - size) * sizeof(real));
      THCudaCheck(err);
    }
    self->size = size;
    self->device = device;
    return;
  }

  if(size == 0)
  {
    if(self->flag & TH_STORAGE_FREEMEM) {
      THCudaCheck(
        (*self->allocator->free)(self->allocatorContext, self->data));
      THCHeapUpdate(state, -self->size * sizeof(real));
    }
    self->data = NULL;
    self->size = 0;
    self->device = device;
  }
  else
  {
    real *data = NULL;
    // update heap *before* attempting malloc, to free space for the malloc
    THCHeapUpdate(state, size * sizeof(real));
    hipError_t err =
      (*self->allocator->malloc)(self->allocatorContext,
                                 (void**)&(data),
                                 size * sizeof(real),
                                 THCState_getCurrentStream(state));

    if(err != hipSuccess) {
      THCHeapUpdate(state, -size * sizeof(real));
    }
    THCudaCheck(err);

    if (self->data) {
      THCudaCheck(hipMemcpyAsync(data,
                                  self->data,
                                  THMin(self->size, size) * sizeof(real),
                                  hipMemcpyDeviceToDevice,
                                  THCState_getCurrentStream(state)));
      if(self->flag & TH_STORAGE_FREEMEM) {
        if(self->data) {
          THCudaCheck(
          (*self->allocator->free)(self->allocatorContext, self->data));
        }
        THCHeapUpdate(state, -self->size * sizeof(real));
      }
    }

    self->data = data;
    self->size = size;
    self->device = device;
  }
}

THC_API int THCStorage_(getDevice)(THCState* state, const THCStorage* storage) {
  return storage->device;
}

#endif
