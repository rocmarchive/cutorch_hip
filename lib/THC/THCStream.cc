#include "THCStream.h"

#include "THAtomic.h"

#include <hip/hip_runtime_api.h>

THCStream* THCStream_new(int flags)
{
  THCStream* self = (THCStream*) malloc(sizeof(THCStream));
  self->refcount = 1;
  THCudaCheck(hipGetDevice(&self->device));
  THCudaCheck(hipStreamCreateWithFlags(&self->stream, flags));
  return self;
}

void THCStream_free(THCStream* self)
{
  if (!self) {
    return;
  }
  if (THAtomicDecrementRef(&self->refcount)) {
    THCudaCheck(hipStreamDestroy(self->stream));
    free(self);
  }
}

void THCStream_retain(THCStream* self)
{
  THAtomicIncrementRef(&self->refcount);
}
