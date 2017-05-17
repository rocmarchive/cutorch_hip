#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/CStorage.c"
#else

#include "THHalf.h"

/* everything is as the generic Storage.c, except few things (see below) */

// FixMe: Requires an unsafe conversion in that we convert from cutorch's 'half'
// to torch's THHalf.  These types are required to be defined in the same way
// (is there some way to enforce this?)
#ifdef THC_REAL_IS_HALF
#define THFILE_REAL_CAST(x) (THHalf *)x
#else
#define THFILE_REAL_CAST(x) x
#endif

#define THFile_readRealRaw(file, data, size)                            \
  {                                                                     \
    real *fdata = (real*)THAlloc(sizeof(real)*size);                    \
    TH_CONCAT_3(THFile_read,Real,Raw)(file, THFILE_REAL_CAST(fdata), size);               \
    THCudaCheck(hipMemcpy(data, fdata, size * sizeof(real), hipMemcpyHostToDevice)); \
    THFree(fdata);                                                      \
  }

#define THFile_writeRealRaw(file, data, size)                           \
  {                                                                     \
    real *fdata = (real*)THAlloc(sizeof(real)*size);                    \
    THCudaCheck(hipMemcpy(fdata, data, size * sizeof(real), hipMemcpyDeviceToHost)); \
    TH_CONCAT_3(THFile_write,Real,Raw)(file, THFILE_REAL_CAST(fdata), size);              \
    THFree(fdata);                                                      \
  }


#define TH_GENERIC_FILE "generic/Storage.c"
#include "generic/Storage.c"

#undef TH_GENERIC_FILE
#undef THFILE_REAL_CAST
#undef THFile_readRealRaw
#undef THFile_writeRealRaw

/* now we overwrite some methods specific to CudaStorage */

static int cutorch_Storage_(copy)(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  THCStorage *storage =(THCStorage *)luaT_checkudata(L, 1, torch_Storage);
  void *src;
  if( (src = luaT_toudata(L, 2, "torch.CudaByteStorage")) )
    THCStorage_(copyCudaByte)(state, storage, (THCudaByteStorage *)src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaCharStorage")) )
    THCStorage_(copyCudaChar)(state, storage, (THCudaCharStorage *)src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaShortStorage")) )
    THCStorage_(copyCudaShort)(state, storage, (THCudaShortStorage *)src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaIntStorage")) )
    THCStorage_(copyCudaInt)(state, storage, (THCudaIntStorage *)src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaLongStorage")) )
    THCStorage_(copyCudaLong)(state, storage, (THCudaLongStorage *)src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaStorage")) )
    THCStorage_(copyCudaFloat)(state, storage, (THCudaStorage *)src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaDoubleStorage")) )
    THCStorage_(copyCudaDouble)(state, storage, (THCudaDoubleStorage *)src);
#ifdef CUDA_HALF_TENSOR
  else if( (src = luaT_toudata(L, 2, "torch.CudaHalfStorage")) )
    THCStorage_(copyCudaHalf)(state, storage, (THCudaHalfStorage *)src);
#endif

  else if( (src = luaT_toudata(L, 2, "torch.ByteStorage")) )
    THCStorage_(copyByte)(state, storage, (THByteStorage *)src);
  else if( (src = luaT_toudata(L, 2, "torch.CharStorage")) )
    THCStorage_(copyChar)(state, storage, (THCharStorage *)src);
  else if( (src = luaT_toudata(L, 2, "torch.ShortStorage")) )
    THCStorage_(copyShort)(state, storage, (THShortStorage *)src);
  else if( (src = luaT_toudata(L, 2, "torch.IntStorage")) )
    THCStorage_(copyInt)(state, storage, (THIntStorage *)src);
  else if( (src = luaT_toudata(L, 2, "torch.LongStorage")) )
    THCStorage_(copyLong)(state, storage, (THLongStorage *)src);
  else if( (src = luaT_toudata(L, 2, "torch.FloatStorage")) )
    THCStorage_(copyFloat)(state, storage, (THFloatStorage *)src);
  else if( (src = luaT_toudata(L, 2, "torch.DoubleStorage")) )
    THCStorage_(copyDouble)(state, storage, (THDoubleStorage *)src);
  else if( (src = luaT_toudata(L, 2, "torch.HalfStorage")) )
    THCStorage_(copyHalf)(state, storage, (THHalfStorage *)src);

  else
    luaL_typerror(L, 2, "torch.*Storage");

  lua_settop(L, 1);
  return 1;
}

static int cutorch_Storage_(getDevice)(lua_State *L) {
  THCStorage *storage = (THCStorage *)luaT_checkudata(L, 1, torch_Storage);
  lua_pushinteger(L, THCStorage_(getDevice)(cutorch_getstate(L), storage) + 1);
  return 1;
}

void cutorch_Storage_(init)(lua_State* L)
{
  /* the standard stuff */
  torch_Storage_(init)(L);

  // Register this even though it is generated elsewhere.
  cutorch_StorageCopy_(init)(L);

  luaT_pushmetatable(L, torch_Storage);
  lua_pushcfunction(L, cutorch_Storage_(copy));
  lua_setfield(L, -2, "copy");
  lua_pop(L, 1);

  luaT_pushmetatable(L, torch_Storage);
  lua_pushcfunction(L, cutorch_Storage_(getDevice));
  lua_setfield(L, -2, "getDevice");
  lua_pop(L, 1);
}

#endif
