#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/CTensorCopy.c"
#else

static int TH_CONCAT_3(cutorch_,Real,Tensor_copy)(lua_State *L)
{
  THTensor *tensor = (THTensor*)luaT_checkudata(L, 1, TH_CONCAT_STRING_3(torch.,Real,Tensor));
  void *src;
  if( (src = luaT_toudata(L, 2, TH_CONCAT_STRING_3(torch.,Real,Tensor)) ))
    THTensor_(copy)(tensor, (THTensor*)src);
  else if( (src = luaT_toudata(L, 2, "torch.ByteTensor")) )
    THTensor_(copyByte)(tensor, (THByteTensor*)src);
  else if( (src = luaT_toudata(L, 2, "torch.CharTensor")) )
    THTensor_(copyChar)(tensor, (THCharTensor*)src);
  else if( (src = luaT_toudata(L, 2, "torch.ShortTensor")) )
    THTensor_(copyShort)(tensor, (THShortTensor*)src);
  else if( (src = luaT_toudata(L, 2, "torch.IntTensor")) )
    THTensor_(copyInt)(tensor, (THIntTensor*)src);
  else if( (src = luaT_toudata(L, 2, "torch.LongTensor")) )
    THTensor_(copyLong)(tensor, (THLongTensor*)src);
  else if( (src = luaT_toudata(L, 2, "torch.FloatTensor")) )
    THTensor_(copyFloat)(tensor, (THFloatTensor*)src);
  else if( (src = luaT_toudata(L, 2, "torch.DoubleTensor")) )
    THTensor_(copyDouble)(tensor, (THDoubleTensor*)src);
  else if( (src = luaT_toudata(L, 2, "torch.HalfTensor")) )
    THTensor_(copyHalf)(tensor, (THHalfTensor*)src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaByteTensor")) )
    THTensor_(copyCudaByte)(cutorch_getstate(L), tensor, (THCudaByteTensor*)src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaCharTensor")) )
    THTensor_(copyCudaChar)(cutorch_getstate(L), tensor, (THCudaCharTensor*)src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaShortTensor")) )
    THTensor_(copyCudaShort)(cutorch_getstate(L), tensor, (THCudaShortTensor*)src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaIntTensor")) )
    THTensor_(copyCudaInt)(cutorch_getstate(L), tensor, (THCudaIntTensor*)src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaLongTensor")) )
    THTensor_(copyCudaLong)(cutorch_getstate(L), tensor, (THCudaLongTensor*)src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaTensor")) )
    THTensor_(copyCudaFloat)(cutorch_getstate(L), tensor, (THCudaTensor*)src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaDoubleTensor")) )
    THTensor_(copyCudaDouble)(cutorch_getstate(L), tensor, (THCudaDoubleTensor*)src);
#ifdef CUDA_HALF_TENSOR
  else if( (src = luaT_toudata(L, 2, "torch.CudaHalfTensor")) )
    THTensor_(copyCudaHalf)(cutorch_getstate(L), tensor, (THCudaHalfTensor*)src);
#endif
  else
    luaL_typerror(L, 2, "torch.*Tensor");

  lua_settop(L, 1);
  return 1;
}

void cutorch_TensorCopy_(init)(lua_State* L)
{
  luaT_pushmetatable(L, TH_CONCAT_STRING_3(torch.,Real,Tensor));
  lua_pushcfunction(L, TH_CONCAT_3(cutorch_,Real,Tensor_copy));
  lua_setfield(L, -2, "copy");
  lua_pop(L, 1);
}

#endif
