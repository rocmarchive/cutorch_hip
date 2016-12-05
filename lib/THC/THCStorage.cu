#include "THCStorage.h"

#ifdef THRUST_PATH
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif
#endif

#include "THCHalf.h"

#include "generic/THCStorage.cu"
#include "THCGenerateAllTypes.h"
