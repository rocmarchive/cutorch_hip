#include "hip/hip_runtime.h"
#include "THC.h"
#include "THCReduceApplyUtils.cuh"
#include "THCTensorCopy.h"
#include "THCTensorMath.h"
#include "THCAsmUtils.cuh"
#include "THCScanUtils.cuh"
#include "THCTensorTypeUtils.cuh"
#include "THCTensorMathReduce.cuh"
#include <algorithm> // for std::min
#ifdef CUDA_PATH
    #if CUDA_VERSION >= 7000
        #include <thrust/system/cuda/execution_policy.h>
    #endif
#endif
#include "THCTensorTopK.cuh"

#include "generic/THCTensorTopK.cu"
#include "THCGenerateAllTypes.h"
