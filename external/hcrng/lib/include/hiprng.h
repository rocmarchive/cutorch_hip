/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

//! HIP = Heterogeneous-compute Interface for Portability
//!
//! Define a extremely thin runtime layer that allows source code to be compiled
//unmodified
//! through either AMD HCC or NVCC.   Key features tend to be in the spirit
//! and terminology of CUDA, but with a portable path to other accelerators as
//well.
//!
//!  This is the master include file for hiprng, wrapping around hcrng and
//curand "version 1"
//

#pragma once

enum hiprngStatus_t {
  HIPRNG_STATUS_SUCCESS = 0,                      //the operation completed successfully.
  HIPRNG_STATUS_ALLOCATION_FAILED = -1,            //resource allocation failed.
  HIPRNG_STATUS_INVALID_VALUE = -2,               //unsupported numerical value was passed to function. (hcRNG only)
  HIPRNG_STATUS_TYPE_ERROR = -3,            //unsupported rng type specified.
  HIPRNG_STATUS_INVALID_STREAM_CREATOR = -4,      //Stream creator is invalid.  (hcRNG only)
  HIPRNG_STATUS_INVALID_SEED = -5,                //Seed value is greater than particular generators’ predefined values. (hcRNG only)
  HIPRNG_STATUS_FUNCTION_NOT_IMPLEMENTED = -6,     //an internal hcRNG function not implemented.
  HIPRNG_STATUS_INITIALIZATION_FAILED = -7,        // if there was a problem setting up the GPU (cuRAND only)
  HIPRNG_STATUS_VERSION_MISMATCH = -8,             //if the header file version does not match the dynamically linked library version (cuRAND only)
  HIPRNG_STATUS_INTERNAL_ERROR = -9
};

enum hiprngRngType_t {
  HIPRNG_RNG_PSEUDO_MRG31K3P,
  HIPRNG_RNG_PSEUDO_MRG32K3A,
  HIPRNG_RNG_PSEUDO_LFSR113,
  HIPRNG_RNG_PSEUDO_PHILOX432,
  HIPRNG_RNG_PSEUDO_DEFAULT=0
};
// Some standard header files, these are included by hc.hpp and so want to make
// them avail on both
// paths to provide a consistent include env and avoid "missing symbol" errors
// that only appears
// on NVCC path:

#if defined(__HIP_PLATFORM_HCC__) and not defined(__HIP_PLATFORM_NVCC__)
#include <hcc_detail/hip_rng.h>
#elif defined(__HIP_PLATFORM_NVCC__) and not defined(__HIP_PLATFORM_HCC__)
#include <nvcc_detail/hip_rng.h>
#else
#error("Must define exactly one of __HIP_PLATFORM_HCC__ or __HIP_PLATFORM_NVCC__");
#endif

