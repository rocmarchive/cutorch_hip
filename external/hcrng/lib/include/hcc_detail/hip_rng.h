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
#pragma once
#include <iostream>
#include <hip/hip_runtime_api.h>
#include <hcRNG/hcRNG.h>
#include <hcRNG/mrg31k3p.h>
#include <hcRNG/mrg32k3a.h>
#include <hcRNG/lfsr113.h>
#include <hcRNG/philox432.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef void *hiprngGenerator_t;

inline static hiprngStatus_t hipHCRNGStatusToHIPStatus(hcrngStatus hcStatus) {
  switch (hcStatus) {
    case HCRNG_SUCCESS:
      return HIPRNG_STATUS_SUCCESS;
    case HCRNG_OUT_OF_RESOURCES:
      return HIPRNG_STATUS_ALLOCATION_FAILED;
    case HCRNG_INVALID_VALUE:
      return HIPRNG_STATUS_INVALID_VALUE;
    case HCRNG_INVALID_RNG_TYPE:
      return HIPRNG_STATUS_TYPE_ERROR;
    case HCRNG_INVALID_STREAM_CREATOR:
      return HIPRNG_STATUS_INVALID_STREAM_CREATOR;
    case HCRNG_INVALID_SEED:
      return HIPRNG_STATUS_INVALID_SEED;
    case HCRNG_FUNCTION_NOT_IMPLEMENTED:
      return HIPRNG_STATUS_FUNCTION_NOT_IMPLEMENTED;
    default:
      throw "Unimplemented status";
  }
}

inline static const char* hiprngGetErrorString() {
    return hcrngGetErrorString();
}

inline static const char* hiprngGetLibraryRoot() {
  return hcrngGetLibraryRoot();
}

inline static hiprngStatus_t hiprngSetErrorString(int err, const char* msg,
                                                  ...) {
  return hipHCRNGStatusToHIPStatus(hcrngSetErrorString(err, msg));
}

static int rngtyp; 

inline static hiprngStatus_t hiprngSetStream(hiprngGenerator_t generator, hipStream_t stream){
  return hipHCRNGStatusToHIPStatus(HCRNG_FUNCTION_NOT_IMPLEMENTED);
      
}
inline static hiprngStatus_t hiprngSetGeneratorOffset(hiprngGenerator_t generator, unsigned long long offset){
 return hipHCRNGStatusToHIPStatus(HCRNG_FUNCTION_NOT_IMPLEMENTED);
}

inline static hiprngStatus_t hiprngCreateGenerator(hiprngGenerator_t* generator,
                                                   hiprngRngType_t rng_type) {
  if(rng_type == 0) {
   *generator = (hcrngMrg31k3pStreamCreator*)malloc(sizeof(hcrngMrg31k3pStreamCreator));
   *(hcrngMrg31k3pStreamCreator*)*generator = defaultStreamCreator_Mrg31k3p;
   rngtyp = 0;
  }
  else if(rng_type == 1) {
   *generator = (hcrngMrg32k3aStreamCreator*)malloc(sizeof(hcrngMrg32k3aStreamCreator));
   *(hcrngMrg32k3aStreamCreator*)*generator = defaultStreamCreator_Mrg32k3a;
   rngtyp = 1;
  }
  else if(rng_type == 2) {
   *generator = (hcrngLfsr113StreamCreator*)malloc(sizeof(hcrngLfsr113StreamCreator));
   *(hcrngLfsr113StreamCreator*)*generator = defaultStreamCreator_Lfsr113;
   rngtyp = 2;
  }
 else if(rng_type == 3) {
   *generator = (hcrngPhilox432StreamCreator*)malloc(sizeof(hcrngPhilox432StreamCreator));
   *(hcrngPhilox432StreamCreator*)*generator = defaultStreamCreator_Philox432;
   rngtyp = 3;
 }
  return hipHCRNGStatusToHIPStatus(HCRNG_SUCCESS);  
}
#define SetSeed(gt) \
    if(temp##gt != 0) {\
      hcrng##gt##StreamState baseState;\
    for (size_t i = 0; i < 3; ++i)\
      baseState.g1[i] =  temp##gt;\
    for (size_t i = 0; i < 3; ++i)\
      baseState.g2[i] =  temp##gt;\
    return hipHCRNGStatusToHIPStatus(\
       hcrng##gt##SetBaseCreatorState((hcrng##gt##StreamCreator*)generator, &baseState));\
   }\
   else\
      return hipHCRNGStatusToHIPStatus(HCRNG_SUCCESS);

#define SetSeedLfsr113() \
    if(tempLfsr113 != 0) {\
      hcrngLfsr113StreamState baseState;\
    for (size_t i = 0; i < 4; ++i)\
      baseState.g[i] =  tempLfsr113;\
    return hipHCRNGStatusToHIPStatus(\
       hcrngLfsr113SetBaseCreatorState((hcrngLfsr113StreamCreator*)generator, &baseState));\
   }\
   else\
      return hipHCRNGStatusToHIPStatus(HCRNG_SUCCESS);

     
#define SetSeedPhilox432() \
    if(tempPhilox432 != 0) {\
      hcrngPhilox432StreamState baseState;\
      baseState.ctr.H.msb = tempPhilox432;\
      baseState.ctr.H.lsb = tempPhilox432;\
      baseState.ctr.L.msb = tempPhilox432;\
      baseState.ctr.L.lsb = tempPhilox432;\
      baseState.deckIndex = 0;\
    return hipHCRNGStatusToHIPStatus(\
       hcrngPhilox432SetBaseCreatorState((hcrngPhilox432StreamCreator*)generator, &baseState));\
    }\
   else\
      return hipHCRNGStatusToHIPStatus(HCRNG_SUCCESS);

inline static hiprngStatus_t hiprngSetPseudoRandomGeneratorSeed(
    hiprngGenerator_t generator, unsigned long long seed) {
  if(rngtyp == 0){
     unsigned long tempMrg31k3p = seed;
     SetSeed(Mrg31k3p)
  }
  else if(rngtyp == 1){
     unsigned int tempMrg32k3a = seed;
     SetSeed(Mrg32k3a)
  }
  else if(rngtyp == 2){
     unsigned int tempLfsr113 = seed;
     SetSeedLfsr113()
  }
  else if(rngtyp == 3){
     unsigned int tempPhilox432 = seed;
     SetSeedPhilox432()
  }
  return hipHCRNGStatusToHIPStatus(HCRNG_SUCCESS);
}

#undef SetSeed
#undef SetSeedLfsr113
#undef SetSeedPhilox432

  #define Generate(gt)\
  hcrng##gt##Stream *streams##gt = hcrng##gt##CreateStreams((hcrng##gt##StreamCreator*)generator, num, NULL, NULL); \
  unsigned int *outHost##gt = (unsigned int*)malloc(num * sizeof(unsigned int));\
  hcrngStatus hcStatus##gt = hcrng##gt##RandomUnsignedIntegerArray(streams##gt, 1, 4294967294, num, outHost##gt);\
  hipMemcpy(outputPtr, outHost##gt, num * sizeof(unsigned int), hipMemcpyHostToDevice);\
  free(streams##gt);\
  free(outHost##gt);\
  return hipHCRNGStatusToHIPStatus(hcStatus##gt); 

inline static hiprngStatus_t hiprngGenerate(hiprngGenerator_t generator,
                                              unsigned int* outputPtr,
                                                  size_t num) {
  if(rngtyp == 0){
    Generate(Mrg31k3p)
  }
  else if(rngtyp == 1){
    Generate(Mrg32k3a)
  }
  else if(rngtyp == 2){
    Generate(Lfsr113)
  }
  else if(rngtyp == 3){
    Generate(Philox432)
  }
  return hipHCRNGStatusToHIPStatus(HCRNG_SUCCESS);
}
#undef Generate

  #define GenerateUniform(gt)\
  hcrng##gt##Stream *streams##gt = hcrng##gt##CreateStreams((hcrng##gt##StreamCreator*)generator, num, NULL, NULL); \
  hcrng##gt##Stream *streams_buffer##gt;\
  hipMalloc((void **)&streams_buffer##gt, num * sizeof(hcrng##gt##Stream));\
  hipMemcpy(streams_buffer##gt, streams##gt, num * sizeof(hcrng##gt##Stream), hipMemcpyHostToDevice);\
  free(streams##gt);\
  hcrngStatus hcStatus##gt = hcrng##gt##DeviceRandomU01Array_single(\
       num, streams_buffer##gt, num, outputPtr);\
  hipFree(streams_buffer##gt);\
  return hipHCRNGStatusToHIPStatus(hcStatus##gt); 

inline static hiprngStatus_t hiprngGenerateUniform(hiprngGenerator_t generator,
                                                   float* outputPtr,
                                                   size_t num) {
  if(rngtyp == 0){
    GenerateUniform(Mrg31k3p)
  }
  else if(rngtyp == 1){
     GenerateUniform(Mrg32k3a)
  }
  else if(rngtyp == 2){
     GenerateUniform(Lfsr113)
  }
  else if(rngtyp == 3){
     GenerateUniform(Philox432)
  }
  return hipHCRNGStatusToHIPStatus(HCRNG_SUCCESS); 
}
  #undef GenerateUniform

  #define GenerateUniformDouble(gt)\
  hcrng##gt##Stream *streams##gt = hcrng##gt##CreateStreams((hcrng##gt##StreamCreator*)generator, num, NULL, NULL); \
  hcrng##gt##Stream *streams_buffer##gt;\
  hipMalloc((void **)&streams_buffer##gt, num * sizeof(hcrng##gt##Stream));\
  hipMemcpy(streams_buffer##gt, streams##gt, num * sizeof(hcrng##gt##Stream), hipMemcpyHostToDevice);\
  free(streams##gt);\
  hcrngStatus hcStatus##gt = hcrng##gt##DeviceRandomU01Array_double(\
       num, streams_buffer##gt, num, outputPtr);\
  hipFree(streams_buffer##gt);\
  return hipHCRNGStatusToHIPStatus(hcStatus##gt);

inline static hiprngStatus_t hiprngGenerateUniformDouble(
    hiprngGenerator_t generator, double* outputPtr, size_t num) {
  if(rngtyp == 0){
    GenerateUniformDouble(Mrg31k3p)
  }
  else if(rngtyp == 1){
     GenerateUniformDouble(Mrg32k3a)
  }
  else if(rngtyp == 2){
     GenerateUniformDouble(Lfsr113)
  }
  else if(rngtyp == 3){
     GenerateUniformDouble(Philox432)
  }
  return hipHCRNGStatusToHIPStatus(HCRNG_SUCCESS); 
}
  #undef GenerateUniformDouble

  #define GenerateNormal(gt)\
  hcrng##gt##Stream *streams##gt = hcrng##gt##CreateStreams((hcrng##gt##StreamCreator*)generator, num, NULL, NULL); \
  hcrng##gt##Stream *streams_buffer##gt;\
  hipMalloc((void **)&streams_buffer##gt, num * sizeof(hcrng##gt##Stream));\
  hipMemcpy(streams_buffer##gt, streams##gt, num * sizeof(hcrng##gt##Stream), hipMemcpyHostToDevice);\
  free(streams##gt);\
  hcrngStatus hcStatus##gt = hcrng##gt##DeviceRandomNArray_single(\
       num, streams_buffer##gt, num, mean, stddev, outputPtr);\
  hipFree(streams_buffer##gt);\
  return hipHCRNGStatusToHIPStatus(hcStatus##gt);


inline static hiprngStatus_t hiprngGenerateNormal(hiprngGenerator_t generator,
                                                   float* outputPtr,
                                                   size_t num, float mean, float stddev) {
  if(rngtyp == 0){
    GenerateNormal(Mrg31k3p)
  }
  else if(rngtyp == 1){
     GenerateNormal(Mrg32k3a)
  }
  else if(rngtyp == 2){
     GenerateNormal(Lfsr113)
  }
  else if(rngtyp == 3){
     GenerateNormal(Philox432)
  }
  return hipHCRNGStatusToHIPStatus(HCRNG_SUCCESS); 
}
  #undef GenerateNormal

  #define GenerateNormalDouble(gt)\
  hcrng##gt##Stream *streams##gt = hcrng##gt##CreateStreams((hcrng##gt##StreamCreator*)generator, num, NULL, NULL); \
  hcrng##gt##Stream *streams_buffer##gt;\
  hipMalloc((void **)&streams_buffer##gt, num * sizeof(hcrng##gt##Stream));\
  hipMemcpy(streams_buffer##gt, streams##gt, num * sizeof(hcrng##gt##Stream), hipMemcpyHostToDevice);\
  free(streams##gt);\
  hcrngStatus hcStatus##gt = hcrng##gt##DeviceRandomNArray_double(\
       num, streams_buffer##gt, num, mean, stddev, outputPtr);\
  hipFree(streams_buffer##gt);\
  return hipHCRNGStatusToHIPStatus(hcStatus##gt);



inline static hiprngStatus_t hiprngGenerateNormalDouble(
    hiprngGenerator_t generator, double* outputPtr, size_t num, double mean, double stddev) {

  if(rngtyp == 0){
    GenerateNormalDouble(Mrg31k3p)
  }
  else if(rngtyp == 1){
     GenerateNormalDouble(Mrg32k3a)
  }
  else if(rngtyp == 2){
     GenerateNormalDouble(Lfsr113)
  }
  else if(rngtyp == 3){
     GenerateNormalDouble(Philox432)
  }
  return hipHCRNGStatusToHIPStatus(HCRNG_SUCCESS); 
}


  #define Destroy(gt)\
  return hipHCRNGStatusToHIPStatus(hcrng##gt##DestroyStreamCreator((hcrng##gt##StreamCreator*)generator));


inline static hiprngStatus_t hiprngDestroyGenerator(hiprngGenerator_t generator){
  
  if(rngtyp == 0){
    Destroy(Mrg31k3p)
  }
  else if(rngtyp == 1){
    Destroy(Mrg32k3a)
  }
  else if(rngtyp == 2){
    Destroy(Lfsr113)
  }
  else if(rngtyp == 3){
    Destroy(Philox432)
  }
  return hipHCRNGStatusToHIPStatus(HCRNG_SUCCESS);
}


 #undef Destroy

#ifdef __cplusplus
}
#endif

