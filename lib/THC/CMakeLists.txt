CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH} "${HIP_PATH}/cmake")
EXECUTE_PROCESS(COMMAND ${HIP_PATH}/bin/hipconfig -P OUTPUT_VARIABLE PLATFORM)

IF(NOT Torch_FOUND)
  FIND_PACKAGE(Torch)
ENDIF()
IF (${PLATFORM} MATCHES "nvcc")
  IF(NOT CUDA_FOUND)
    FIND_PACKAGE(CUDA 5.5 REQUIRED)
  ENDIF()
  IF(NOT MAGMA_FOUND)
    FIND_PACKAGE(MAGMA)
  ENDIF()
ENDIF() # PLATFORM MATCHES

EXECUTE_PROCESS(COMMAND ${HIP_PATH}/bin/hipconfig --cpp_config OUTPUT_VARIABLE HIP_CXX_FLAGS)

# temoorary check with hipcc on AMD stack
IF (${PLATFORM} MATCHES "hcc")
  SET(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
ENDIF()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9.3")
    if(CUDA_VERSION VERSION_LESS "8.0")
      MESSAGE(STATUS "Found gcc >=5 and CUDA <= 7.5, adding workaround C++ flags")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__")
    endif(CUDA_VERSION VERSION_LESS "8.0")
  endif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9.3")
endif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.7" OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL "4.7" )
    # add c++11 flag
    set_property(SOURCE THCCachingAllocator.cpp APPEND PROPERTY COMPILE_FLAGS "-std=c++11") 
  else()
    # add c++0x flag
    set_property(SOURCE THCCachingAllocator.cpp APPEND PROPERTY COMPILE_FLAGS "-std=c++0x")
  endif()
else()
  SET(CMAKE_CXX_STANDARD 11)
endif()


INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
INCLUDE_DIRECTORIES($ENV{HIP_THRUST_PATH})
INCLUDE_DIRECTORIES(${HIPBLAS_PATH}/include)
INCLUDE_DIRECTORIES("${CUDA_SDK_ROOT_DIR}/common/inc")
LINK_DIRECTORIES("${HIP_PATH}/lib")

IF(MAGMA_FOUND)
  INCLUDE_DIRECTORIES(${MAGMA_INCLUDE_DIR})
  SET(CMAKE_REQUIRED_INCLUDES "${MAGMA_INCLUDE_DIR};${CUDA_INCLUDE_DIRS}")
  INCLUDE(CheckPrototypeDefinition)
  check_prototype_definition(magma_get_sgeqrf_nb
   "magma_int_t magma_get_sgeqrf_nb( magma_int_t m, magma_int_t n );"
   "0"
   "magma.h"
    MAGMA_V2)
  IF (MAGMA_V2)
    add_definitions(-DMAGMA_V2)
  ENDIF (MAGMA_V2)

  SET(USE_MAGMA 1)
  MESSAGE(STATUS "Compiling with MAGMA support")
  MESSAGE(STATUS "MAGMA INCLUDE DIRECTORIES: ${MAGMA_INCLUDE_DIR}")
  MESSAGE(STATUS "MAGMA LIBRARIES: ${MAGMA_LIBRARIES}")
  MESSAGE(STATUS "MAGMA V2 check: ${MAGMA_V2}")
ELSE(MAGMA_FOUND)
  MESSAGE(STATUS "MAGMA not found. Compiling without MAGMA support")
ENDIF(MAGMA_FOUND)

IF ($ENV{TH_BINARY_BUILD})
  MESSAGE(STATUS "TH_BINARY_BUILD detected. Statically linking libstdc++")
  SET(CMAKE_CXX_FLAGS "-static-libstdc++ ${CMAKE_CXX_FLAGS}")
ENDIF()

IF(${PLATFORM} MATCHES "nvcc")
  # Detect CUDA architecture and get best NVCC flags
  IF(NOT COMMAND CUDA_SELECT_NVCC_ARCH_FLAGS)
    INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/select_compute_arch.cmake)
  ENDIF()
  LIST(APPEND CUDA_NVCC_FLAGS $ENV{TORCH_NVCC_FLAGS})
  CUDA_SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_EXTRA $ENV{TORCH_CUDA_ARCH_LIST})
  LIST(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})
  LIST(APPEND CUDA_NVCC_FLAGS "-lineinfo")
ENDIF() #PLATFORM MATCHES

IF(NOT THC_INSTALL_BIN_SUBDIR
    OR NOT THC_INSTALL_LIB_SUBDIR
    OR NOT THC_INSTALL_INCLUDE_SUBDIR
    OR NOT THC_INSTALL_CMAKE_SUBDIR)

  INCLUDE_DIRECTORIES(${TH_INCLUDE_PATH} ${TH_INCLUDE_PATH}/TH)
  LINK_DIRECTORIES(${TH_LIB_PATH})

  SET(THC_INSTALL_BIN_SUBDIR "bin" CACHE PATH "THC install binary subdirectory")
  SET(THC_INSTALL_LIB_SUBDIR "lib" CACHE PATH "THC install library subdirectory")
  SET(THC_INSTALL_INCLUDE_SUBDIR "include" CACHE PATH "THC install include subdirectory")
  SET(THC_INSTALL_CMAKE_SUBDIR "share/cmake/THC" CACHE PATH "THC install cmake subdirectory")
ELSE()
  SET(THC_INSTALL_BIN_SUBDIR ${Torch_INSTALL_BIN_SUBDIR})
  SET(THC_INSTALL_LIB_SUBDIR ${Torch_INSTALL_LIB_SUBDIR})
  SET(THC_INSTALL_INCLUDE_SUBDIR ${Torch_INSTALL_INCLUDE_SUBDIR})
  SET(THC_INSTALL_CMAKE_SUBDIR ${Torch_INSTALL_CMAKE_SUBDIR})
ENDIF()

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_BINARY_DIR}")
CONFIGURE_FILE(THCGeneral.h.in "${CMAKE_CURRENT_BINARY_DIR}/THCGeneral.h")

IF(MSVC)
  LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler /wd4819")
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_CXX_FLAGS}")
ENDIF()

SET(src
    THCCachingAllocator.cpp
    THCGeneral.cc
    THCStorageCopy.cc
    THCStream.cc
    THCTensor.cc
    THCTensorCopy.cc
    THCThreadLocal.cc
    )

SET(src-cuda
  THCReduceApplyUtils.cu
  THCBlas.cu
  THCStorage.cu
  THCStorageCopy.cu
  THCTensor.cu
  THCTensorCopy.cu
  THCTensorMath.cu
  THCThrustAlternate.cu
  THCTensorMath2.cu
  THCTensorMathBlas.cu
  THCTensorMathMagma.cu
  THCTensorMathPairwise.cu
  THCTensorMathReduce.cu
  THCTensorMathScan.cu
  THCTensorIndex.cu
  THCTensorConv.cu
  THCTensorRandom.cu
  THCTensorScatterGather.cu
  THCTensorTopK.cu
  THCTensorSort.cu
  THCTensorTypeUtils.cu
  )

# loop over all types
foreach(THC_TYPE Byte Char Short Int Long Half Float Double)
   # loop over files which need to be split between types (because of long compile times)
   foreach(THC_FILE TensorSort TensorMathCompareT TensorMathPointwise TensorMathCompare TensorMathReduce TensorMasked)
      if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}${THC_TYPE}.cu")
         FILE(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}${THC_TYPE}.cu"
              "#include \"../THC${THC_FILE}.cuh\"\n#include \"../generic/THC${THC_FILE}.cu\"\n#include \"../THCGenerate${THC_TYPE}Type.h\"\n")
      endif()
      LIST(APPEND src-cuda "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}${THC_TYPE}.cu")
   endforeach()
endforeach()

IF(${PLATFORM} MATCHES "nvcc")
  MESSAGE(STATUS "got cuda version " ${CUDA_VERSION})

IF(CUDA_HAS_FP16 OR NOT ${CUDA_VERSION} LESS 7.5)
  MESSAGE(STATUS "Found CUDA with FP16 support, compiling with torch.CudaHalfTensor")
  LIST(APPEND src-cuda THCHalf.cu)
# TODO: HIP EQUIVALENT: MANEESH
# the below statement is temporarily commented. Got to have an equivalent FP16 support flag
#  LIST(APPEND CUDA_NVCC_FLAGS "-DCUDA_HAS_FP16=1")
#  SET(CMAKE_C_FLAGS "-DCUDA_HAS_FP16=1 ${CMAKE_C_FLAGS}")
ELSE(CUDA_HAS_FP16 OR NOT ${CUDA_VERSION} LESS 7.5)
  MESSAGE(STATUS "Could not find CUDA with FP16 support, compiling without torch.CudaHalfTensor")
ENDIF(CUDA_HAS_FP16 OR NOT ${CUDA_VERSION} LESS 7.5)

MESSAGE(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
ENDIF() #PLATFORM

set (CMAKE_SHARED_LINKER_FLAGS "-Wl,-Bsymbolic")
SET(HIP_HCC_FLAGS "${HIP_HCC_FLAGS} -DGENERIC_GRID_LAUNCH=1")
HIP_ADD_LIBRARY(THC SHARED ${src} ${src-cuda})

IF(${PLATFORM} STREQUAL "nvcc")
  CUDA_ADD_CUBLAS_TO_TARGET(THC)
  TARGET_LINK_LIBRARIES(THC TH ${CUDA_curand_LIBRARY})
ENDIF()

IF(${PLATFORM} STREQUAL "hcc")
  FIND_LIBRARY(HIPBLAS_LIBRARY hipblas_hcc HINTS ${HIPBLAS_PATH}/lib)
  TARGET_LINK_LIBRARIES(THC TH hip_hcc ${HIPBLAS_LIBRARY})
ENDIF()

IF(${PLATFORM} STREQUAL "nvcc")
IF(USE_MAGMA)
  FIND_LIBRARY(HIPBLAS_LIBRARY hipblas_nvcc HINTS ${HIPBLAS_PATH}/lib)
  FIND_LIBRARY(CUDART_LIBRARY cudart HINTS ${CUDA_SDK_ROOT_DIR}/lib64)
  TARGET_LINK_LIBRARIES(THC ${MAGMA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${HIPBLAS_LIBRARY} ${CUDART_LIBRARY})
ELSE()
  FIND_LIBRARY(HIPBLAS_LIBRARY hipblas_nvcc HINTS ${HIPBLAS_PATH}/lib)
  FIND_LIBRARY(CUDART_LIBRARY cudart HINTS /usr/local/cuda/lib64)
  TARGET_LINK_LIBRARIES(THC ${CUDA_cusparse_LIBRARY} ${HIPBLAS_LIBRARY} ${CUDART_LIBRARY})
ENDIF(USE_MAGMA)
ENDIF()

INSTALL(TARGETS THC
          RUNTIME DESTINATION "${THC_INSTALL_BIN_SUBDIR}"
          LIBRARY DESTINATION "${THC_INSTALL_LIB_SUBDIR}"
          ARCHIVE DESTINATION "${THC_INSTALL_LIB_SUBDIR}")

INSTALL(FILES
          THC.h
          ${CMAKE_CURRENT_BINARY_DIR}/THCGeneral.h
          THCBlas.h
          THCStorage.h
          THCStorageCopy.h
          THCStream.h
          THCThreadLocal.h
          THCTensor.h
          THCTensorCopy.h
          THCTensorRandom.h
          THCTensorMath.h
          THCTensorConv.h
          THCTensorTopK.h
          THCApply.cuh
          THCReduce.cuh
          THCReduceAll.cuh
          THCReduceApplyUtils.cuh
          THCAsmUtils.cuh
          THCAtomics.cuh
          THCScanUtils.cuh
          THCSortUtils.cuh
          THCAllocator.h
          THCCachingAllocator.h
          THCDeviceUtils.cuh
          THCDeviceTensor.cuh
          THCDeviceTensor-inl.cuh
          THCDeviceTensorUtils.cuh
          THCDeviceTensorUtils-inl.cuh
          THCGenerateAllTypes.h
          THCGenerateByteType.h
          THCGenerateCharType.h
          THCGenerateShortType.h
          THCGenerateIntType.h
          THCGenerateLongType.h
          THCGenerateHalfType.h
          THCGenerateFloatType.h
          THCGenerateFloatTypes.h
          THCGenerateDoubleType.h
          THCHalf.h
          THCNumerics.cuh
          THCTensorSort.cuh
          THCTensorInfo.cuh
          THCTensorTypeUtils.cuh
          DESTINATION "${THC_INSTALL_INCLUDE_SUBDIR}/THC")

INSTALL(FILES
          generic/THCStorage.c
          generic/THCStorage.cu
          generic/THCStorage.h
          generic/THCTensor.c
          generic/THCTensor.cu
          generic/THCTensor.h
          generic/THCStorageCopy.c
          generic/THCStorageCopy.cu
          generic/THCStorageCopy.h
          generic/THCTensorCopy.c
          generic/THCTensorCopy.cu
          generic/THCTensorCopy.h
          generic/THCTensorMasked.h
          generic/THCTensorMasked.cu
          generic/THCTensorMath.h
          generic/THCTensorMath.cu
          generic/THCThrustAlternate.cu
          generic/THCTensorMathBlas.cu
          generic/THCTensorMathBlas.h
          generic/THCTensorMathCompare.h
          generic/THCTensorMathCompare.cu
          generic/THCTensorMathCompareT.h
          generic/THCTensorMathCompareT.cu
          generic/THCTensorMathPairwise.h
          generic/THCTensorMathPairwise.cu
          generic/THCTensorMathPointwise.h
          generic/THCTensorMathPointwise.cu
          generic/THCTensorMathReduce.h
          generic/THCTensorMathReduce.cu
          generic/THCTensorScatterGather.h
          generic/THCTensorScatterGather.cu
          generic/THCTensorIndex.h
          generic/THCTensorIndex.cu
          generic/THCTensorSort.h
          generic/THCTensorSort.cu
          generic/THCDeviceTensorUtils.cu
          DESTINATION "${THC_INSTALL_INCLUDE_SUBDIR}/THC/generic")
