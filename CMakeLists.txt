CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

# HIP_PATH
IF(NOT DEFINED $ENV{HIP_PATH})
  SET(HIP_PATH /opt/rocm/hip)
ELSE()
  SET(HIP_PATH $ENV{HIP_PATH})
ENDIF()

# HIPBLAS_PATH
IF(NOT DEFINED $ENV{HIPBLAS_PATH})
  SET(HIPBLAS_PATH /opt/rocm/hcblas/)
ELSE()
  SET(HIPBLAS_PATH $ENV{HIPBLAS_PATH})
ENDIF()

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake
                      ${CMAKE_CURRENT_SOURCE_DIR}/lib/THC/cmake
                      ${CMAKE_MODULE_PATH}
                      "${HIP_PATH}/cmake")
EXECUTE_PROCESS(COMMAND ${HIP_PATH}/bin/hipconfig -P OUTPUT_VARIABLE PLATFORM)

include(HipTorchWrap)

FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(HIP 1.0 REQUIRED)
IF(${PLATFORM} STREQUAL "nvcc")
  FIND_PACKAGE(CUDA 6.5 REQUIRED)
  FIND_PACKAGE(MAGMA)
ENDIF()

EXECUTE_PROCESS(COMMAND ${HIP_PATH}/bin/hipconfig --cpp_config OUTPUT_VARIABLE HIP_CXX_FLAGS)
IF (${PLATFORM} MATCHES "hcc")
  SET(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
ENDIF()
IF (NOT WIN32)
SET(CMAKE_C_FLAGS "-std=c99 -Werror=implicit-function-declaration ${CMAKE_C_FLAGS}")
SET(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_CXX_FLAGS}")
ENDIF (NOT WIN32)
IF(CUDA_HAS_FP16 OR NOT ${CUDA_VERSION} LESS 7.5)
 # TODO: NEEL FP!6 support
 # SET(CMAKE_C_FLAGS "-DCUDA_HAS_FP16=1 ${CMAKE_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS  "-std=c++11 ${CMAKE_CXX_FLAGS} ${HIP_CXX_FLAGS}")
ENDIF()

# TODO: temporary blunt implements.
include_directories(/home/alexv/Programming/HCC/cmake-build-release/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/bolt/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/boost)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/hcblas/lib/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/hcrng/lib/include)
#include_directories(${HIP_ROOT_DIR}/include)
include_directories(/home/alexv/Programming/HIP_private_github/include)

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

ADD_SUBDIRECTORY(lib)

INCLUDE_DIRECTORIES(BEFORE "${CMAKE_CURRENT_BINARY_DIR}/lib/THC")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/lib/THC")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/torch")
LINK_DIRECTORIES("${HIP_PATH}/lib")

SET(src Storage.cc init.cc Tensor.cc TensorMath.cpp TensorOperator.cc torch/utils.c)
SET(luasrc init.lua Tensor.lua FFI.lua test/test.lua)

ADD_HIP_TORCH_WRAP(HipTensormathwrap TensorMath.lua)
ADD_TORCH_PACKAGE(cutorch "${src}" "${luasrc}")

IF(${PLATFORM} MATCHES "hcc")
TARGET_LINK_LIBRARIES(cutorch luaT THC hip_hcc)
ELSEIF(${PLATFORM} MATCHES "nvcc")
TARGET_LINK_LIBRARIES(cutorch luaT THC)
ENDIF()

IF(LUALIB)
  TARGET_LINK_LIBRARIES(cutorch ${LUALIB})
ENDIF()

  SET(EXE cutorch.test)
  ADD_EXECUTABLE(${EXE} test/test.cpp)
  TARGET_LINK_LIBRARIES(${EXE} qlua qtlua luajit m)
  SET_PROPERTY(TARGET ${EXE} APPEND_STRING PROPERTY LINK_FLAGS " -I/usr/local/include -I./test")
