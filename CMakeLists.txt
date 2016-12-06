CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake
                      ${CMAKE_CURRENT_SOURCE_DIR}/lib/THC/cmake
                      ${CMAKE_MODULE_PATH}
                      /opt/rocm/hip/cmake) # TODO: this is a temporary kludge.

FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(CUDA 6.5 REQUIRED)
FIND_PACKAGE(HIP 1.0 REQUIRED)
FIND_PACKAGE(MAGMA)


#EXECUTE_PROCESS(COMMAND $ENV{HIP_PATH}/bin/hipconfig --cpp_config OUTPUT_VARIABLE HIP_C_FLAGS)
IF (NOT WIN32)
    SET(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
    SET(CMAKE_C_FLAGS "-std=c99 -Werror=implicit-function-declaration ${CMAKE_C_FLAGS}")
#    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${HIP_C_FLAGS}")
ENDIF (NOT WIN32)
IF(CUDA_HAS_FP16 OR NOT ${CUDA_VERSION} LESS 7.5)
 # TODO: NEEL FP!6 support
 # SET(CMAKE_C_FLAGS "-DCUDA_HAS_FP16=1 ${CMAKE_C_FLAGS}")
  SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${HIP_C_FLAGS}")
ENDIF()

# TODO: temporary blunt implements.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/bolt/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/boost)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/hcblas/lib/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/hcrng/lib/include)
include_directories(${HIP_ROOT_DIR}/include)

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

ADD_SUBDIRECTORY(lib)

INCLUDE_DIRECTORIES(BEFORE "${CMAKE_CURRENT_BINARY_DIR}/lib/THC")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/lib/THC")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/torch")

SET(src Storage.cpp init.cpp Tensor.cpp TensorMath.cpp TensorOperator.cpp torch/utils.c)
SET(luasrc init.lua Tensor.lua FFI.lua test/test.lua)

ADD_TORCH_WRAP(cudatensormathwrap TensorMath.lua)
ADD_TORCH_PACKAGE(cutorch "${src}" "${luasrc}")
TARGET_LINK_LIBRARIES(cutorch luaT THC)

IF(LUALIB)
  TARGET_LINK_LIBRARIES(cutorch ${LUALIB})
ENDIF()
