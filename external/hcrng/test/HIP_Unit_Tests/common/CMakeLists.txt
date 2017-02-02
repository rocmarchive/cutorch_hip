#Object libraries require CMAKE 2.8.8 version 
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8) 
MESSAGE(STATUS "CMAKE VERSION ${CMAKE_VERSION}")

SET (TESTSRCS
lfsrGenerate.cpp  mrg31k3pGenerate.cpp  mrg32k3aGenerate.cpp  philox432Generate.cpp
)

set (HIP_PATH $ENV{HIP_PATH})
if (NOT DEFINED HIP_PATH)
	set (HIP_PATH /opt/rocm/hip)
endif()

set (HIPRNG_ROOT $ENV{HIPRNG_ROOT})
if (NOT DEFINED HIPRNG_ROOT)
	set (HIPRNG_ROOT ../../../)
endif()
include_directories(${HIPRNG_ROOT}/lib/include)

execute_process(COMMAND ${HIP_PATH}/bin/hipconfig --platform OUTPUT_VARIABLE HIP_PLATFORM)

MESSAGE ("HIP_PATH=" ${HIP_PATH})

if (${HIP_PLATFORM} STREQUAL "hcc")
	MESSAGE ("HIP_PLATFORM=hcc")
	set (HSA_PATH /opt/hsa)

	# Add HSA library:
#	add_library(hsa-runtime64 SHARED IMPORTED)
#	set_property(TARGET hsa-runtime64 PROPERTY IMPORTED_LOCATION "${HSA_PATH}/lib/libhsa-runtime64.so")

	set (HCRNG_ROOT $ENV{HCFFT_ROOT})
	if (NOT DEFINED HCRNG_ROOT)
		set (HCRNG_ROOT ../../)
	endif()

	#These includes are used for all files.
	#Include HIP and HC since the tests need both of these:
	#Note below HSA path is surgically included only where necessary.
	include_directories(${HIP_PATH}/include ${HC_PATH}/lib/include  ${HCRNG_ROOT}/lib/include  )

	# hip_hcc.o:
#	add_library(hip_hcc OBJECT ${HIP_PATH}/src/hip_hcc.cpp)
#	target_include_directories(hip_hcc PRIVATE ${HSA_PATH}/include)

        add_library(hcrng SHARED IMPORTED)
        set_target_properties( hcrng PROPERTIES IMPORTED_LOCATION "${HCRNG_ROOT}/libhcrng.so" )

        MESSAGE ("CURRENTSRC${CMAKE_CURRENT_SOURCE_DIR}")
	SET(HCRNG_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/include/")
        SET(HCRNG_LIBRARY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../build/lib/src")

  	set (HCC_CXXFLAGS "-I${HCRNG_INCLUDE_PATH} -I${CMAKE_CURRENT_SOURCE_DIR}/../../lib/include/hcc_detail/")
 	set (HCC_LDFLAGS "-L${HCRNG_LIBRARY_PATH}")
  	set (LINK "-lhcrng")

elseif (${HIP_PLATFORM} STREQUAL "nvcc")

	MESSAGE ("HIP_PLATFORM = nvcc")
	#NVCC does not support -rdynamic option 
	set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS )
	set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS )

	find_package( CUDA REQUIRED )

else()
	MESSAGE (FATAL_ERROR "UNKNOWN HIP_PLATFORM=" ${HIP_PLATFORM})
endif()

set (HIPCC ${HIP_PATH}/bin/hipcc)
set (CMAKE_CXX_COMPILER ${HIPCC})
add_definitions(-stdlib=libc++) 

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin/")
macro (make_hip_executable exe cpp)  
  if (${HIP_PLATFORM} STREQUAL "hcc")
    add_executable (${exe} ${cpp} )
    TARGET_LINK_LIBRARIES(${exe} ${LINK} ${HCC_LDFLAGS})
  else()
    EXECUTE_PROCESS (COMMAND nvcc ${cpp} -lcublas -I${HIPBLAS_ROOT}/include -I${HIP_PATH}/include -o ${exe})
  endif() 
endmacro()

if (${HIP_PLATFORM} STREQUAL "hcc")
	FOREACH(test_file ${TESTSRCS})
	  get_filename_component (name_without_extension ${test_file} NAME_WE)
	  make_hip_executable(${name_without_extension} ${test_file})
	ENDFOREACH()	
endif()

