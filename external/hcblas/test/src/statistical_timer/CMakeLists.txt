# Object libraries require CMAKE 2.8.8 version 
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8) 
MESSAGE(STATUS "CMAKE VERSION ${CMAKE_VERSION}")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# Find HCC compiler
FIND_PACKAGE(HC++ 1.0 REQUIRED)

#Setting a variable for source files
SET (STSRCS
     sgemm_st_timer_test.cpp statisticalTimer.cpp
    )

  # Choice to take compilation flags from source or package
  option(PROFILE "HCBLAS profiling" OFF)
  IF(PROFILE)
   ADD_DEFINITIONS(-DPROFILE)
  ENDIF(PROFILE)

  execute_process(COMMAND ${HCC_CONFIG} --install --cxxflags
                            OUTPUT_VARIABLE HCC_CXXFLAGS)
  execute_process(COMMAND ${HCC_CONFIG} --install --ldflags 
                            OUTPUT_VARIABLE HCC_LDFLAGS)
  SET(HCBLAS_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/include/")
  SET(HCBLAS_LIBRARY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../build/lib/src")
  string(STRIP "${HCC_CXXFLAGS}" HCC_CXXFLAGS)
  set (HCC_CXXFLAGS "${HCC_CXXFLAGS} -I${HCBLAS_INCLUDE_PATH} -I${CMAKE_CURRENT_SOURCE_DIR}")
  string(STRIP "${HCC_LDFLAGS}" HCC_LDFLAGS)
  set (HCC_LDFLAGS "${HCC_LDFLAGS} -L${HCBLAS_LIBRARY_PATH}")
  SET (LINK "-lblas -lhcblas -lhc_am")
  SET_PROPERTY(SOURCE ${STSRCS} APPEND_STRING PROPERTY COMPILE_FLAGS " ${HCC_CXXFLAGS}")
  STRING( REPLACE "_test.cpp" "" testname sgemm_st_timer_test.cpp )
  ADD_EXECUTABLE( ../bin/${testname} ${STSRCS} )
  SET_PROPERTY(TARGET ../bin/${testname} APPEND_STRING PROPERTY LINK_FLAGS " ${HCC_LDFLAGS} ${LINK}")
