# Object libraries require CMAKE 2.8.8 version 
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8) 
MESSAGE(STATUS "CMAKE VERSION ${CMAKE_VERSION}")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# Find HCC compiler
FIND_PACKAGE(HC++ 1.0 REQUIRED)

#Setting a variable for source files
SET (TESTSRCS
    cgemm_test.cpp  dcopy_test.cpp  dscal_test.cpp  saxpy_test.cpp  sdot_test.cpp   sgemv_test.cpp  sscal_test.cpp  
    dasum_test.cpp  ddot_test.cpp   sasum_test.cpp  scopy_test.cpp  sgemm_test.cpp  sger_test.cpp  sgemm_cn_test.cpp
    sgemm_timer_test.cpp
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
  SET(HCBLAS_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/include/")
  SET(TEST_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../include/")
  SET(HCBLAS_LIBRARY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../build/lib/src")
  string(STRIP "${HCC_CXXFLAGS}" HCC_CXXFLAGS)
  set (HCC_CXXFLAGS "-I${TEST_INCLUDE_PATH} ${HCC_CXXFLAGS} -I${HCBLAS_INCLUDE_PATH}")
  string(STRIP "${HCC_LDFLAGS}" HCC_LDFLAGS)
  set (HCC_LDFLAGS "${HCC_LDFLAGS} -L${HCBLAS_LIBRARY_PATH}")
  SET (LINK "-lblas -lhcblas -lhc_am")
  FOREACH(test_file ${TESTSRCS})
    SET_PROPERTY(SOURCE ${test_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${HCC_CXXFLAGS}")
    STRING( REPLACE "_test.cpp" "" testname ${test_file} )
    ADD_EXECUTABLE( bin/${testname} ${test_file} )
    SET_PROPERTY(TARGET bin/${testname} APPEND_STRING PROPERTY LINK_FLAGS " ${HCC_LDFLAGS} ${LINK}")
  ENDFOREACH()
