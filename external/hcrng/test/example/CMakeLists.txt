# Object libraries require CMAKE 2.8.8 version 
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8) 
MESSAGE(STATUS "CMAKE VERSION ${CMAKE_VERSION}")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# Find HCC compiler
FIND_PACKAGE(HC++ 1.0 REQUIRED)

#Setting a variable for source files
SET (TESTSRCS
    Host_only.cpp  Multistream_uniform.cpp Multistream_normal.cpp  Randomarray_uniform.cpp Randomarray_normal.cpp
    )

  # Choice to take compilation flags from source or package
  if(EXISTS ${MCWHCCBUILD})
    execute_process(COMMAND ${HCC_CONFIG} --build --cxxflags
                            OUTPUT_VARIABLE HCC_CXXFLAGS)
    execute_process(COMMAND ${HCC_CONFIG} --build --ldflags 
                            OUTPUT_VARIABLE HCC_LDFLAGS)
  else(EXISTS ${MCWHCCBUILD})
    execute_process(COMMAND ${HCC_CONFIG} --install --cxxflags
                            OUTPUT_VARIABLE HCC_CXXFLAGS)
    execute_process(COMMAND ${HCC_CONFIG} --install --ldflags 
                            OUTPUT_VARIABLE HCC_LDFLAGS)
  endif(EXISTS ${MCWHCCBUILD})

  SET(HCRNG_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/include/")
  SET(HCRNG_LIBRARY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../build/lib/src")

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin/")
  string(STRIP "${HCC_CXXFLAGS}" HCC_CXXFLAGS)
  set (HCC_CXXFLAGS "${HCC_CXXFLAGS} -I${HCRNG_INCLUDE_PATH}")
  string(STRIP "${HCC_LDFLAGS}" HCC_LDFLAGS)
  set (HCC_LDFLAGS "${HCC_LDFLAGS} -L${HCRNG_LIBRARY_PATH}")
  FOREACH(test_file ${TESTSRCS})
    SET_PROPERTY(SOURCE ${test_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${HCC_CXXFLAGS}")
    STRING( REPLACE ".cpp" "" testname ${test_file} )
    ADD_EXECUTABLE(${testname} ${test_file} )
    SET_PROPERTY(TARGET ${testname} APPEND_STRING PROPERTY LINK_FLAGS " ${HCC_LDFLAGS}")
    TARGET_LINK_LIBRARIES(${testname} hcrng hc_am)
  ENDFOREACH()
