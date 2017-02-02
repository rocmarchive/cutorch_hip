
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8) 
MESSAGE(STATUS "CMAKE VERSION ${CMAKE_VERSION}")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Find HCC compiler
FIND_PACKAGE(HC++ 1.0 REQUIRED)

ADD_SUBDIRECTORY(blas)

#Setting a Project name
SET(PROJECT_NAME "hcblas")
SET(PROJECT_NAME_EXT "hipblas")
SET(HIP_PATH "/opt/rocm/hip")
PROJECT (${PROJECT_NAME})
MESSAGE(STATUS "PROJECT NAME    : ${PROJECT_NAME}")

#Setting a variable for source files
SET (HCBLASSRCS ${BLASSRC} hcblas.cpp)
SET (HIPBLASSRCS ${HCBLASSRCS} hipblas.cpp)

  execute_process(COMMAND ${HCC_CONFIG} --install --cxxflags
                            OUTPUT_VARIABLE HCC_CXXFLAGS)
  execute_process(COMMAND ${HCC_CONFIG} --install --ldflags --shared
                            OUTPUT_VARIABLE HCC_LDFLAGS)
 
  string(STRIP "${HCC_CXXFLAGS}" HCC_CXXFLAGS)
  string(STRIP "${HCC_LDFLAGS}" HCC_LDFLAGS)
  set (HCC_CXXFLAGS "${HCC_CXXFLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/../include -I${HIP_PATH}/include -D__HIP_PLATFORM_HCC__=")
  set (HCC_LDFLAGS "${HCC_LDFLAGS} -L${HIP_PATH}/lib -lhip_hcc")
  
  # Setting header dependencies for hcblas library generation
  FOREACH(src_file ${HCBLASSRCS})
	  SET_PROPERTY(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${HCC_CXXFLAGS} ")
  ENDFOREACH()

  # Setting header dependencies for hipblas library generation
  FOREACH(src_file ${HIPBLASSRCS})
	  SET_PROPERTY(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${HCC_CXXFLAGS} ")
  ENDFOREACH()

  #Generating hcblas shared object
  ADD_LIBRARY("${PROJECT_NAME}" SHARED  ${HCBLASSRCS})
  SET_PROPERTY(TARGET "${PROJECT_NAME}" APPEND_STRING PROPERTY LINK_FLAGS " ${HCC_LDFLAGS} ")
  TARGET_LINK_LIBRARIES("${PROJECT_NAME}" hc_am)

  #Generating hipblas shared object
  ADD_LIBRARY("${PROJECT_NAME_EXT}" SHARED ${HIPBLASSRCS})
  SET_PROPERTY(TARGET "${PROJECT_NAME_EXT}" APPEND_STRING PROPERTY LINK_FLAGS " ${HCC_LDFLAGS} ")
  TARGET_LINK_LIBRARIES("${PROJECT_NAME_EXT}" hc_am)

  INSTALL(TARGETS "${PROJECT_NAME}" 
    RUNTIME DESTINATION lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )

  INSTALL(TARGETS "${PROJECT_NAME_EXT}" 
    RUNTIME DESTINATION lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )
  file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h")
  file(GLOB hcc_detail "${CMAKE_CURRENT_SOURCE_DIR}/../include/hcc_detail/*.h")
  file(GLOB nvcc_detail "${CMAKE_CURRENT_SOURCE_DIR}/../include/nvcc_detail/*.h")
  INSTALL(FILES ${headers} DESTINATION include)
  INSTALL(FILES ${hcc_detail} DESTINATION include/hcc_detail)
  INSTALL(FILES ${nvcc_detail} DESTINATION include/nvcc_detail)
