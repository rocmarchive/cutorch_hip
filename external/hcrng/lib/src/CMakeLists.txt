
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.8) 
MESSAGE(STATUS "CMAKE VERSION ${CMAKE_VERSION}")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Find HCC compiler
FIND_PACKAGE(HC++ 1.0 REQUIRED)

ADD_SUBDIRECTORY(generators)

#Setting a Project name
SET(PROJECT_NAME "libhcrng")
PROJECT (${PROJECT_NAME})
MESSAGE(STATUS "PROJECT NAME    : ${PROJECT_NAME}")

#Setting a variable for source files
SET (SRCS ${RNGSRC})

  # Choice to take compilation flags from source or package
  if(EXISTS ${MCWHCCBUILD})
    execute_process(COMMAND ${HCC_CONFIG} --build --cxxflags
                            OUTPUT_VARIABLE HCC_CXXFLAGS)
    execute_process(COMMAND ${HCC_CONFIG} --build --ldflags --shared
                            OUTPUT_VARIABLE HCC_LDFLAGS)
  else(EXISTS ${MCWHCCBUILD})
    execute_process(COMMAND ${HCC_CONFIG} --install --cxxflags
                            OUTPUT_VARIABLE HCC_CXXFLAGS)
    execute_process(COMMAND ${HCC_CONFIG} --install --ldflags --shared
                            OUTPUT_VARIABLE HCC_LDFLAGS)
  endif(EXISTS ${MCWHCCBUILD})
  
  string(STRIP "${HCC_CXXFLAGS}" HCC_CXXFLAGS)
  string(STRIP "${HCC_LDFLAGS}" HCC_LDFLAGS)
  set (HCC_CXXFLAGS "${HCC_CXXFLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/../include")
  set (HCC_LDFLAGS "${HCC_LDFLAGS}")

  FOREACH(src_file ${SRCS})
          SET_PROPERTY(SOURCE ${src_file} APPEND_STRING PROPERTY COMPILE_FLAGS " ${HCC_CXXFLAGS} ")
  ENDFOREACH()

  ADD_EXECUTABLE("${PROJECT_NAME}.so" ${SRCS})
  SET_PROPERTY(TARGET "${PROJECT_NAME}.so" APPEND_STRING PROPERTY LINK_FLAGS " ${HCC_LDFLAGS} ")
  TARGET_LINK_LIBRARIES("${PROJECT_NAME}.so" hc_am)
  INSTALL(TARGETS "${PROJECT_NAME}.so" 
    RUNTIME DESTINATION lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )

  INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../include/" DESTINATION include)
