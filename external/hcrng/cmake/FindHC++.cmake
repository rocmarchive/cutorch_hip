# http://www.cmake.org/Wiki/CMake:How_To_Find_Libraries

# - Try to find HC++ Compiler
# Once done this will define
#  HC++_FOUND - System has HC++
#  HC++_BIN_DIR - The HC++ binaries directories
#  HCC_CXXFLAGS - The HC++ compilation flags
#  HCC_LDFLAGS - The HC++ linker flags

# The following are available when in installation mode
#  HC++_INCLUDE_DIRS - The HC++ include directories
#  HC++_LIBRARIES - The libraries needed to use HC++

if( MSVC OR APPLE)
  message(FATAL_ERROR "Unsupported platform.")
endif()

set(MCWHCCBUILD $ENV{MCWHCCBUILD})

if(EXISTS ${MCWHCCBUILD})
  find_path(HC++_BIN_DIR clang++
           HINTS ${MCWHCCBUILD}/compiler/bin)
  find_path(HC++_CONFIGURE_DIR hcc-config
           HINTS ${MCWHCCBUILD}/bin)
  include(FindPackageHandleStandardArgs)
  # handle the QUIETLY and REQUIRED arguments and set HC++_FOUND to TRUE
  # if all listed variables are TRUE
  find_package_handle_standard_args(HC++  DEFAULT_MSG
                                    HC++_BIN_DIR HC++_CONFIGURE_DIR)
  mark_as_advanced(HC++_BIN_DIR HC++_CONFIGURE_DIR)
  if (HC++_FOUND)
    message(STATUS "HC++ Compiler found in ${HC++_BIN_DIR}/..")
    set(CMAKE_C_COMPILER ${HC++_BIN_DIR}/clang)
    set(CMAKE_CXX_COMPILER ${HC++_BIN_DIR}/clang++)
  elseif()
    message(FATAL_ERROR "HC++ Compiler not found.")
  endif()

  # Build mode
  set (CLANG_AMP "${HC++_BIN_DIR}/clang++")
  set (HCC_CONFIG "${HC++_CONFIGURE_DIR}/hcc-config")
  execute_process(COMMAND ${HCC_CONFIG} --build --cxxflags
                  OUTPUT_VARIABLE HCC_CXXFLAGS)
  string(STRIP "${HCC_CXXFLAGS}" HCC_CXXFLAGS)
  set (HCC_CXXFLAGS "${HCC_CXXFLAGS}")
  execute_process(COMMAND ${HCC_CONFIG} --build --ldflags --shared
                  OUTPUT_VARIABLE HCC_LDFLAGS)
  string(STRIP "${HCC_LDFLAGS}" HCC_LDFLAGS)
  set (HCC_CXXFLAGS "${HCC_CXXFLAGS} -Wall -Wno-deprecated-register -Wno-deprecated-declarations")
  set (HCC_LDFLAGS "${HCC_LDFLAGS}")
# Package built from sources
# Compiler and configure file are two key factors to advance
elseif(EXISTS /opt/rocm/hcc/bin/clang++)
  find_path(HC++_BIN_DIR clang++
           HINTS /opt/rocm/hcc/bin)
  find_path(HC++_CONFIGURE_DIR hcc-config
           HINTS /opt/rocm/hcc/bin)
  include(FindPackageHandleStandardArgs)
  # handle the QUIETLY and REQUIRED arguments and set HC++_FOUND to TRUE
  # if all listed variables are TRUE
  find_package_handle_standard_args(HC++  DEFAULT_MSG
                                    HC++_BIN_DIR HC++_CONFIGURE_DIR)
  mark_as_advanced(HC++_BIN_DIR HC++_CONFIGURE_DIR)
  if (HC++_FOUND)
    message(STATUS "HC++ Compiler found in ${HC++_BIN_DIR}/..")
    set(CMAKE_C_COMPILER ${HC++_BIN_DIR}/clang)
    set(CMAKE_CXX_COMPILER ${HC++_BIN_DIR}/clang++)
  elseif()
    message(FATAL_ERROR "HC++ Compiler not found.")
  endif()

  # Build mode
  set (CLANG_AMP "${HC++_BIN_DIR}/clang++")
  set (HCC_CONFIG "${HC++_CONFIGURE_DIR}/hcc-config")
  execute_process(COMMAND ${HCC_CONFIG} --cxxflags
                  OUTPUT_VARIABLE HCC_CXXFLAGS)
  string(STRIP "${HCC_CXXFLAGS}" HCC_CXXFLAGS)
  set (HCC_CXXFLAGS "${HCC_CXXFLAGS}")
  execute_process(COMMAND ${HCC_CONFIG} --ldflags --shared
                  OUTPUT_VARIABLE HCC_LDFLAGS)
  string(STRIP "${HCC_LDFLAGS}" HCC_LDFLAGS)
  set (HCC_CXXFLAGS "${HCC_CXXFLAGS} -Wall -Wno-deprecated-register -Wno-deprecated-declarations")
  set (HCC_LDFLAGS "${HCC_LDFLAGS}")

endif()


