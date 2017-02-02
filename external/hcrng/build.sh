#!/bin/bash -e 
# This script is invoked to install the hcRNG library and test sources

# CHECK FOR COMPILER PATH

if [ ! -z $MCWHCCBUILD ]; 
then
  if [ -x "$MCWHCCBUILD/compiler/bin/clang++" ]; 
  then
    cmake_c_compiler="$MCWHCCBUILD/compiler/bin/clang"
    cmake_cxx_compiler="$MCWHCCBUILD/compiler/bin/clang++"
  fi

elif [ -x "/opt/rocm/hcc/bin/clang++" ]; 
then
  cmake_c_compiler="/opt/rocm/hcc/bin/clang"
  cmake_cxx_compiler="/opt/rocm/hcc/bin/clang++"
else
  echo "Clang compiler not found"
  exit 1
fi

# CURRENT_WORK_DIRECTORY
current_work_dir=$PWD

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$current_work_dir/build/lib/src

red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`

# Help menu
print_help() {
cat <<-HELP
=============================================================================================================================
This script is invoked to build hcRNG library and test sources. Please provide the following arguments:

  ${green}--test${reset}    Test to enable the library testing. 

=============================================================================================================================
Usage: ./build.sh --test=on (or off) 
=============================================================================================================================
Example: 
(2) ${green}./install.sh --test=on ${reset} (needs sudo access)
       <library gets installed in /opt/ROCm/, testing = on>

=============================================================================================================================
HELP
exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --test=*)
      testing="${1#*=}"
      ;;
    --help) print_help;;
    *)
      printf "************************************************************\n"
      printf "* Error: Invalid arguments, run --help for valid arguments.*\n"
      printf "************************************************************\n"
      exit 1
  esac
  shift
done

set +e
# MAKE BUILD DIR
mkdir $current_work_dir/build
mkdir $current_work_dir/build/test
mkdir -p $current_work_dir/build/test/HIP_Unit_Tests/common/bin/
mkdir -p $current_work_dir/build/test/HIP_Unit_Tests/normal/bin/
mkdir -p $current_work_dir/build/test/HIP_Unit_Tests/uniform/bin/
set -e

# SET BUILD DIR
build_dir=$current_work_dir/build

# change to library build
cd $build_dir

# Cmake and make libhcRNG: Install hcRNG
cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC $current_work_dir
make package
make

if [ "$testing" = "on" ]; then
  # Build Tests
    cd $build_dir/test/ && cmake -DCMAKE_C_COMPILER=$cmake_c_compiler -DCMAKE_CXX_COMPILER=$cmake_cxx_compiler -DCMAKE_CXX_FLAGS=-fPIC $current_work_dir/test/
    make

# Move to test folder
    cd $current_work_dir/test/

#Invoke test script
    ./test.sh

chmod +x $current_work_dir/test/HIP_Unit_Tests/test.sh
cd $current_work_dir/test/HIP_Unit_Tests/
  
#Invoke test script
    ./test.sh

else
  echo "${green}HCRNG Installation Completed!${reset}"
fi

# TODO: ADD More options to perform benchmark
