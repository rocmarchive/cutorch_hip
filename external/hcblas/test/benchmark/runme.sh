#!/bin/bash -e
#This script is invoked to profile major five kernels of the library

#CURRENT_WORK_DIRECTORY
CURRENTDIR=$PWD

#Export HCBLAS_PATH to be used by python scripts
export HCBLAS_PATH=$CURRENTDIR/../../

cd $CURRENTDIR/../../build/test/ && cmake -DCMAKE_CXX_FLAGS=-fPIC $HCBLAS_PATH/test/ -DPROFILE=ON 
set +e
mkdir $CURRENTDIR/../../build/test/src/bin/
mkdir $CURRENTDIR/../../build/test/unit/gtest/bin/
set -e
make
#Move to test benchmark 
cd $CURRENTDIR

#Profile SGEMM
./runmesgemm.sh

#Profile SAXPY
./runmesaxpy.sh

#Profile SGER
./runmesger.sh

#Profile CGEMM
./runmecgemm.sh

#Profile SGEMV
./runmesgemv.sh

red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`

echo "${green}HCBLAS Profiling Completed!${reset}"

