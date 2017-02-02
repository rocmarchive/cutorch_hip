#!/bin/bash -e
#This script is invoked to profile SGEMM

#CURRENT_WORK_DIRECTORY
CURRENTDIR=$PWD
export HCBLAS_PATH=$CURRENTDIR/../../

cd $CURRENTDIR/../../build/test/ && cmake -DCMAKE_CXX_FLAGS=-fPIC $HCBLAS_PATH/test/ -DPROFILE=ON 
set +e
mkdir $CURRENTDIR/../../build/test/src/bin/
mkdir $CURRENTDIR/../../build/test/unit/gtest/bin/
set -e
make
cd $CURRENTDIR

#Path to SGEMM executable
path2exe="$CURRENTDIR/../../build/test/src/bin/sgemm_st_timer"
workingdir="$CURRENTDIR"

#Create Profile Data directory to store profile results
profDir="$workingdir/sgemmbenchData"
mkdir -p $profDir

echo -e "\n M\t N\t K\t TransA\t TransB\t lda\t ldb\t ldc\t alpha\t beta\t aoff\t boff\t coff\t Avg Time(ms) \t GFlops/s" >> $workingdir/Benchmark_sgemm.csv

while read line; do
    Input=$(echo $line | cut -f1 -d" " )
echo -e $Input >> $workingdir/Profilesummary_sgemm.csv

#Start profiling sgemm
while read line; do
    Mvalue=$(echo $line | cut -f1 -d" " )
    Nvalue=$(echo $line | cut -f2 -d" " )
    Kvalue=$(echo $line | cut -f3 -d" " )
    transA=$(echo $line | cut -f4 -d" " )
    transB=$(echo $line | cut -f5 -d" " )
    lda=$(echo $line | cut -f6 -d" " )
    ldb=$(echo $line | cut -f7 -d" " )
    ldc=$(echo $line | cut -f8 -d" " )
    alpha=$(echo $line | cut -f9 -d" " )
    beta=$(echo $line | cut -f10 -d" " )
    aoff=$(echo $line | cut -f11 -d" " )
    boff=$(echo $line | cut -f12 -d" " )
    coff=$(echo $line | cut -f13 -d" " )
    datetime=$(date +%b-%d-%a_%H-%M-%S_)
    path2outdir="$profDir/$datetime$Mvalue$Nvalue$Kvalue$transA$transB$lda$ldb$ldc$alpha$beta$aoff$boff$coff"
    mkdir -p $path2outdir

#Check if executable exixts
    if [ -x $path2exe ]; then
      echo $path2exe $Mvalue $Nvalue $Kvalue $transA $transB $lda $ldb $ldc $alpha $beta $aoff $boff $coff  

#Generate ATP file
      runcmd="$path2exe $Mvalue $Nvalue $Kvalue $transA $transB $lda $ldb $ldc $alpha $beta $aoff $boff $coff >> $path2outdir/output_$datetime.txt"
      echo $runcmd
      eval $runcmd
      filename="output_$datetime.txt"
      passarg=$path2outdir/$filename

#Store profile timings in CSV using python script
      if [ -f "$workingdir/extracttime_sgemm.py" ]; then
        python $workingdir/extracttime_sgemm.py $passarg $Mvalue $Nvalue $Kvalue $transA $transB $lda $ldb $ldc $alpha $beta $aoff $boff $coff 
      fi

    else
      echo $path2exe "doesnot exist" 
    fi

#Input file
done < $workingdir/$Input

done < $workingdir/Input_timer.txt
