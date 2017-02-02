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

#Path to profile
path2profiler="${CODEXL_PATH}/CodeXLGpuProfiler"

#Path to SGEMM executable
path2exe="$CURRENTDIR/../../build/test/src/bin/sgemm_cn"
workingdir="$CURRENTDIR"

#Create Profile Data directory to store profile results
profDir="$workingdir/SgemmprofileData"
mkdir -p $profDir

#Check if profiler exists
if [ ! -x $path2profiler ]; then
  echo "profiler does not exist..Exiting.."
  exit
fi
echo -e "\n M\t N\t K\t TransA\t TransB\t lda\t ldb\t ldc\t alpha\t beta\t aoff\t boff\t coff\t Avg Time(ms)" >> $workingdir/Profilesummary_sgemm.csv

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

#Grep CLKernel Summary
    cmd="(ls -a $path2outdir) | grep HSAKernelSummary"

#Check if executable exixts
    if [ -x $path2exe ]; then
      echo $path2exe $Mvalue $Nvalue $Kvalue $transA $transB $lda $ldb $ldc $alpha $beta $aoff $boff $coff  

#Generate ATP file
      runcmd="$path2profiler --hsatrace -o $path2outdir/output.atp -t -T -w $path2outdir $path2exe $Mvalue $Nvalue $Kvalue $transA $transB $lda $ldb $ldc $alpha $beta $aoff $boff $coff --device gpu"
      echo $runcmd
      eval $runcmd
      echo $cmd
      filename="$(eval $cmd)"
      passarg=$path2outdir/$filename

#Store profile timings in CSV using python script
      if [ -f "$workingdir/extracthtml_sgemm.py" ]; then
        python $workingdir/extracthtml_sgemm.py $passarg $Mvalue $Nvalue $Kvalue $transA $transB $lda $ldb $ldc $alpha $beta $aoff $boff $coff 
      fi

    else
      echo $path2exe "doesnot exist" 
    fi

#Input file
done < $workingdir/$Input

done < $workingdir/Input.txt
