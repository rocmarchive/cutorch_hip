#!/bin/bash -e
#This script is invoked to profile SGEMM

#CURRENT_WORK_DIRECTORY
CURRENTDIR=$PWD

#Path to profile
path2profiler="${CODEXL_PATH}/CodeXLGpuProfiler"

#Path to SGEMM executable
path2exe="$CURRENTDIR/../../build/test/src/bin/sgemm"
workingdir="$CURRENTDIR"

#Create Profile Data directory to store profile results
profDir="$workingdir/sgemmProfileData"
mkdir -p $profDir

#Check if profiler exists
if [ ! -x $path2profiler ]; then
  echo "profiler does not exist..Exiting.."
  exit
fi
echo -e "\n M\t N\t K\t TransA\t TransB\t Imple\t Avg Time(ms)" >> $workingdir/profileSummary_sgemm.csv

#Start profiling sgemm
while read line; do
    Mvalue=$(echo $line | cut -f1 -d" " )
    Nvalue=$(echo $line | cut -f2 -d" " )
    Kvalue=$(echo $line | cut -f3 -d" " )
    transA=$(echo $line | cut -f4 -d" " )
    transB=$(echo $line | cut -f5 -d" " )
    Implem=$(echo $line | cut -f6 -d" " )
    datetime=$(date +%b-%d-%a_%H-%M-%S_)
#    pc="perfCounter"
    path2outdir="$profDir/$datetime$Mvalue$Nvalue$Kvalue$transA$transB$Implem"
    mkdir -p $path2outdir
#    path2perf="$path2outdir/$pc"
#    mkdir -p $path2perf

#Grep CLKernel Summary
    cmd="(ls -a $path2outdir) | grep HSAKernelSummary"

#Check if executable exixts
    if [ -x $path2exe ]; then
      echo $path2exe $Mvalue $Nvalue $Kvalue $transA $transB  

#Generate ATP file
      runcmd="$path2profiler --hsatrace -o $path2outdir/output.atp -t -T -w $path2outdir $path2exe $Mvalue $Nvalue $Kvalue $transA $transB $Implem --device gpu"
      echo $runcmd
      eval $runcmd
      echo $cmd
      filename="$(eval $cmd)"
      passarg=$path2outdir/$filename

#Store profile timings in CSV using python script
      if [ -f "$workingdir/extracthtml_sgemm.py" ]; then
        python $workingdir/extracthtml_sgemm.py $passarg $Mvalue $Nvalue $Kvalue $transA $transB $Implem
      fi

#Run perf counter
#      runcmd2="$path2profiler --hsapmc -o $path2perf/output.csv -O -p -w $path2perf $path2exe $Mvalue $Nvalue $Kvalue $transA $transB $Implem --device gpu"
#      echo $runcmd2
#      eval $runcmd2
    else
      echo $path2exe "doesnot exist" 
    fi

#Input file
done < $workingdir/sgemm_input.txt
