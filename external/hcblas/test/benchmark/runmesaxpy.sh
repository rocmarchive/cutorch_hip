#!/bin/bash -e
#This script is invoked to profile SAXPY

#CURRENT_WORK_DIRECTORY
CURRENTDIR=$PWD

#Path to profile
path2profiler="${CODEXL_PATH}/CodeXLGpuProfiler"

#Path to SAXPY executable
path2exe="$CURRENTDIR/../../build/test/src/bin/saxpy"
workingdir="$CURRENTDIR"

#Create Profile Data directory to store profile results
profDir="$workingdir/saxpyProfileData"
mkdir -p $profDir

#Check if profiler exists
if [ ! -x $path2profiler ]; then
  echo "profiler does not exist..Exiting.."
  exit
fi
echo -e "\n N\t Imple\t Avg Time(ms)" >> $workingdir/profileSummary_saxpy.csv

#Start profiling saxpy
while read line; do    
    Mvalue=$(echo $line | cut -f1 -d" " )
    Implem=$(echo $line | cut -f2 -d" " )
    datetime=$(date +%b-%d-%a_%H-%M-%S_)
#    pc="perfCounter"
    path2outdir="$profDir/$datetime$Mvalue$Implem"
    mkdir -p $path2outdir
#    path2perf="$path2outdir/$pc"
#    mkdir -p $path2perf

#Grep CLKernel Summary
    cmd="(ls -a $path2outdir) | grep HSAKernelSummary"

#Check if executable exixts
    if [ -x $path2exe ]; then
      echo $path2exe $Mvalue $Implem

#Generate ATP file
      runcmd="$path2profiler --hsatrace -o $path2outdir/output.atp -t -T -w $path2outdir $path2exe $Mvalue $Implem --device gpu"
      echo $runcmd
      eval $runcmd
      echo $cmd
      filename="$(eval $cmd)"
      passarg=$path2outdir/$filename

#Store profile timings in CSV using python script
      if [ -f "$workingdir/extracthtml_saxpy.py" ]; then
        python $workingdir/extracthtml_saxpy.py $passarg $Mvalue $Implem
      fi

#Run perf counter
#      runcmd2="$path2profiler --hsapmc -o $path2perf/output.csv -O -p -w $path2perf $path2exe $Mvalue $Implem --device gpu"
#      echo $runcmd2
#      eval $runcmd2
    else
      echo $path2exe "doesnot exist" 
    fi

#Input file
done < $workingdir/saxpy_input.txt
