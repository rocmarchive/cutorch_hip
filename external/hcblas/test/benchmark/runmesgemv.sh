#!/bin/bash -e
#This script is invoked to profile SGEMV

#CURRENT_WORK_DIRECTORY
CURRENTDIR=$PWD

#Path to profile
path2profiler="${CODEXL_PATH}/CodeXLGpuProfiler"

#Path to SGEMV executable
path2exe="$CURRENTDIR/../../build/test/src/bin/sgemv"
workingdir="$CURRENTDIR"

#Create Profile Data directory to store profile results
profDir="$workingdir/sgemvProfileData"
mkdir -p $profDir

#Check if profiler exists
if [ ! -x $path2profiler ]; then
  echo "profiler does not exist..Exiting.."
  exit
fi
echo -e "\n M\t N\t Trans\t Imple\t Avg Time(ms)" >> $workingdir/profileSummary_sgemv.csv

#Start profiling sgemv
while read line; do    
    Mvalue=$(echo $line | cut -f1 -d" " )
    Nvalue=$(echo $line | cut -f2 -d" " )
    transA=$(echo $line | cut -f3 -d" " )
    Implem=$(echo $line | cut -f4 -d" " )
    datetime=$(date +%b-%d-%a_%H-%M-%S_)
#    pc="perfCounter"
    path2outdir="$profDir/$datetime$Mvalue$Nvalue$transA$Implem"
    mkdir -p $path2outdir
#    path2perf="$path2outdir/$pc"
#    mkdir -p $path2perf

#Grep CLKernel Summary
    cmd="(ls -a $path2outdir) | grep HSAKernelSummary"

#Check if executable exixts
    if [ -x $path2exe ]; then
      echo $path2exe $Mvalue $Nvalue $transA $Implem

#Generate ATP file
      runcmd="$path2profiler --hsatrace -o $path2outdir/output.atp -t -T -w $path2outdir $path2exe $Mvalue $Nvalue $transA $Implem --device gpu"
      echo $runcmd
      eval $runcmd
      echo $cmd
      filename="$(eval $cmd)"
      passarg=$path2outdir/$filename

#Store profile timings in CSV using python script
      if [ -f "$workingdir/extracthtml_sgemv.py" ]; then
        python $workingdir/extracthtml_sgemv.py $passarg $Mvalue $Nvalue $transA $Implem
      fi

#Run perf counter
#      runcmd2="$path2profiler --hsapmc -o $path2perf/output.csv -O -p -w $path2perf $path2exe $Mvalue $Nvalue $transA $Implem --device gpu"
#      echo $runcmd2
#      eval $runcmd2
    else
      echo $path2exe "doesnot exist" 
    fi

#Input file
done < $workingdir/sgemv_input.txt
