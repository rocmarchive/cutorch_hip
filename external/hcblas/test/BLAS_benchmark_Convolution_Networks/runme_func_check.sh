#!/bin/bash -e
#This script is invoked to check the functionality of SGEMM

#CURRENT_WORK_DIRECTORY
CURRENTDIR=$PWD

#Path to SGEMM executable
path2exe="$CURRENTDIR/../../build/test/src/bin/sgemm_cn"
workingdir="$CURRENTDIR"

while read line; do
    Input=$(echo $line | cut -f1 -d" " )

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

#Check if executable exixts
    if [ -x $path2exe ]; then
      echo $path2exe $Mvalue $Nvalue $Kvalue $transA $transB $lda $ldb $ldc $alpha $beta $aoff $boff $coff 

#Generate ATP file
      runcmd="$path2exe $Mvalue $Nvalue $Kvalue $transA $transB $lda $ldb $ldc $alpha $beta $aoff $boff $coff --device gpu >> test_log.txt"
      echo $runcmd
      eval $runcmd

    else
      echo $path2exe "doesnot exist" 
    fi

#Input file
done < $workingdir/$Input

done < $workingdir/Input.txt

#create color code
red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`

echo "TEST PASSED" >> $workingdir/testlog_temp.txt
echo "TEST PASSED" >> $workingdir/test_log.txt

DIFF=$(diff test_log.txt testlog_temp.txt)
if [ "$DIFF" != "" ] 
then
    echo "${red}Functionality check ----- [ FAILED ]${reset}"
    rm testlog_temp.txt
else
    echo "${green}Functionality check ----- [ PASSED ]${reset}"
    rm test*
fi


