#!/bin/bash
#This script is invoked to test all generators of the hcfft library
#Preliminary version

# CURRENT_WORK_DIRECTORY
current_work_dir=$PWD

red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`

export HCFFT_LIBRARY_PATH=$current_work_dir/../../build/lib/src/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HCFFT_LIBRARY_PATH

testdirectories=(common normal uniform)
common=(lfsrGenerate mrg31k3pGenerate mrg32k3aGenerate philox432Generate)
normal=(lfsr113Single_test_normal mrg31k3pSingle_test_normal mrg32k3aSingle_test_normal philox432Single_test_normal lfsr113Double_test_normal mrg31k3pDouble_test_normal mrg32k3aDouble_test_normal philox432Double_test_normal)
uniform=(lfsr113Single_test_uniform mrg31k3pSingle_test_uniform mrg32k3aSingle_test_uniform philox432Single_test_uniform lfsr113Double_test_uniform mrg31k3pDouble_test_uniform mrg32k3aDouble_test_uniform philox432Double_test_uniform)


## now loop through the above array
for i in 0 1 2
do
  working_dir1="$current_work_dir/../../build/test/HIP_Unit_Tests/${testdirectories[$i]}/bin/"
  cd $working_dir1

  for j in 0 1 2 3
  do
    if [ $i == 0 ];then
      unittest="$working_dir1/${common[$j]}"
      runcmd1="$unittest 2> testlog.txt"
      eval $runcmd1

      Log_file="$working_dir1/testlog.txt"
      if [ ! -e "$Log_file" ];then
        echo "{red}TEST IS NOT WORKING....${reset}"
      else
        if grep -q error "$Log_file";
        then
          echo "${red} HIP${common[$j]}               ----- [ FAILED ]${reset}"
          rm -f $working_dir1/testlog.txt
        else
          echo "${green} HIP${common[$j]}             ----- [ PASSED ]${reset}"
          rm -f $working_dir1/testlog.txt
        fi
      fi
    fi
  done

  for j in 0 1 2 3 4 5 6 7
  do
    #rm -f $working_dir1/testlog.txt
    if [ $i == 1 ];then
      unittest="$working_dir1/${normal[$j]}"
      runcmd1="$unittest 2> testlog.txt"
      eval $runcmd1

      Log_file="$working_dir1/testlog.txt"
      if [ ! -e "$Log_file" ];then
        echo "{red}TEST IS NOT WORKING....${reset}"
      else
        if grep -q error "$Log_file";
        then
          echo "${red} HIP${normal[$j]}               ----- [ FAILED ]${reset}"
          rm -f $working_dir1/testlog.txt
        else
          echo "${green} HIP${normal[$j]}             ----- [ PASSED ]${reset}"
          rm -f $working_dir1/testlog.txt
        fi
      fi
    elif [ $i == 2 ];then
          unittest="$working_dir1/${uniform[$j]}"
      runcmd1="$unittest 2> testlog.txt"
      eval $runcmd1

      Log_file="$working_dir1/testlog.txt"
      if [ ! -e "$Log_file" ];then
        echo "{red}TEST IS NOT WORKING....${reset}"
      else
        if grep -q error "$Log_file";
        then
          echo "${red} HIP${uniform[$j]}               ----- [ FAILED ]${reset}"
          rm -f $working_dir1/testlog.txt
        else
          echo "${green} HIP${uniform[$j]}             ----- [ PASSED ]${reset}"
          rm -f $working_dir1/testlog.txt
        fi
      fi
    fi
  done
done
