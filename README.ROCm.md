### Hardware Requirements ###

* For ROCm hardware requirements, see [here](https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md#supported-cpus)

### Software and Driver Requirements ###

* For ROCm software requirements, see [here](https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md#the-latest-rocm-platform---rocm-15)

## Installation ##

### AMD ROCm Installation ###

For further background information on ROCm, refer [here](https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md)

Installing ROCm Debian packages:  
  
      wget -qO - http://packages.amd.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
      
      sudo sh -c 'echo deb [arch=amd64] http://packages.amd.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list'
     
      sudo apt-get update
      
      sudo apt-get install rocm rocm-opencl rocm-opencl-dev
      
      sudo reboot

Then, verify the installation. Double-check your kernel (at a minimum, you should see "kfd" in the name):

      uname -r

In addition, check that you can run the simple HSA vector_copy sample application:

      cd /opt/rocm/hsa/sample
        
      make
       
      ./vector_copy

