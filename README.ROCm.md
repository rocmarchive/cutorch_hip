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

###Torch installation###

To install Torch, we follow the guidelines outlined in the [Torch getting started](http://torch.ch/docs/getting-started.html#_) site with one modification. The HIP-ified versions of CuTorch and CuNN are not the latest versions, and therefore are compatible with an earlier version of Torch. The hipification of the most recent versions of CuTorch and CuNN is in progress.
Another modification is that we found it more convenient to build Torch using Lua rather than LuaJIT. 


    git clone https://github.com/torch/distro.git ~/torch --recursive 
    cd ~/torch
    git checkout a58889e
    git submodule update --init --recursive
    bash install-deps
    TORCH_LUA_VERSION=LUA52 ./install.sh
 
###CuTorch and CuNN installation###
For both Torch modules, CuTorch and CuNN, the current working branches are labeled 'hip'. CuTorch has dependencies of hcBLAS and hcRNG. These dependencies can be found at

https://github.com/ROCmSoftwarePlatform/hcBLAS
https://github.com/ROCmSoftwarePlatform/hcRNG.

To install hcBLAS:

    cd ~ && git clone -b master https://github.com/ROCmSoftwarePlatform/hcBLAS.git 
    cd hcBLAS && ./build.sh 
    dpkg -i build/*.deb
    cd /opt/rocm/hcblas/lib && ln -s libhipblas_hcc.so libhipblas.so

For hcRNG:

    cd ~ && git clone -b master https://github.com/ROCmSoftwarePlatform/hcRNG.git 
    cd hcRNG && ./build.sh 
    dpkg -i build/*.deb
    cd /opt/rocm/hcrng/lib && ln -s libhiprng_hcc.so libhiprng.so

After install all of the dependencies, we can now move forward with CuTorch. To install CuTorch, we following the prescription:

    cd ~ 
    git clone -b hip https://github.com/ROCmSoftwarePlatform/cutorch_hip.git && cd ~/cutorch_hip 
    CLAMP_NOTILECHECK=ON luarocks make ./rocks/cutorch-scm-1.rockspec 

The CuNN installation, goes as follows:

    cd ~ 
    git clone -b hip https://github.com/ROCmSoftwarePlatform/cunn_hip.git 
    cd ~/cunn_hip 
    CLAMP_NOTILECHECK=ON luarocks make ./rocks/cunn-scm-1.rockspec 

Currently, the passing rates of the directed tests of CuTorch and CuNN are 92% and 97% respectively.

