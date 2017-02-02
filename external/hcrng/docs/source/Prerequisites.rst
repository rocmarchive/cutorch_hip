******************
1.2. Prerequisites
******************
-------------------------------------------------------------------------------------------------------------------------------------------

This section lists the known set of hardware and software requirements to build this library

1.2.1. Hardware
^^^^^^^^^^^^^^^

* CPU: mainstream brand, Better if with >=4 Cores Intel Haswell based CPU
* System Memory >= 4GB (Better if >10GB for NN application over multiple GPUs)
* Hard Drive > 200GB (Better if SSD or NVMe driver  for NN application over multiple GPUs)
* Minimum GPU Memory (Global) > 2GB

1.2.2. GPU cards supported
^^^^^^^^^^^^^^^^^^^^^^^^^^

* dGPU: AMD R9 Fury X, R9 Fury, R9 Nano
* APU: AMD Kaveri or Carrizo

1.2.3. AMD Driver and Runtime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Radeon Open Compute Kernel (ROCK) driver : https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver
* HSA runtime API and runtime for Boltzmann:  https://github.com/RadeonOpenCompute/ROCR-Runtime

1.2.4. System software
^^^^^^^^^^^^^^^^^^^^^^

* Ubuntu 14.04 trusty
* GCC 4.6 and later
* CPP 4.6 and later (come with GCC package)
* python 2.7 and later
* HCC 0.9 from `here <https://bitbucket.org/multicoreware/hcc/downloads/hcc-0.9.16041-0be508d-ff03947-5a1009a-Linux.deb>`_


1.2.5. Tools and Misc
^^^^^^^^^^^^^^^^^^^^^

* git 1.9 and later
* cmake 2.6 and later (2.6 and 2.8 are tested)
* firewall off
* root privilege or user account in sudo group


1.2.6. Ubuntu Packages
^^^^^^^^^^^^^^^^^^^^^^

* libc6-dev-i386
* liblapack-dev
* graphicsmagick
