## A. Introduction: ##

The hcRNG library is an implementation of uniform random number generators targetting the AMD heterogenous hardware via HCC compiler runtime. The computational resources of underlying AMD heterogenous compute gets exposed and exploited through the HCC C++ frontend. Refer [here](https://bitbucket.org/multicoreware/hcc/wiki/Home) for more details on HCC compiler.

The following list enumerates the current set of RNG generators that are supported so far.

1. MRG31k3p
2. MRG32k3a
3. LFSR113
4. Philox-4x32-10

To know more, go through the [Documentation](http://hcrng-documentation.readthedocs.org/en/latest/)

## B. Key Features ##

* Support for 4 commonly used uniform random number generators.
* Single and Double precision.
* Multiple streams, created on the host and generates random numbers either on the host or on computing devices.

## C. Prerequisites ##

* Refer Prerequisites section [here](http://hcrng-documentation.readthedocs.org/en/latest/Prerequisites.html)

## D. Tested Environment so far 

* Refer Tested environments enumerated [here](http://hcrng-documentation.readthedocs.org/en/latest/Tested_Environments.html)

## E. Installation  

* Follow installation steps as described [here](http://hcrng-documentation.readthedocs.org/en/latest/Installation_steps.html)

## F. API reference

* The Specification of API's supported along with description  can be found [here](http://hcrng-documentation.readthedocs.org/en/latest/API_reference.html)

## G. Unit testing

### Testing:

     * cd ~/hcrng/test/

     * ./test.sh

### Manual testing: 

(i)  Google testing (GTEST) with Functionality check

     * cd ~/hcrng/build/test/unit/bin/
       All functions are tested against google test.

## H. Example Code

Random number generator Mrg31k3p example:

file: Randomarray.cpp

```
#!c++

//This example is a simple random array generation and it compares host output with device output
//Random number generator Mrg31k3p
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <hcRNG/mrg31k3p.h>
#include <hcRNG/hcRNG.h>
#include <hc.hpp>
#include <hc_am.hpp>
using namespace hc;

int main()
{
      hcrngStatus status = HCRNG_SUCCESS;
      bool ispassed = 1;
      size_t streamBufferSize;
      // Number oi streams
      size_t streamCount = 10;
      //Number of random numbers to be generated
      //numberCount must be a multiple of streamCount
      size_t numberCount = 100; 
      //Enumerate the list of accelerators
      std::vector<hc::accelerator>acc = hc::accelerator::get_all();
      accelerator_view accl_view = (acc[1].create_view());
      //Allocate memory for host pointers
      float *Random1 = (float*) malloc(sizeof(float) * numberCount);
      float *Random2 = (float*) malloc(sizeof(float) * numberCount);
      float *outBufferDevice = hc::am_alloc(sizeof(float) * numberCount, acc[1], 0);

      //Create streams
      hcrngMrg31k3pStream *streams = hcrngMrg31k3pCreateStreams(NULL, streamCount, &streamBufferSize, NULL);
      hcrngMrg31k3pStream *streams_buffer = hc::am_alloc(sizeof(hcrngMrg31k3pStream) * streamCount, acc[1], 0);
      accl_view.copy(streams, streams_buffer, streamCount* sizeof(hcrngMrg31k3pStream));

      //Invoke random number generators in device (here strean_length and streams_per_thread arguments are default) 
      status = hcrngMrg31k3pDeviceRandomU01Array_single(accl_view, streamCount, streams_buffer, numberCount, outBufferDevice);
 
      if(status) std::cout << "TEST FAILED" << std::endl;
      accl_view.copy(outBufferDevice, Random1, numberCount * sizeof(float));

      //Invoke random number generators in host
      for (size_t i = 0; i < numberCount; i++)
          Random2[i] = hcrngMrg31k3pRandomU01(&streams[i % streamCount]);   
      // Compare host and device outputs
      for(int i =0; i < numberCount; i++) {
          if (Random1[i] != Random2[i]) {
              ispassed = 0;
              std::cout <<" RANDDEVICE[" << i<< "] " << Random1[i] << "and RANDHOST[" << i <<"] mismatches"<< Random2[i] << std::endl;
              break;
          }
          else
              continue;
      }
      if(!ispassed) std::cout << "TEST FAILED" << std::endl;
 
      //Free host resources
      free(Random1);
      free(Random2);
      //Release device resources
      hc::am_free(outBufferDevice);
      hc::am_free(streams_buffer);
      return 0;
}  

```
* Compiling the example code:

          /opt/hcc/bin/clang++ `/opt/hcc/bin/hcc-config --cxxflags --ldflags` -lhc_am -lhcrng Randomarray.cpp