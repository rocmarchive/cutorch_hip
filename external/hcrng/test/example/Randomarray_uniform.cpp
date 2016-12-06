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
#include <iostream>
using namespace hc;

#define HCRNG_SINGLE_PRECISION
#ifdef HCRNG_SINGLE_PRECISION
typedef float fp_type;
#else
typedef double fp_type;
#endif


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
        fp_type *Random1 = (fp_type*) malloc(sizeof(fp_type) * numberCount);
        fp_type *Random2 = (fp_type*) malloc(sizeof(fp_type) * numberCount);
        fp_type *outBufferDevice = hc::am_alloc(sizeof(fp_type) * numberCount, acc[1], 0);
  
        //Create streams
        hcrngMrg31k3pStream *streams = hcrngMrg31k3pCreateStreams(NULL, streamCount, &streamBufferSize, NULL);
        hcrngMrg31k3pStream *streams_buffer = hc::am_alloc(sizeof(hcrngMrg31k3pStream) * streamCount, acc[1], 0);
        accl_view.copy(streams, streams_buffer, streamCount* sizeof(hcrngMrg31k3pStream));

        //Invoke random number generators in device (here strean_length and streams_per_thread arguments are default) 
#ifdef HCRNG_SINGLE_PRECISION
        status = hcrngMrg31k3pDeviceRandomU01Array_single(streamCount, streams_buffer, numberCount, outBufferDevice);
#else
        status = hcrngMrg31k3pDeviceRandomU01Array_double(streamCount, streams_buffer, numberCount, outBufferDevice);
#endif
        if(status) std::cout << "TEST FAILED" << std::endl;
        accl_view.copy(outBufferDevice, Random1, numberCount * sizeof(fp_type));

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
