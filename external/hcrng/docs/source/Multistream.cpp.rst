*************************************
2.3.3. Multistream usage Code example
*************************************
-------------------------------------------------------------------------------------------------------------------------------------------

.. code-block:: c++

  //Example on Multistream random number generation with Mrg31k3p generator 
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

  #define HCRNG_SINGLE_PRECISION
  #ifdef HCRNG_SINGLE_PRECISION
  typedef float fp_type;
  #else
  typedef double fp_type;
  #endif

  //Multistream generation with host 
  void multistream_fill_array(size_t spwi, size_t gsize, size_t quota, int substream_length, hcrngMrg31k3pStream* streams, fp_type* out_)
  {
    for (size_t i = 0; i < quota; i++) {
      for (size_t gid = 0; gid < gsize; gid++) {
      	//Create streams 
          hcrngMrg31k3pStream* s = &streams[spwi * gid];
          fp_type* out = &out_[spwi * (i * gsize + gid)];
          //Do nothing when subtsream_length is equal to 0
          if ((i > 0) && (substream_length > 0) && (i % substream_length == 0))
          //Forward to next substream when substream_length is greater than 0
              hcrngMrg31k3pForwardToNextSubstreams(spwi, s);
          else if ((i > 0) && (substream_length < 0) && (i % (-substream_length) == 0))
          //Rewind substreams when substream_length is smaller than 0
              hcrngMrg31k3pRewindSubstreams(spwi, s);
          //Generate Random Numbers
          for (size_t sid = 0; sid < spwi; sid++) {
              out[sid] = hcrngMrg31k3pRandomU01(&s[sid]);
          }
      }
    }
  }

  int main()
  {
        hcrngStatus status = HCRNG_SUCCESS;
        bool ispassed = 1;
        size_t streamBufferSize;
        //Number of streams 
        size_t streamCount = 10;
        //Number of Random numbers to be generated (numberCount should be a multiple of streamCount)
        size_t numberCount = 100;
        //Substream length
        //Substream_length       = 0   // do not use substreams
        //Substream_length       = > 0   // go to next substreams after Substream_length values
        //Substream_length       = < 0  // restart substream after Substream_length values
        int stream_length = 5; 
        size_t streams_per_thread = 2;
       
        //Enumerate the list of accelerators
        std::vector<hc::accelerator>acc = hc::accelerator::get_all();
        accelerator_view accl_view = (acc[1].create_view());

        //Allocate Host pointers 
        fp_type* Random1 = (fp_type*) malloc(sizeof(fp_type) * numberCount);
        fp_type* Random2 = (fp_type*) malloc(sizeof(fp_type) * numberCount);
        //Allocate buffer for Device output
        fp_type* outBufferDevice_substream = hc::am_alloc(sizeof(fp_type) * numberCount, acc[1], 0);
        hcrngMrg31k3pStream* streams = hcrngMrg31k3pCreateStreams(NULL, streamCount, &streamBufferSize, NULL);
        hcrngMrg31k3pStream* streams_buffer = hc::am_alloc(sizeof(hcrngMrg31k3pStream) * streamCount, acc[1], 0);
        accl_view.copy(streams, streams_buffer, streamCount* sizeof(hcrngMrg31k3pStream));

        //Invoke Random number generator function in Device
  #ifdef HCRNG_SINGLE_PRECISION        	
        status = hcrngMrg31k3pDeviceRandomU01Array_single(accl_view, streamCount, streams_buffer, numberCount, outBufferDevice_substream, stream_length, streams_per_thread);
  #else
      	status = hcrngMrg31k3pDeviceRandomU01Array_double(accl_view, streamCount, streams_buffer, numberCount, outBufferDevice_substream, stream_length, streams_per_thread);
  #endif       	
        //Status check
        if(status) std::cout << "TEST FAILED" << std::endl;
        accl_view.copy(outBufferDevice, Random1_substream, numberCount * sizeof(fp_type));       

        //Invoke random number generator from Host
        multistream_fill_array(streams_per_thread, streamCount/streams_per_thread, numberCount/streamCount, stream_length, streams, Random2);
        
        //Compare Host and device outputs
        for(int i =0; i < numberCount; i++) {
           if (Random1[i] != Random2[i]) {
                ispassed = 0;
                std::cout <<" RANDDEVICE_SUBSTREAM[" << i<< "] " << Random1[i] << "and RANDHOST_SUBSTREAM[" << i <<"] mismatches"<< Random2[i] << std::endl;
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
        hc::am_free(outBufferDevice_substream);
        hc::am_free(streams_buffer);

        return 0;     
  }

