#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <hcRNG/lfsr113.h>
#include <hcRNG/hcRNG.h>
#include <hc.hpp>
#include <hc_am.hpp>
#include "gtest/gtest.h"
using namespace hc;

void multistream_fill_array_uniform(size_t spwi, size_t gsize, size_t quota, int substream_length, hcrngLfsr113Stream* streams, float* out_)
{
  for (size_t i = 0; i < quota; i++) {
      for (size_t gid = 0; gid < gsize; gid++) {
          hcrngLfsr113Stream* s = &streams[spwi * gid];
          float* out = &out_[spwi * (i * gsize + gid)];
          if ((i > 0) && (substream_length > 0) && (i % substream_length == 0))
              hcrngLfsr113ForwardToNextSubstreams(spwi, s);
          else if ((i > 0) && (substream_length < 0) && (i % (-substream_length) == 0))
              hcrngLfsr113RewindSubstreams(spwi, s);
          for (size_t sid = 0; sid < spwi; sid++) {
              out[sid] = hcrngLfsr113RandomU01(&s[sid]);
          }
      }
  }
}



TEST(lfsr113Single_test_uniform, Functional_check_lfsr113Single_uniform)
{
        hcrngLfsr113Stream* stream = NULL;
        hcrngStatus status = HCRNG_SUCCESS;
        bool ispassed1 = 1, ispassed2 = 1;
        size_t streamBufferSize;
        size_t NbrStreams = 1;
        size_t streamCount = 10;
        size_t numberCount = 100;
        int stream_length = 5;
        size_t streams_per_thread = 2;
        float *Random1 = (float*) malloc(sizeof(float) * numberCount);
        float *Random2 = (float*) malloc(sizeof(float) * numberCount);
        std::vector<hc::accelerator>acc = hc::accelerator::get_all();
        accelerator_view accl_view = (acc[1].create_view());
        float *outBufferDevice = hc::am_alloc(sizeof(float) * numberCount, acc[1], 0);
        hcrngLfsr113Stream *streams = hcrngLfsr113CreateStreams(NULL, streamCount, &streamBufferSize, NULL);
        hcrngLfsr113Stream *streams_buffer = hc::am_alloc(sizeof(hcrngLfsr113Stream) * streamCount, acc[1], 0);
        accl_view.copy(streams, streams_buffer, streamCount* sizeof(hcrngLfsr113Stream));
        status = hcrngLfsr113DeviceRandomU01Array_single(streamCount, streams_buffer, numberCount, outBufferDevice);
        EXPECT_EQ(status, 0);
        accl_view.copy(outBufferDevice, Random1, numberCount * sizeof(float));
        for (size_t i = 0; i < numberCount; i++)
           Random2[i] = hcrngLfsr113RandomU01(&streams[i % streamCount]);   
        for(int i =0; i < numberCount; i++) {
           EXPECT_EQ(Random1[i], Random2[i]);
        }
        float *Random3 = (float*) malloc(sizeof(float) * numberCount);
        float *Random4 = (float*) malloc(sizeof(float) * numberCount);
        float *outBufferDevice_substream = hc::am_alloc(sizeof(float) * numberCount, acc[1], 0);
        status = hcrngLfsr113DeviceRandomU01Array_single(streamCount, streams_buffer, numberCount, outBufferDevice_substream, stream_length, streams_per_thread);
        EXPECT_EQ(status, 0);
        accl_view.copy(outBufferDevice_substream, Random3, numberCount * sizeof(float));
        multistream_fill_array_uniform(streams_per_thread, streamCount/streams_per_thread, numberCount/streamCount, stream_length, streams, Random4);
        for(int i =0; i < numberCount; i++) {
           EXPECT_EQ(Random3[i], Random4[i]);
        }

}


