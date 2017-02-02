#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <hcRNG/philox432.h>
#include <hcRNG/hcRNG.h>
#include <hc.hpp>
#include <hc_am.hpp>
#include "gtest/gtest.h"
using namespace hc;

void multistream_fill_array_normal(size_t spwi, size_t gsize, size_t quota, int substream_length, hcrngPhilox432Stream* streams, double* out_)
{
  for (size_t i = 0; i < quota; i++) {
      for (size_t gid = 0; gid < gsize; gid++) {
          hcrngPhilox432Stream* s = &streams[spwi * gid];
          double* out = &out_[spwi * (i * gsize + gid)];
          if ((i > 0) && (substream_length > 0) && (i % substream_length == 0))
              hcrngPhilox432ForwardToNextSubstreams(spwi, s);
          else if ((i > 0) && (substream_length < 0) && (i % (-substream_length) == 0))
              hcrngPhilox432RewindSubstreams(spwi, s);
          for (size_t sid = 0; sid < spwi; sid++) {
              out[sid] = hcrngPhilox432RandomN(&s[sid], &s[sid + 1], 0.0, 1.0);
          }
      }
  }
}

TEST(philox432Double_test_normal, Functional_check_philox432Double_normal)
{
        hcrngPhilox432Stream* stream = NULL;
        hcrngStatus status = HCRNG_SUCCESS;
        bool ispassed1 = 1, ispassed2 = 1;
        size_t streamBufferSize;
        size_t NbrStreams = 1;
        size_t streamCount = 10;
        size_t numberCount = 100;
        int stream_length = 5;
        size_t streams_per_thread = 2;
        double *Random1 = (double*) malloc(sizeof(double) * numberCount);
        double *Random2 = (double*) malloc(sizeof(double) * numberCount);
        std::vector<hc::accelerator>acc = hc::accelerator::get_all();
        accelerator_view accl_view = (acc[1].create_view());
        double *outBufferDevice = hc::am_alloc(sizeof(double) * numberCount, acc[1], 0);
        hcrngPhilox432Stream *streams = hcrngPhilox432CreateStreams(NULL, streamCount, &streamBufferSize, NULL);
        hcrngPhilox432Stream *streams_buffer = hc::am_alloc(sizeof(hcrngPhilox432Stream) * streamCount, acc[1], 0);
        accl_view.copy(streams, streams_buffer, streamCount* sizeof(hcrngPhilox432Stream));
        status = hcrngPhilox432DeviceRandomNArray_double(streamCount, streams_buffer, numberCount, 0.0, 1.0, outBufferDevice);
        EXPECT_EQ(status, 0);
        accl_view.copy(outBufferDevice, Random1, numberCount * sizeof(double));
        for (size_t i = 0; i < numberCount; i++)
           Random2[i] = hcrngPhilox432RandomN(&streams[i % streamCount], &streams[(i + 1) % streamCount], 0.0, 1.0);   
        for(int i =0; i < numberCount; i++) {
           EXPECT_NEAR(Random1[i], Random2[i], 0.00001);
        }
        double *Random3 = (double*) malloc(sizeof(double) * numberCount);
        double *Random4 = (double*) malloc(sizeof(double) * numberCount);
        double *outBufferDevice_substream = hc::am_alloc(sizeof(double) * numberCount, acc[1], 0);
        status = hcrngPhilox432DeviceRandomNArray_double(streamCount, streams_buffer, numberCount, 0.0, 1.0, outBufferDevice_substream, stream_length, streams_per_thread);
        EXPECT_EQ(status, 0);
        accl_view.copy(outBufferDevice_substream, Random3, numberCount * sizeof(double));
        multistream_fill_array_normal(streams_per_thread, streamCount/streams_per_thread, numberCount/streamCount, stream_length, streams, Random4);
        for(int i =0; i < numberCount; i++) {
           EXPECT_NEAR(Random3[i], Random4[i], 0.00001);
        }
}


