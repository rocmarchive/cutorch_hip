#include <hcRNG/hcRNG.h>
#include <hcRNG/mrg31k3p.h>
#include <hcRNG/mrg32k3a.h>
#include <hcRNG/lfsr113.h>
#include <hcRNG/philox432.h>
#include "gtest/gtest.h"
#define STREAM_COUNT 10

TEST(hcrng_Streams, Return_Check_Streams_Mrg31k3p ) {
    hcrngMrg31k3pStream* stream1 = NULL;
    hcrngStatus status, err;

   /* Create Over streams when passing NULL stream */
    status = hcrngMrg31k3pCreateOverStreams(NULL, STREAM_COUNT, stream1);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE); 

    /* Copy streams with NULL buffer */
    hcrngMrg31k3pStream* stream = hcrngMrg31k3pCopyStreams(STREAM_COUNT, stream1, &err);
    EXPECT_EQ(err, HCRNG_INVALID_VALUE);

   /* Destroy NULL stream */
    status = hcrngMrg31k3pDestroyStreams(stream1);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Allocate streams of size streamBufferSize */
    size_t streamBufferSize;
    hcrngMrg31k3pStream *stream2 = hcrngMrg31k3pAllocStreams(STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Create Over Streams with NULL Creator */
    hcrngMrg31k3pStreamCreator* creator1 = NULL;
    status = hcrngMrg31k3pCreateOverStreams(creator1, STREAM_COUNT, stream2);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Create streams with NULL creator */
    hcrngMrg31k3pStream *stream3 = hcrngMrg31k3pCreateStreams(creator1, STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Destroy streams of streamBufferSize */
    status = hcrngMrg31k3pDestroyStreams(stream2);
    EXPECT_EQ(status, HCRNG_SUCCESS);
    status = hcrngMrg31k3pDestroyStreams(stream3);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Allocate streams of size streamBufferSize */
    hcrngMrg31k3pStream *stream4 = hcrngMrg31k3pAllocStreams(STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Copy stream creator*/
    hcrngMrg31k3pStreamCreator* creator2 = hcrngMrg31k3pCopyStreamCreator(NULL, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Create over streams with allocated creator */
    status = hcrngMrg31k3pCreateOverStreams(creator2, STREAM_COUNT, stream4);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Create streams with allocated creator */
    hcrngMrg31k3pStream *stream5 = hcrngMrg31k3pCreateStreams(creator2, STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Copy streams with allocated buffer */
    hcrngMrg31k3pStream* stream6 = hcrngMrg31k3pCopyStreams(STREAM_COUNT, stream5, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS); 

    /* Destroy streams */
    status = hcrngMrg31k3pDestroyStreams(stream4);
    EXPECT_EQ(status, HCRNG_SUCCESS);
    status = hcrngMrg31k3pDestroyStreams(stream5);
    EXPECT_EQ(status, HCRNG_SUCCESS);
}

TEST(hcrng_Streams, Return_Check_Streams_Mrg32k3a ) {
    hcrngMrg32k3aStream* stream1 = NULL;
    hcrngStatus status, err;

    /* Create Over streams when passing NULL stream */
    status = hcrngMrg32k3aCreateOverStreams(NULL, STREAM_COUNT, stream1);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);

    /* Copy streams with NULL buffer */
    hcrngMrg32k3aStream* stream = hcrngMrg32k3aCopyStreams(STREAM_COUNT, stream1, &err);
    EXPECT_EQ(err, HCRNG_INVALID_VALUE);


    /* Destroy NULL stream */
    status = hcrngMrg32k3aDestroyStreams(stream1);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Allocate streams of size streamBufferSize */
    size_t streamBufferSize;
    hcrngMrg32k3aStream *stream2 = hcrngMrg32k3aAllocStreams(STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Create Over Streams with NULL Creator */
    hcrngMrg32k3aStreamCreator* creator1 = NULL;
    status = hcrngMrg32k3aCreateOverStreams(creator1, STREAM_COUNT, stream2);
    EXPECT_EQ(status, HCRNG_SUCCESS);
    
    /* Create streams with NULL creator */
    hcrngMrg32k3aStream *stream3 = hcrngMrg32k3aCreateStreams(creator1, STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Destroy streams of streamBufferSize */
    status = hcrngMrg32k3aDestroyStreams(stream2);
    EXPECT_EQ(status, HCRNG_SUCCESS);
    status = hcrngMrg32k3aDestroyStreams(stream3);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Allocate streams of size streamBufferSize */
    hcrngMrg32k3aStream *stream4 = hcrngMrg32k3aAllocStreams(STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Copy stream creator*/
    hcrngMrg32k3aStreamCreator* creator2 = hcrngMrg32k3aCopyStreamCreator(NULL, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Create over streams with allocated creator */
    status = hcrngMrg32k3aCreateOverStreams(creator2, STREAM_COUNT, stream4);
    EXPECT_EQ(status, HCRNG_SUCCESS);
 
    /* Create streams with allocated creator */
    hcrngMrg32k3aStream *stream5 = hcrngMrg32k3aCreateStreams(creator2, STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);
   
    /* Copy streams with allocated buffer */
    hcrngMrg32k3aStream* stream6 = hcrngMrg32k3aCopyStreams(STREAM_COUNT, stream5, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Destroy streams of streamBufferSize */
    status = hcrngMrg32k3aDestroyStreams(stream4);
    EXPECT_EQ(status, HCRNG_SUCCESS);
    status = hcrngMrg32k3aDestroyStreams(stream5);
    EXPECT_EQ(status, HCRNG_SUCCESS);
}

TEST(hcrng_Streams, Return_Check_Streams_Lfsr113 ) {
    hcrngLfsr113Stream* stream1 = NULL;
    hcrngStatus status, err;

    /* Create Over streams when passing NULL stream */
    status = hcrngLfsr113CreateOverStreams(NULL, STREAM_COUNT, stream1);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);

    /* Copy streams with NULL buffer */
    hcrngLfsr113Stream* stream = hcrngLfsr113CopyStreams(STREAM_COUNT, stream1, &err);
    EXPECT_EQ(err, HCRNG_INVALID_VALUE);
  
    /* Destroy NULL stream */
    status = hcrngLfsr113DestroyStreams(stream1);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Allocate streams of size streamBufferSize */
    size_t streamBufferSize;
    hcrngLfsr113Stream *stream2 = hcrngLfsr113AllocStreams(STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Create Over Streams with NULL Creator */
    hcrngLfsr113StreamCreator* creator1 = NULL;
    status = hcrngLfsr113CreateOverStreams(creator1, STREAM_COUNT, stream2);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Create streams with NULL creator */
    hcrngLfsr113Stream *stream3 = hcrngLfsr113CreateStreams(creator1, STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Destroy streams of streamBufferSize */
    status = hcrngLfsr113DestroyStreams(stream2);
    EXPECT_EQ(status, HCRNG_SUCCESS);
    status = hcrngLfsr113DestroyStreams(stream3);
    EXPECT_EQ(status, HCRNG_SUCCESS);
 
    /* Allocate streams of size streamBufferSize */
    hcrngLfsr113Stream *stream4 = hcrngLfsr113AllocStreams(STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Copy stream creator*/
    hcrngLfsr113StreamCreator* creator2 = hcrngLfsr113CopyStreamCreator(NULL, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Create over streams with allocated creator */
    status = hcrngLfsr113CreateOverStreams(creator2, STREAM_COUNT, stream4);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Create streams with allocated creator */
    hcrngLfsr113Stream *stream5 = hcrngLfsr113CreateStreams(creator2, STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Copy streams with allocated buffer */
    hcrngLfsr113Stream* stream6 = hcrngLfsr113CopyStreams(STREAM_COUNT, stream5, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Destroy streams of streamBufferSize */
    status = hcrngLfsr113DestroyStreams(stream4);
    EXPECT_EQ(status, HCRNG_SUCCESS);
    status = hcrngLfsr113DestroyStreams(stream5);
    EXPECT_EQ(status, HCRNG_SUCCESS);
}

TEST(hcrng_Streams, Return_Check_Streams_Philox432 ) {
    hcrngPhilox432Stream* stream1 = NULL;
    hcrngStatus status, err;

    /* Create Over streams when passing NULL stream */
    status = hcrngPhilox432CreateOverStreams(NULL, STREAM_COUNT, stream1);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);

    /* Copy streams with NULL buffer */
    hcrngPhilox432Stream* stream = hcrngPhilox432CopyStreams(STREAM_COUNT, stream1, &err);
    EXPECT_EQ(err, HCRNG_INVALID_VALUE);

 
    /* Destroy NULL stream */
    status = hcrngPhilox432DestroyStreams(stream1);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Allocate streams of size streamBufferSize */
    size_t streamBufferSize;
    hcrngPhilox432Stream *stream2 = hcrngPhilox432AllocStreams(STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Create Over Streams with NULL Creator */
    hcrngPhilox432StreamCreator* creator1 = NULL;
    status = hcrngPhilox432CreateOverStreams(creator1, STREAM_COUNT, stream2);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Create streams with NULL creator */
    hcrngPhilox432Stream *stream3 = hcrngPhilox432CreateStreams(creator1, STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Destroy streams of streamBufferSize */
    status = hcrngPhilox432DestroyStreams(stream2);
    EXPECT_EQ(status, HCRNG_SUCCESS);
    status = hcrngPhilox432DestroyStreams(stream3);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Allocate streams of size streamBufferSize */
    hcrngPhilox432Stream *stream4 = hcrngPhilox432AllocStreams(STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Copy stream creator*/
    hcrngPhilox432StreamCreator* creator2 = hcrngPhilox432CopyStreamCreator(NULL, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Create over streams with allocated creator */
    status = hcrngPhilox432CreateOverStreams(creator2, STREAM_COUNT, stream4);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Create streams with allocated creator */
    hcrngPhilox432Stream *stream5 = hcrngPhilox432CreateStreams(creator2, STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Copy streams with allocated buffer */
    hcrngPhilox432Stream* stream6 = hcrngPhilox432CopyStreams(STREAM_COUNT, stream5, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Destroy streams of streamBufferSize */
    status = hcrngPhilox432DestroyStreams(stream4);
    EXPECT_EQ(status, HCRNG_SUCCESS);
    status = hcrngPhilox432DestroyStreams(stream5);
    EXPECT_EQ(status, HCRNG_SUCCESS);
}

