#include <hcRNG/hcRNG.h>
#include <hcRNG/mrg31k3p.h>
#include <hcRNG/mrg32k3a.h>
#include <hcRNG/lfsr113.h>
#include <hcRNG/philox432.h>
#include "gtest/gtest.h"
#define STREAM_COUNT 10

TEST(hcrng_Substreams, Return_Check_Substreams_Mrg31k3p ) {
    hcrngMrg31k3pStream* stream1 = NULL;
    hcrngStatus status, err;
    hcrngMrg31k3pStreamCreator* creator1 = NULL;

    /* Create substreams with NULL creator */
    size_t streamBufferSize;
    hcrngMrg31k3pStream *stream2 = hcrngMrg31k3pCreateStreams(creator1, STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Destination substream is NULL */
    status = hcrngMrg31k3pCopyOverStreams (STREAM_COUNT, stream1, stream2);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);

    /* source substream is NULL */
    status = hcrngMrg31k3pCopyOverStreams (STREAM_COUNT, stream2, stream1);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);

    /* Create substreams with allocated creator */
    hcrngMrg31k3pStreamCreator* creator2 = hcrngMrg31k3pCopyStreamCreator(NULL, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);
    hcrngMrg31k3pStream *stream3 = hcrngMrg31k3pCreateStreams(creator2, STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Destination substream is NULL */
    status = hcrngMrg31k3pCopyOverStreams (STREAM_COUNT, stream1, stream3);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);

    /* source substream is NULL */
    status = hcrngMrg31k3pCopyOverStreams (STREAM_COUNT, stream3, stream1);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);

    /* pass allocated substreams */
    status = hcrngMrg31k3pCopyOverStreams (STREAM_COUNT, stream2, stream3);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Forward to next substream with NULL substream and allocated substream */
    status = hcrngMrg31k3pForwardToNextSubstreams(STREAM_COUNT, stream1);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);
    status = hcrngMrg31k3pForwardToNextSubstreams(STREAM_COUNT, stream2);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Rewind substream with NULL substream and allocated substream */
    status = hcrngMrg31k3pRewindSubstreams(STREAM_COUNT, stream1);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);
    status = hcrngMrg31k3pRewindSubstreams(STREAM_COUNT, stream2);
    EXPECT_EQ(status, HCRNG_SUCCESS);
 
    /* Make over substreams */
    hcrngMrg31k3pStream* substreams = hcrngMrg31k3pAllocStreams(STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);
    status = hcrngMrg31k3pMakeOverSubstreams(stream2, STREAM_COUNT, substreams);/*stream2 is allocated*/
    EXPECT_EQ(status, HCRNG_SUCCESS);
    status = hcrngMrg31k3pMakeOverSubstreams(stream1, STREAM_COUNT, substreams);/*stream1 is not allocated*/
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);

    /* Make Substreams */
    hcrngMrg31k3pStream* substreams1 = hcrngMrg31k3pMakeSubstreams(stream1, STREAM_COUNT, &streamBufferSize, &err);/* stream1 is not allocated */
    EXPECT_EQ(err, HCRNG_INVALID_VALUE);
    hcrngMrg31k3pStream* substreams2 = hcrngMrg31k3pMakeSubstreams(stream2, STREAM_COUNT, &streamBufferSize, &err);/* stream2 is allocated */
    EXPECT_EQ(err, HCRNG_SUCCESS);
}

TEST(hcrng_Substreams, Return_Check_Substreams_Mrg32k3a ) {
    hcrngMrg32k3aStream* stream1 = NULL;
    hcrngStatus status, err;
    hcrngMrg32k3aStreamCreator* creator1 = NULL;

    /* Create substreams with NULL creator */
    size_t streamBufferSize;
    hcrngMrg32k3aStream *stream2 = hcrngMrg32k3aCreateStreams(creator1, STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Destination substream is NULL */
    status = hcrngMrg32k3aCopyOverStreams (STREAM_COUNT, stream1, stream2);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);

    /* source substream is NULL */
    status = hcrngMrg32k3aCopyOverStreams (STREAM_COUNT, stream2, stream1);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);

    /* Create substreams with allocated creator */
    hcrngMrg32k3aStreamCreator* creator2 = hcrngMrg32k3aCopyStreamCreator(NULL, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);
    hcrngMrg32k3aStream *stream3 = hcrngMrg32k3aCreateStreams(creator2, STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Destination substream is NULL */
    status = hcrngMrg32k3aCopyOverStreams (STREAM_COUNT, stream1, stream3);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);

    /* source substream is NULL */
    status = hcrngMrg32k3aCopyOverStreams (STREAM_COUNT, stream3, stream1);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);

    /* pass allocated substreams */
    status = hcrngMrg32k3aCopyOverStreams (STREAM_COUNT, stream2, stream3);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Forward to next substream with NULL substream and allocated substream */
    status = hcrngMrg32k3aForwardToNextSubstreams(STREAM_COUNT, stream1);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);
    status = hcrngMrg32k3aForwardToNextSubstreams(STREAM_COUNT, stream2);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Rewind substream with NULL substream and allocated substream */
    status = hcrngMrg32k3aRewindSubstreams(STREAM_COUNT, stream1);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);
    status = hcrngMrg32k3aRewindSubstreams(STREAM_COUNT, stream2);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Make over substreams */
    hcrngMrg32k3aStream* substreams = hcrngMrg32k3aAllocStreams(STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);
    status = hcrngMrg32k3aMakeOverSubstreams(stream2, STREAM_COUNT, substreams);/*stream2 is allocated*/
    EXPECT_EQ(status, HCRNG_SUCCESS);
    status = hcrngMrg32k3aMakeOverSubstreams(stream1, STREAM_COUNT, substreams);/*stream1 is not allocated*/
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);

    /* Make Substreams */
    hcrngMrg32k3aStream* substreams1 = hcrngMrg32k3aMakeSubstreams(stream1, STREAM_COUNT, &streamBufferSize, &err);/* stream1 is not allocated */
    EXPECT_EQ(err, HCRNG_INVALID_VALUE);
    hcrngMrg32k3aStream* substreams2 = hcrngMrg32k3aMakeSubstreams(stream2, STREAM_COUNT, &streamBufferSize, &err);/* stream2 is allocated */
    EXPECT_EQ(err, HCRNG_SUCCESS);
}

TEST(hcrng_Substreams, Return_Check_Substreams_Lfsr113 ) {
    hcrngLfsr113Stream* stream1 = NULL;
    hcrngStatus status, err;
    hcrngLfsr113StreamCreator* creator1 = NULL;

    /* Create substreams with NULL creator */
    size_t streamBufferSize;
    hcrngLfsr113Stream *stream2 = hcrngLfsr113CreateStreams(creator1, STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Destination substream is NULL */
    status = hcrngLfsr113CopyOverStreams (STREAM_COUNT, stream1, stream2);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);

    /* source substream is NULL */
    status = hcrngLfsr113CopyOverStreams (STREAM_COUNT, stream2, stream1);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);

    /* Create substreams with allocated creator */
    hcrngLfsr113StreamCreator* creator2 = hcrngLfsr113CopyStreamCreator(NULL, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);
    hcrngLfsr113Stream *stream3 = hcrngLfsr113CreateStreams(creator2, STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Destination substream is NULL */
    status = hcrngLfsr113CopyOverStreams (STREAM_COUNT, stream1, stream3);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);

    /* source substream is NULL */
    status = hcrngLfsr113CopyOverStreams (STREAM_COUNT, stream3, stream1);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);

    /* pass allocated substreams */
    status = hcrngLfsr113CopyOverStreams (STREAM_COUNT, stream2, stream3);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Forward to next substream with NULL substream and allocated substream */
    status = hcrngLfsr113ForwardToNextSubstreams(STREAM_COUNT, stream1);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);
    status = hcrngLfsr113ForwardToNextSubstreams(STREAM_COUNT, stream2);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Rewind substream with NULL substream and allocated substream */
    status = hcrngLfsr113RewindSubstreams(STREAM_COUNT, stream1);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);
    status = hcrngLfsr113RewindSubstreams(STREAM_COUNT, stream2);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Make over substreams */
    hcrngLfsr113Stream* substreams = hcrngLfsr113AllocStreams(STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);
    status = hcrngLfsr113MakeOverSubstreams(stream2, STREAM_COUNT, substreams);/*stream2 is allocated*/
    EXPECT_EQ(status, HCRNG_SUCCESS);
    status = hcrngLfsr113MakeOverSubstreams(stream1, STREAM_COUNT, substreams);/*stream1 is not allocated*/
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);

    /* Make Substreams */
    hcrngLfsr113Stream* substreams1 = hcrngLfsr113MakeSubstreams(stream1, STREAM_COUNT, &streamBufferSize, &err);/* stream1 is not allocated */
    EXPECT_EQ(err, HCRNG_INVALID_VALUE);
    hcrngLfsr113Stream* substreams2 = hcrngLfsr113MakeSubstreams(stream2, STREAM_COUNT, &streamBufferSize, &err);/* stream2 is allocated */
    EXPECT_EQ(err, HCRNG_SUCCESS);
}

TEST(hcrng_Substreams, Return_Check_Substreams_Philox432 ) {
    hcrngPhilox432Stream* stream1 = NULL;
    hcrngStatus status, err;
    hcrngPhilox432StreamCreator* creator1 = NULL;

    /* Create substreams with NULL creator */
    size_t streamBufferSize;
    hcrngPhilox432Stream *stream2 = hcrngPhilox432CreateStreams(creator1, STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Destination substream is NULL */
    status = hcrngPhilox432CopyOverStreams (STREAM_COUNT, stream1, stream2);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);

    /* source substream is NULL */
    status = hcrngPhilox432CopyOverStreams (STREAM_COUNT, stream2, stream1);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);

    /* Create substreams with allocated creator */
    hcrngPhilox432StreamCreator* creator2 = hcrngPhilox432CopyStreamCreator(NULL, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);
    hcrngPhilox432Stream *stream3 = hcrngPhilox432CreateStreams(creator2, STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);

    /* Destination substream is NULL */
    status = hcrngPhilox432CopyOverStreams (STREAM_COUNT, stream1, stream3);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);

    /* source substream is NULL */
    status = hcrngPhilox432CopyOverStreams (STREAM_COUNT, stream3, stream1);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);

    /* pass allocated substreams */
    status = hcrngPhilox432CopyOverStreams (STREAM_COUNT, stream2, stream3);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Forward to next substream with NULL substream and allocated substream */
    status = hcrngPhilox432ForwardToNextSubstreams(STREAM_COUNT, stream1);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);
    status = hcrngPhilox432ForwardToNextSubstreams(STREAM_COUNT, stream2);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Rewind substream with NULL substream and allocated substream */
    status = hcrngPhilox432RewindSubstreams(STREAM_COUNT, stream1);
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);
    status = hcrngPhilox432RewindSubstreams(STREAM_COUNT, stream2);
    EXPECT_EQ(status, HCRNG_SUCCESS);

    /* Make over substreams */
    hcrngPhilox432Stream* substreams = hcrngPhilox432AllocStreams(STREAM_COUNT, &streamBufferSize, &err);
    EXPECT_EQ(err, HCRNG_SUCCESS);
    status = hcrngPhilox432MakeOverSubstreams(stream2, STREAM_COUNT, substreams);/*stream2 is allocated*/
    EXPECT_EQ(status, HCRNG_SUCCESS);
    status = hcrngPhilox432MakeOverSubstreams(stream1, STREAM_COUNT, substreams);/*stream1 is not allocated*/
    EXPECT_EQ(status, HCRNG_INVALID_VALUE);

    /* Make Substreams */
    hcrngPhilox432Stream* substreams1 = hcrngPhilox432MakeSubstreams(stream1, STREAM_COUNT, &streamBufferSize, &err);/* stream1 is not allocated */
    EXPECT_EQ(err, HCRNG_INVALID_VALUE);
    hcrngPhilox432Stream* substreams2 = hcrngPhilox432MakeSubstreams(stream2, STREAM_COUNT, &streamBufferSize, &err);/* stream2 is allocated */
    EXPECT_EQ(err, HCRNG_SUCCESS);
}
