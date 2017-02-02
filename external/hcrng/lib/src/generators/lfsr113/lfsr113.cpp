#include <hcRNG/lfsr113.h>
#include "hcRNG/hcRNG.h"
#include "hcRNG/box_muller_transform.h"
#include <iostream>
#include <hc.hpp>
#include "hc_short_vector.hpp"

//using namespace hc;
using namespace hc;
using namespace hc::short_vector;
using namespace std;

#include <stdlib.h>
#define BLOCK_SIZE 256
// code that is common to host and device
#include "../include/hcRNG/private/lfsr113.c.h"

/*! @brief Check the validity of a seed for Lfsr113
*/
static hcrngStatus validateSeed(const hcrngLfsr113StreamState* seed)
{
	// Check that the seeds have valid values
	if (seed->g[0] < 2)
		return hcrngSetErrorString(HCRNG_INVALID_SEED, "seed.g[%u] must be greater than 1", 0);

	if (seed->g[1] < 8)
		return hcrngSetErrorString(HCRNG_INVALID_SEED, "seed.g[%u] must be greater than 7", 1);

	if (seed->g[2] < 16)
		return hcrngSetErrorString(HCRNG_INVALID_SEED, "seed.g[%u] must be greater than 15", 2);

	if (seed->g[3] < 128)
		return hcrngSetErrorString(HCRNG_INVALID_SEED, "seed.g[%u] must be greater than 127", 3);

	return HCRNG_SUCCESS;
}

hcrngLfsr113StreamCreator* hcrngLfsr113CopyStreamCreator(const hcrngLfsr113StreamCreator* creator, hcrngStatus* err)
{
	hcrngStatus err_ = HCRNG_SUCCESS;

	// allocate creator
	hcrngLfsr113StreamCreator* newCreator = (hcrngLfsr113StreamCreator*)malloc(sizeof(hcrngLfsr113StreamCreator));

	if (newCreator == NULL)
		// allocation failed
		err_ = hcrngSetErrorString(HCRNG_OUT_OF_RESOURCES, "%s(): could not allocate memory for stream creator", __func__);
	else {
		if (creator == NULL)
			creator = &defaultStreamCreator_Lfsr113;
		// initialize creator
		*newCreator = *creator;
	}

	// set error status if needed
	if (err != NULL)
		*err = err_;

	return newCreator;
}

hcrngStatus hcrngLfsr113DestroyStreamCreator(hcrngLfsr113StreamCreator* creator)
{
	if (creator != NULL)
          free(creator);
	return HCRNG_SUCCESS;
}

hcrngStatus hcrngLfsr113RewindStreamCreator(hcrngLfsr113StreamCreator* creator)
{
	if (creator == NULL)
		creator = &defaultStreamCreator_Lfsr113;
	creator->nextState = creator->initialState;
	return HCRNG_SUCCESS;
}

hcrngStatus hcrngLfsr113SetBaseCreatorState(hcrngLfsr113StreamCreator* creator, const hcrngLfsr113StreamState* baseState)
{
	//Check params
	if (creator == NULL)
		return hcrngSetErrorString(HCRNG_INVALID_STREAM_CREATOR, "%s(): modifying the default stream creator is forbidden", __func__);
	if (baseState == NULL)
		return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): baseState cannot be NULL", __func__);

	hcrngStatus err = validateSeed(baseState);

	if (err == HCRNG_SUCCESS) {
		// initialize new creator
		creator->initialState = creator->nextState = *baseState;
	}

	return err;
}

hcrngStatus hcrngLfsr113ChangeStreamsSpacing(hcrngLfsr113StreamCreator* creator, int e, int c)
{
	return hcrngSetErrorString(HCRNG_FUNCTION_NOT_IMPLEMENTED, "%s(): Not Implemented", __func__);
}

hcrngLfsr113Stream* hcrngLfsr113AllocStreams(size_t count, size_t* bufSize, hcrngStatus* err)
{
	hcrngStatus err_ = HCRNG_SUCCESS;
	size_t bufSize_ = count * sizeof(hcrngLfsr113Stream);

	// allocate streams
	hcrngLfsr113Stream* buf = (hcrngLfsr113Stream*)malloc(bufSize_);

	if (buf == NULL) {
		// allocation failed
		err_ = hcrngSetErrorString(HCRNG_OUT_OF_RESOURCES, "%s(): could not allocate memory for streams", __func__);
		bufSize_ = 0;
	}

	// set buffer size if needed
	if (bufSize != NULL)
		*bufSize = bufSize_;

	// set error status if needed
	if (err != NULL)
		*err = err_;

	return buf;
}

hcrngStatus hcrngLfsr113DestroyStreams(hcrngLfsr113Stream* streams)
{
	if (streams != NULL)
		free(streams);
	return HCRNG_SUCCESS;
}
void lfsr113AdvanceState(hcrngLfsr113StreamState* currentState)
{
	int z, b;
	unsigned int* nextSeed = currentState->g;

	//Calculate the new value for nextSeed[0]
	z = nextSeed[0] & (unsigned int)(-2);
	b = (z << 6) ^ z;
	z = (z) ^ (z << 2) ^ (z << 3) ^ (z << 10) ^ (z << 13) ^
		(z << 16) ^ (z << 19) ^ (z << 22) ^ (z << 25) ^
		(z << 27) ^ (z << 28) ^
		((b >> 3) & 0x1FFFFFFF) ^
		((b >> 4) & 0x0FFFFFFF) ^
		((b >> 6) & 0x03FFFFFF) ^
		((b >> 9) & 0x007FFFFF) ^
		((b >> 12) & 0x000FFFFF) ^
		((b >> 15) & 0x0001FFFF) ^
		((b >> 18) & 0x00003FFF) ^
		((b >> 21) & 0x000007FF);
	nextSeed[0] = z;

	//Calculate the new value for nextSeed[1]
	z = nextSeed[1] & (unsigned int)(-8);
	b = (z << 2) ^ z;
	z = ((b >> 13) & 0x0007FFFF) ^ (z << 16);
	nextSeed[1] = z;

	//Calculate the new value for nextSeed[2]
	z = nextSeed[2] & (unsigned int)(-16);
	b = (z << 13) ^ z;
	z = (z << 2) ^ (z << 4) ^ (z << 10) ^ (z << 12) ^ (z << 13) ^
		(z << 17) ^ (z << 25) ^
		((b >> 3) & 0x1FFFFFFF) ^
		((b >> 11) & 0x001FFFFF) ^
		((b >> 15) & 0x0001FFFF) ^
		((b >> 16) & 0x0000FFFF) ^
		((b >> 24) & 0x000000FF);
	nextSeed[2] = z;

	//Calculate the new value for nextSeed[3]
	z = nextSeed[3] & (unsigned int)(-128);
	b = (z << 3) ^ z;
	z = (z << 9) ^ (z << 10) ^ (z << 11) ^ (z << 14) ^ (z << 16) ^
		(z << 18) ^ (z << 23) ^ (z << 24) ^
		((b >> 1) & 0x7FFFFFFF) ^
		((b >> 2) & 0x3FFFFFFF) ^
		((b >> 7) & 0x01FFFFFF) ^
		((b >> 9) & 0x007FFFFF) ^
		((b >> 11) & 0x001FFFFF) ^
		((b >> 14) & 0x0003FFFF) ^
		((b >> 15) & 0x0001FFFF) ^
		((b >> 16) & 0x0000FFFF) ^
		((b >> 23) & 0x000001FF) ^
		((b >> 24) & 0x000000FF);

	nextSeed[3] = z;
}
static hcrngStatus Lfsr113CreateStream(hcrngLfsr113StreamCreator* creator, hcrngLfsr113Stream* buffer)
{
	//Check params
	if (buffer == NULL)
		return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): buffer cannot be NULL", __func__);

	// use default creator if not given
	if (creator == NULL)
		creator = &defaultStreamCreator_Lfsr113;

	// initialize stream
	buffer->current = buffer->initial = buffer->substream = creator->nextState;

	//Advance next state in stream creator
	lfsr113AdvanceState(&creator->nextState);

	return HCRNG_SUCCESS;
}

hcrngStatus hcrngLfsr113CreateOverStreams(hcrngLfsr113StreamCreator* creator, size_t count, hcrngLfsr113Stream* streams)
{
	// iterate over all individual stream buffers
	for (size_t i = 0; i < count; i++) {

		hcrngStatus err = Lfsr113CreateStream(creator, &streams[i]);

		// abort on error
		if (err != HCRNG_SUCCESS)
			return err;
	}

	return HCRNG_SUCCESS;
}

hcrngLfsr113Stream* hcrngLfsr113CreateStreams(hcrngLfsr113StreamCreator* creator, size_t count, size_t* bufSize, hcrngStatus* err)
{
	hcrngStatus err_;
	size_t bufSize_;
	hcrngLfsr113Stream* streams = hcrngLfsr113AllocStreams(count, &bufSize_, &err_);

	if (err_ == HCRNG_SUCCESS)
		err_ = hcrngLfsr113CreateOverStreams(creator, count, streams);

	if (bufSize != NULL)
		*bufSize = bufSize_;

	if (err != NULL)
		*err = err_;

	return streams;
}

hcrngLfsr113Stream* hcrngLfsr113CopyStreams(size_t count, const hcrngLfsr113Stream* streams, hcrngStatus* err)
{
	hcrngStatus err_ = HCRNG_SUCCESS;
	hcrngLfsr113Stream* dest = NULL;

	//Check params
	if (streams == NULL)
		err_ = hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): stream cannot be NULL", __func__);

	if (err_ == HCRNG_SUCCESS)
		dest = hcrngLfsr113AllocStreams(count, NULL, &err_);

	if (err_ == HCRNG_SUCCESS)
		err_ = hcrngLfsr113CopyOverStreams(count, dest, streams);

	if (err != NULL)
		*err = err_;

	return dest;
}

hcrngLfsr113Stream* hcrngLfsr113MakeSubstreams(hcrngLfsr113Stream* stream, size_t count, size_t* bufSize, hcrngStatus* err)
{
	hcrngStatus err_;
	size_t bufSize_;
	hcrngLfsr113Stream* substreams = hcrngLfsr113AllocStreams(count, &bufSize_, &err_);

	if (err_ == HCRNG_SUCCESS)
		err_ = hcrngLfsr113MakeOverSubstreams(stream, count, substreams);

	if (bufSize != NULL)
		*bufSize = bufSize_;

	if (err != NULL)
		*err = err_;

	return substreams;
}

hcrngStatus hcrngLfsr113AdvanceStreams(size_t count, hcrngLfsr113Stream* streams, int e, int c)
{
	return hcrngSetErrorString(HCRNG_FUNCTION_NOT_IMPLEMENTED, "%s(): Not Implemented", __func__);
}

hcrngStatus hcrngLfsr113WriteStreamInfo(const hcrngLfsr113Stream* stream, FILE *file)
{
	//Check params
	if (stream == NULL)
		return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): stream cannot be NULL", __func__);
	if (file == NULL)
		return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): file cannot be NULL", __func__);

	// The Initial state of the Stream
	fprintf(file, "\n   initial = { ");
	for (size_t i = 0; i < 3; i++)
		fprintf(file, "%u, ", stream->initial.g[i]);

	fprintf(file, "%u }\n", stream->initial.g[3]);

	//The Current state of the Stream
	fprintf(file, "\n   Current = { ");
	for (size_t i = 0; i < 3; i++)
		fprintf(file, "%u, ", stream->current.g[i]);

	fprintf(file, "%u }\n", stream->current.g[3]);

	return HCRNG_SUCCESS;
}

hcrngStatus hcrngLfsr113DeviceRandomU01Array_single(size_t streamCount, hcrngLfsr113Stream* streams,
	size_t numberCount, float* outBuffer, int streamlength, size_t streams_per_thread)
{
#define HCRNG_SINGLE_PRECISION
        std::vector<hc::accelerator>acc = hc::accelerator::get_all();
        accelerator_view accl_view = (acc[1].get_default_view());
	//Check params
	if (streamCount < 1)
		return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): streamCount cannot be less than 1", __func__);
	if (numberCount < 1)
		return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): numberCount cannot be less than 1", __func__);
        if (numberCount % streamCount != 0)
                return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): numberCount must be a multiple of streamCount", __func__);
        hcrngStatus status = HCRNG_SUCCESS;
        long size = ((streamCount/streams_per_thread) + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1);
        hc::extent<1> grdExt(size);
        hc::tiled_extent<1> t_ext(grdExt, BLOCK_SIZE);
        hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<1> tidx) __attribute__((hc, cpu)) {
           int gid = tidx.global[0];
           if(gid < streamCount/streams_per_thread) {
           for(int i =0; i < numberCount/streamCount; i++) {
              if ((i > 0) && (streamlength > 0) && (i % streamlength == 0)) {
               hcrngLfsr113ForwardToNextSubstreams(streams_per_thread, &streams[streams_per_thread * gid]);
              }
              if ((i > 0) && (streamlength < 0) && (i % streamlength == 0)) {
               hcrngLfsr113RewindSubstreams(streams_per_thread, &streams[streams_per_thread * gid]);
              }
              for (int j = 0; j < streams_per_thread; j++)
               outBuffer[streams_per_thread * (i * (streamCount/streams_per_thread) + gid) + j] = hcrngLfsr113RandomU01(&streams[streams_per_thread * gid + j]);
              }
           }
        }).wait();
#undef HCRNG_SINGLE_PRECISION
        return status;
}

hcrngStatus hcrngLfsr113DeviceRandomNArray_single(size_t streamCount, hcrngLfsr113Stream *streams,
	size_t numberCount, float mu, float sigma, float *outBuffer, int streamlength, size_t streams_per_thread)
{
#define HCRNG_SINGLE_PRECISION
        std::vector<hc::accelerator>acc = hc::accelerator::get_all();
        accelerator_view accl_view = (acc[1].get_default_view());
	if (streamCount < 1)
		return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): streamCount cannot be less than 1", __func__);
	if (numberCount < 1)
		return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): numberCount cannot be less than 1", __func__);
        if (numberCount % streamCount != 0)
                return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): numberCount must be a multiple of streamCount", __func__);
        hcrngStatus status = hcrngLfsr113DeviceRandomU01Array_single(streamCount, streams,numberCount, outBuffer, streamlength, streams_per_thread);
        if (status == HCRNG_SUCCESS){
	    	status = box_muller_transform_single(accl_view, mu, sigma, outBuffer, numberCount);
                return status;
               }
#undef HCRNG_SINGLE_PRECISION
        return status;
}

hcrngStatus hcrngLfsr113DeviceRandomU01Array_double(size_t streamCount, hcrngLfsr113Stream* streams,
        size_t numberCount, double* outBuffer, int streamlength, size_t streams_per_thread)
{
        //Check params
        std::vector<hc::accelerator>acc = hc::accelerator::get_all();
        accelerator_view accl_view = (acc[1].get_default_view());
        if (streamCount < 1)
                return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): streamCount cannot be less than 1", __func__);
        if (numberCount < 1)
                return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): numberCount cannot be less than 1", __func__);
        if (numberCount % streamCount != 0)
                return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): numberCount must be a multiple of streamCount", __func__);
        hcrngStatus status = HCRNG_SUCCESS;
        long size = ((streamCount/streams_per_thread) + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1);
        hc::extent<1> grdExt(size);
        hc::tiled_extent<1> t_ext(grdExt, BLOCK_SIZE);
        hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<1> tidx) __attribute__((hc, cpu)) {
           int gid = tidx.global[0];
           if(gid < streamCount/streams_per_thread) {
           for(int i =0; i < numberCount/streamCount; i++) {
              if ((i > 0) && (streamlength > 0) && (i % streamlength == 0)) {
               hcrngLfsr113ForwardToNextSubstreams(streams_per_thread, &streams[streams_per_thread * gid]);
              }
              if ((i > 0) && (streamlength < 0) && (i % streamlength == 0)) {
               hcrngLfsr113RewindSubstreams(streams_per_thread, &streams[streams_per_thread * gid]);
              }
              for (int j = 0; j < streams_per_thread; j++)
               outBuffer[streams_per_thread * (i * (streamCount/streams_per_thread) + gid) + j] = hcrngLfsr113RandomU01(&streams[streams_per_thread * gid + j]);
              }
           }
        }).wait();
        return status;
}

hcrngStatus hcrngLfsr113DeviceRandomNArray_double(size_t streamCount, hcrngLfsr113Stream *streams,
	size_t numberCount, double mu, double sigma, double *outBuffer, int streamlength, size_t streams_per_thread)
{
        std::vector<hc::accelerator>acc = hc::accelerator::get_all();
        accelerator_view accl_view = (acc[1].get_default_view());
	if (streamCount < 1)
		return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): streamCount cannot be less than 1", __func__);
	if (numberCount < 1)
		return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): numberCount cannot be less than 1", __func__);
        if (numberCount % streamCount != 0)
                return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): numberCount must be a multiple of streamCount", __func__);
        hcrngStatus status = hcrngLfsr113DeviceRandomU01Array_double(streamCount, streams,numberCount, outBuffer, streamlength, streams_per_thread);
        if (status == HCRNG_SUCCESS){
	    	status = box_muller_transform_double(accl_view, mu, sigma, outBuffer, numberCount);
                return status;
                }
        return status;
}
