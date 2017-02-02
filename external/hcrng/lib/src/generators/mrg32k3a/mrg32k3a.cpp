#include "hcRNG/mrg32k3a.h"
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
#define MODULAR_NUMBER_TYPE unsigned long
#define MODULAR_FIXED_SIZE 3
#include "../include/hcRNG/private/modular.c.h"

// code that is common to host and device
#include "../include/hcRNG/private/mrg32k3a.c.h"
/*! @brief Matrices to advance to the next state
*/
static unsigned long Mrg32k3a_A1p0[3][3] = {
	{ 0, 1, 0 },
	{ 0, 0, 1 },
	{ 4294156359, 1403580, 0 }
};

static unsigned long Mrg32k3a_A2p0[3][3] = {
	{ 0, 1, 0 },
	{ 0, 0, 1 },
	{ 4293573854, 0, 527612 }
};


/*! @brief Inverse of Mrg32k3a_A1p0 mod Mrg32k3a_M1
*
*  Matrices to go back to the previous state.
*/
static unsigned long invA1[3][3] = {
	{ 184888585, 0, 1945170933 },
	{ 1, 0, 0 },
	{ 0, 1, 0 }
};

// inverse of Mrg32k3a_A2p0 mod Mrg32k3a_M1
static unsigned long invA2[3][3] = {
	{ 0, 360363334, 4225571728 },
	{ 1, 0, 0 },
	{ 0, 1, 0 }
};

/*! @brief Check the validity of a seed for Mrg32k3a
*/
static hcrngStatus validateSeed(const hcrngMrg32k3aStreamState* seed)
{
	// Check that the seeds have valid values
	for (size_t i = 0; i < 3; ++i)
		if (seed->g1[i] >= Mrg32k3a_M1)
			return hcrngSetErrorString(HCRNG_INVALID_SEED, "seed.g1[%u] >= Mrg32k3a_M1", i);

	for (size_t i = 0; i < 3; ++i)
		if (seed->g2[i] >= Mrg32k3a_M2)
			return hcrngSetErrorString(HCRNG_INVALID_SEED, "seed.g2[%u] >= Mrg32k3a_M2", i);

	if (seed->g1[0] == 0 && seed->g1[1] == 0 && seed->g1[2] == 0)
		return hcrngSetErrorString(HCRNG_INVALID_SEED, "seed.g1 = (0,0,0)");

	if (seed->g2[0] == 0 && seed->g2[1] == 0 && seed->g2[2] == 0)
		return hcrngSetErrorString(HCRNG_INVALID_SEED, "seed.g2 = (0,0,0)");

	return HCRNG_SUCCESS;
}

hcrngMrg32k3aStreamCreator* hcrngMrg32k3aCopyStreamCreator(const hcrngMrg32k3aStreamCreator* creator, hcrngStatus* err)
{
	hcrngStatus err_ = HCRNG_SUCCESS;

	// allocate creator
	hcrngMrg32k3aStreamCreator* newCreator = (hcrngMrg32k3aStreamCreator*)malloc(sizeof(hcrngMrg32k3aStreamCreator));

	if (newCreator == NULL)
		// allocation failed
		err_ = hcrngSetErrorString(HCRNG_OUT_OF_RESOURCES, "%s(): could not allocate memory for stream creator", __func__);
	else {
		if (creator == NULL)
			creator = &defaultStreamCreator_Mrg32k3a;
		// initialize creator
		*newCreator = *creator;
	}

	// set error status if needed
	if (err != NULL)
		*err = err_;

	return newCreator;
}

hcrngStatus hcrngMrg32k3aDestroyStreamCreator(hcrngMrg32k3aStreamCreator* creator)
{
	if (creator != NULL)
		free(creator);
	return HCRNG_SUCCESS;
}

hcrngStatus hcrngMrg32k3aRewindStreamCreator(hcrngMrg32k3aStreamCreator* creator)
{
	if (creator == NULL)
		creator = &defaultStreamCreator_Mrg32k3a;
	creator->nextState = creator->initialState;
	return HCRNG_SUCCESS;
}

hcrngStatus hcrngMrg32k3aSetBaseCreatorState(hcrngMrg32k3aStreamCreator* creator, const hcrngMrg32k3aStreamState* baseState)
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

hcrngStatus hcrngMrg32k3aChangeStreamsSpacing(hcrngMrg32k3aStreamCreator* creator, int e, int c)
{
	//Check params
	if (creator == NULL)
		return hcrngSetErrorString(HCRNG_INVALID_STREAM_CREATOR, "%s(): modifying the default stream creator is forbidden", __func__);
	if (e < 0)
		return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): e must be >= 0", __func__);

	unsigned long B[3][3];

	if (c >= 0)
		modMatPow(Mrg32k3a_A1p0, creator->nuA1, Mrg32k3a_M1, c);
	else
		modMatPow(invA1, creator->nuA1, Mrg32k3a_M1, -c);
	if (e > 0) {
		modMatPowLog2(Mrg32k3a_A1p0, B, Mrg32k3a_M1, e);
		modMatMat(B, creator->nuA1, creator->nuA1, Mrg32k3a_M1);
	}

	if (c >= 0)
		modMatPow(Mrg32k3a_A2p0, creator->nuA2, Mrg32k3a_M2, c);
	else
		modMatPow(invA2, creator->nuA2, Mrg32k3a_M2, -c);
	if (e > 0) {
		modMatPowLog2(Mrg32k3a_A2p0, B, Mrg32k3a_M2, e);
		modMatMat(B, creator->nuA2, creator->nuA2, Mrg32k3a_M2);
	}

	return HCRNG_SUCCESS;
}

hcrngMrg32k3aStream* hcrngMrg32k3aAllocStreams(size_t count, size_t* bufSize, hcrngStatus* err)
{
	hcrngStatus err_ = HCRNG_SUCCESS;
	size_t bufSize_ = count * sizeof(hcrngMrg32k3aStream);


	hcrngMrg32k3aStream* buf = (hcrngMrg32k3aStream*)malloc(bufSize_);

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

hcrngStatus hcrngMrg32k3aDestroyStreams(hcrngMrg32k3aStream* streams)
{
	if (streams != NULL)
		free(streams);
	return HCRNG_SUCCESS;
}

static hcrngStatus Mrg32k3aCreateStream(hcrngMrg32k3aStreamCreator* creator, hcrngMrg32k3aStream* buffer)
{
	//Check params
	if (buffer == NULL)
		return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): buffer cannot be NULL", __func__);

	// use default creator if not given
	if (creator == NULL)
		creator = &defaultStreamCreator_Mrg32k3a;

	// initialize stream
	buffer->current = buffer->initial = buffer->substream = creator->nextState;

	// advance next state in stream creator
	modMatVec(creator->nuA1, creator->nextState.g1, creator->nextState.g1, Mrg32k3a_M1);
	modMatVec(creator->nuA2, creator->nextState.g2, creator->nextState.g2, Mrg32k3a_M2);

	return HCRNG_SUCCESS;
}

hcrngStatus hcrngMrg32k3aCreateOverStreams(hcrngMrg32k3aStreamCreator* creator, size_t count, hcrngMrg32k3aStream* streams)
{
	// iterate over all individual stream buffers
	for (size_t i = 0; i < count; i++) {

		hcrngStatus err = Mrg32k3aCreateStream(creator, &streams[i]);

		// abort on error
		if (err != HCRNG_SUCCESS)
			return err;
	}

	return HCRNG_SUCCESS;
}

hcrngMrg32k3aStream* hcrngMrg32k3aCreateStreams(hcrngMrg32k3aStreamCreator* creator, size_t count, size_t* bufSize, hcrngStatus* err)
{
	hcrngStatus err_;
	size_t bufSize_;
	hcrngMrg32k3aStream* streams = hcrngMrg32k3aAllocStreams(count, &bufSize_, &err_);

	if (err_ == HCRNG_SUCCESS)
		err_ = hcrngMrg32k3aCreateOverStreams(creator, count, streams);

	if (bufSize != NULL)
		*bufSize = bufSize_;

	if (err != NULL)
		*err = err_;

	return streams;
}

hcrngMrg32k3aStream* hcrngMrg32k3aCopyStreams(size_t count, const hcrngMrg32k3aStream* streams, hcrngStatus* err)
{
	hcrngStatus err_ = HCRNG_SUCCESS;
	hcrngMrg32k3aStream* dest = NULL;

	//Check params
	if (streams == NULL)
		err_ = hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): stream cannot be NULL", __func__);

	if (err_ == HCRNG_SUCCESS)
		dest = hcrngMrg32k3aAllocStreams(count, NULL, &err_);

	if (err_ == HCRNG_SUCCESS)
		err_ = hcrngMrg32k3aCopyOverStreams(count, dest, streams);

	if (err != NULL)
		*err = err_;

	return dest;
}

hcrngMrg32k3aStream* hcrngMrg32k3aMakeSubstreams(hcrngMrg32k3aStream* stream, size_t count, size_t* bufSize, hcrngStatus* err)
{
	hcrngStatus err_;
	size_t bufSize_;
	hcrngMrg32k3aStream* substreams = hcrngMrg32k3aAllocStreams(count, &bufSize_, &err_);

	if (err_ == HCRNG_SUCCESS)
		err_ = hcrngMrg32k3aMakeOverSubstreams(stream, count, substreams);
	if (bufSize != NULL)
		*bufSize = bufSize_;

	if (err != NULL)
		*err = err_;

	return substreams;
}

hcrngStatus hcrngMrg32k3aAdvanceStreams(size_t count, hcrngMrg32k3aStream* streams, int e, int c)
{
	//Check params
	if (streams == NULL)
		return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): streams cannot be NULL", __func__);

	//Advance Stream
	unsigned long B1[3][3], C1[3][3], B2[3][3], C2[3][3];

	// if e == 0, do not add 2^0; just behave as in docs
	if (e > 0) {
		modMatPowLog2(Mrg32k3a_A1p0, B1, Mrg32k3a_M1, e);
		modMatPowLog2(Mrg32k3a_A2p0, B2, Mrg32k3a_M2, e);
	}
	else if (e < 0) {
		modMatPowLog2(invA1, B1, Mrg32k3a_M1, -e);
		modMatPowLog2(invA2, B2, Mrg32k3a_M2, -e);
	}

	if (c >= 0) {
		modMatPow(Mrg32k3a_A1p0, C1, Mrg32k3a_M1, c);
		modMatPow(Mrg32k3a_A2p0, C2, Mrg32k3a_M2, c);
	}
	else {
		modMatPow(invA1, C1, Mrg32k3a_M1, -c);
		modMatPow(invA2, C2, Mrg32k3a_M2, -c);
	}

	if (e) {
		modMatMat(B1, C1, C1, Mrg32k3a_M1);
		modMatMat(B2, C2, C2, Mrg32k3a_M2);
	}

	for (size_t i = 0; i < count; i++) {
		modMatVec(C1, streams[i].current.g1, streams[i].current.g1, Mrg32k3a_M1);
		modMatVec(C2, streams[i].current.g2, streams[i].current.g2, Mrg32k3a_M2);
	}

	return HCRNG_SUCCESS;
}

hcrngStatus hcrngMrg32k3aWriteStreamInfo(const hcrngMrg32k3aStream* stream, FILE *file)
{
	//Check params
	if (stream == NULL)
		return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): stream cannot be NULL", __func__);
	if (file == NULL)
		return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): file cannot be NULL", __func__);

	// The Initial state of the Stream
	fprintf(file, "\n   initial = { ");
	for (size_t i = 0; i < 3; i++)
		fprintf(file, "%lu, ", stream->initial.g1[i]);

	for (size_t i = 0; i < 2; i++)
		fprintf(file, "%lu, ", stream->initial.g2[i]);

	fprintf(file, "%lu }\n", stream->initial.g2[2]);
	//The Current state of the Stream
	fprintf(file, "\n   Current = { ");
	for (size_t i = 0; i < 3; i++)
		fprintf(file, "%lu, ", stream->current.g1[i]);

	for (size_t i = 0; i < 2; i++)
		fprintf(file, "%lu, ", stream->current.g2[i]);

	fprintf(file, "%lu }\n", stream->current.g2[2]);

	return HCRNG_SUCCESS;
}

hcrngStatus hcrngMrg32k3aDeviceRandomU01Array_single(size_t streamCount, hcrngMrg32k3aStream* streams,
	size_t numberCount, float* outBuffer, int streamlength, size_t streams_per_thread)
{
#define HCRNG_SINGLE_PRECISION
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
          if(gid < (streamCount/streams_per_thread)) {
           for(int i =0; i < numberCount/streamCount; i++) {
              if ((i > 0) && (streamlength > 0) && (i % streamlength == 0)) {
               hcrngMrg32k3aForwardToNextSubstreams(streams_per_thread, &streams[streams_per_thread * gid]);
              }
              if ((i > 0) && (streamlength < 0) && (i % streamlength == 0)) {
               hcrngMrg32k3aRewindSubstreams(streams_per_thread, &streams[streams_per_thread * gid]);
              }
              for (int j = 0; j < streams_per_thread; j++)
               outBuffer[streams_per_thread * (i * (streamCount/streams_per_thread) + gid) + j] = hcrngMrg32k3aRandomU01(&streams[streams_per_thread * gid + j]);
              }
           }
        }).wait();
#undef HCRNG_SINGLE_PRECISION
        return status;
}


hcrngStatus hcrngMrg32k3aDeviceRandomNArray_single(size_t streamCount, hcrngMrg32k3aStream *streams,
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
        hcrngStatus status = hcrngMrg32k3aDeviceRandomU01Array_single(streamCount, streams,numberCount, outBuffer, streamlength, streams_per_thread);
        if (status == HCRNG_SUCCESS){
               	status = box_muller_transform_single(accl_view, mu, sigma, outBuffer, numberCount);
                return status;
            }
#undef HCRNG_SINGLE_PRECISION
        return status;
}

hcrngStatus hcrngMrg32k3aDeviceRandomU01Array_double(size_t streamCount, hcrngMrg32k3aStream* streams,
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
        long size = (streamCount/streams_per_thread + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1);
        hc::extent<1> grdExt(size);
        hc::tiled_extent<1> t_ext(grdExt, BLOCK_SIZE);
        hc::parallel_for_each(accl_view, t_ext, [ = ] (hc::tiled_index<1> tidx) __attribute__((hc, cpu)) {
           int gid = tidx.global[0];
           if(gid < streamCount/streams_per_thread) {
           for(int i =0; i < numberCount/streamCount; i++) {
              if ((i > 0) && (streamlength > 0) && (i % streamlength == 0)) {
               hcrngMrg32k3aForwardToNextSubstreams(streams_per_thread, &streams[streams_per_thread * gid]);
              }
              if ((i > 0) && (streamlength < 0) && (i % streamlength == 0)) {
               hcrngMrg32k3aRewindSubstreams(streams_per_thread, &streams[streams_per_thread * gid]);
              }
              for (int j = 0; j < streams_per_thread; j++)
               outBuffer[streams_per_thread * (i * (streamCount/streams_per_thread) + gid) + j] = hcrngMrg32k3aRandomU01(&streams[streams_per_thread * gid + j]);
              }
           }
        }).wait();
        return status;
}

hcrngStatus hcrngMrg32k3aDeviceRandomNArray_double(size_t streamCount, hcrngMrg32k3aStream *streams,
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
        hcrngStatus status = hcrngMrg32k3aDeviceRandomU01Array_double(streamCount, streams,numberCount, outBuffer, streamlength, streams_per_thread);
        if (status == HCRNG_SUCCESS) {
	    	status = box_muller_transform_double(accl_view, mu, sigma, outBuffer, numberCount);
                return status;
               }
        return status;
}
