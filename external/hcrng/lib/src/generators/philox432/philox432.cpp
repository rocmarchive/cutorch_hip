#include "hcRNG/philox432.h"
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
#include "hcRNG/private/philox432.c.h"
/*! @brief Check the validity of a seed for Philox432
*/
static hcrngStatus validateSeed(const hcrngPhilox432StreamState* seed)
{
	return HCRNG_SUCCESS;
}

hcrngPhilox432StreamCreator* hcrngPhilox432CopyStreamCreator(const hcrngPhilox432StreamCreator* creator, hcrngStatus* err)
{
	hcrngStatus err_ = HCRNG_SUCCESS;

	// allocate creator
	hcrngPhilox432StreamCreator* newCreator = (hcrngPhilox432StreamCreator*)malloc(sizeof(hcrngPhilox432StreamCreator));

	if (newCreator == NULL)
		// allocation failed
		err_ = hcrngSetErrorString(HCRNG_OUT_OF_RESOURCES, "%s(): could not allocate memory for stream creator", __func__);
	else {
		if (creator == NULL)
			creator = &defaultStreamCreator_Philox432;
		// initialize creator
		*newCreator = *creator;
	}

	// set error status if needed
	if (err != NULL)
		*err = err_;

	return newCreator;
}

hcrngStatus hcrngPhilox432DestroyStreamCreator(hcrngPhilox432StreamCreator* creator)
{
	if (creator != NULL)
		free(creator);
	return HCRNG_SUCCESS;
}

hcrngStatus hcrngPhilox432RewindStreamCreator(hcrngPhilox432StreamCreator* creator)
{
	if (creator == NULL)
		creator = &defaultStreamCreator_Philox432;
	creator->nextState = creator->initialState;
	return HCRNG_SUCCESS;
}

hcrngStatus hcrngPhilox432SetBaseCreatorState(hcrngPhilox432StreamCreator* creator, const hcrngPhilox432StreamState* baseState)
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

hcrngStatus hcrngPhilox432ChangeStreamsSpacing(hcrngPhilox432StreamCreator* creator, int e, int c)
{
	//Check params
	if (creator == NULL)
		return hcrngSetErrorString(HCRNG_INVALID_STREAM_CREATOR, "%s(): modifying the default stream creator is forbidden", __func__);
	if (e < 2 && e != 0)
		return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): e must be 0 or >= 2", __func__);
	if ((c % 4) != 0)
		return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): c must be a multiple of 4", __func__);

	//Create Base Creator
	hcrngPhilox432StreamCreator* baseCreator = hcrngPhilox432CopyStreamCreator(NULL, NULL);
	hcrngPhilox432StreamState baseState = { { { 0, 0 }, { 0, 0 } },	{ 0, 0, 0, 0 },	0 };
	hcrngPhilox432SetBaseCreatorState(baseCreator, &baseState);

	//Create stream
	hcrngPhilox432Stream* dumpStream = hcrngPhilox432CreateStreams(baseCreator, 1, NULL, NULL);
	
	//Advance stream
	hcrngPhilox432AdvanceStreams(1, dumpStream, e, c);
	creator->JumpDistance = dumpStream->current.ctr;
	
	//Free ressources
	hcrngPhilox432DestroyStreamCreator(baseCreator);
	hcrngPhilox432DestroyStreams(dumpStream);

	return HCRNG_SUCCESS;
}

hcrngPhilox432Stream* hcrngPhilox432AllocStreams(size_t count, size_t* bufSize, hcrngStatus* err)
{
	hcrngStatus err_ = HCRNG_SUCCESS;
	size_t bufSize_ = count * sizeof(hcrngPhilox432Stream);

	// allocate streams
	hcrngPhilox432Stream* buf = (hcrngPhilox432Stream*)malloc(bufSize_);

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

hcrngStatus hcrngPhilox432DestroyStreams(hcrngPhilox432Stream* streams)
{
	if (streams != NULL)
		free(streams);
	return HCRNG_SUCCESS;
}

static hcrngStatus Philox432CreateStream(hcrngPhilox432StreamCreator* creator, hcrngPhilox432Stream* buffer)
{
	//Check params
	if (buffer == NULL)
		return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): buffer cannot be NULL", __func__);

	// use default creator if not given
	if (creator == NULL)
		creator = &defaultStreamCreator_Philox432;

	// initialize stream
	buffer->current = buffer->initial = buffer->substream = creator->nextState;

	//Advance next state in stream creator
	creator->nextState.ctr = hcrngPhilox432Add(creator->nextState.ctr, creator->JumpDistance);

	return HCRNG_SUCCESS;
}

hcrngStatus hcrngPhilox432CreateOverStreams(hcrngPhilox432StreamCreator* creator, size_t count, hcrngPhilox432Stream* streams)
{
	// iterate over all individual stream buffers
	for (size_t i = 0; i < count; i++) {

		hcrngStatus err = Philox432CreateStream(creator, &streams[i]);

		// abort on error
		if (err != HCRNG_SUCCESS)
			return err;
	}

	return HCRNG_SUCCESS;
}

hcrngPhilox432Stream* hcrngPhilox432CreateStreams(hcrngPhilox432StreamCreator* creator, size_t count, size_t* bufSize, hcrngStatus* err)
{
	hcrngStatus err_;
	size_t bufSize_;
	hcrngPhilox432Stream* streams = hcrngPhilox432AllocStreams(count, &bufSize_, &err_);

	if (err_ == HCRNG_SUCCESS)
		err_ = hcrngPhilox432CreateOverStreams(creator, count, streams);

	if (bufSize != NULL)
		*bufSize = bufSize_;

	if (err != NULL)
		*err = err_;

	return streams;
}

hcrngPhilox432Stream* hcrngPhilox432CopyStreams(size_t count, const hcrngPhilox432Stream* streams, hcrngStatus* err)
{
	hcrngStatus err_ = HCRNG_SUCCESS;
	hcrngPhilox432Stream* dest = NULL;

	//Check params
	if (streams == NULL)
		err_ = hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): stream cannot be NULL", __func__);

	if (err_ == HCRNG_SUCCESS)
		dest = hcrngPhilox432AllocStreams(count, NULL, &err_);

	if (err_ == HCRNG_SUCCESS)
		err_ = hcrngPhilox432CopyOverStreams(count, dest, streams);

	if (err != NULL)
		*err = err_;

	return dest;
}

hcrngPhilox432Stream* hcrngPhilox432MakeSubstreams(hcrngPhilox432Stream* stream, size_t count, size_t* bufSize, hcrngStatus* err)
{
	hcrngStatus err_;
	size_t bufSize_;
	hcrngPhilox432Stream* substreams = hcrngPhilox432AllocStreams(count, &bufSize_, &err_);

	if (err_ == HCRNG_SUCCESS)
		err_ = hcrngPhilox432MakeOverSubstreams(stream, count, substreams);

	if (bufSize != NULL)
		*bufSize = bufSize_;

	if (err != NULL)
		*err = err_;

	return substreams;
}

hcrngStatus hcrngPhilox432WriteStreamInfo(const hcrngPhilox432Stream* stream, FILE *file)
{
	//Check params
	if (stream == NULL)
		return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): stream cannot be NULL", __func__);
	if (file == NULL)
		return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): file cannot be NULL", __func__);

	//The Initial state of the Stream
	//fprintf(file, "initial : (ctr, index)=( %u %u %u %u , [%u]) [deck = { %u %u %u %u}] \n\n",
	//	stream->initial.ctr.H.msb, stream->initial.ctr.H.lsb, stream->initial.ctr.L.msb, stream->initial.ctr.L.lsb, stream->initial.deckIndex,
	//	stream->initial.deck[3], stream->initial.deck[2], stream->initial.deck[1], stream->initial.deck[0]);

	//The Current state of the Stream
	fprintf(file, "Current : (ctr, index)=( %u %u %u %u , [%u])  [deck = { %u %u %u %u}] \n\n",
		stream->current.ctr.H.msb, stream->current.ctr.H.lsb, stream->current.ctr.L.msb, stream->current.ctr.L.lsb, stream->current.deckIndex
		, stream->current.deck[3], stream->current.deck[2], stream->current.deck[1], stream->current.deck[0]
		);


	return HCRNG_SUCCESS;
}

//hcrngStatus hcrngPhilox432AdvanceStreams(size_t count, hcrngPhilox432Stream* streams, int e, int c)
//{
//
//
//	hcrngPhilox432Counter Steps = { { 0, 0 }, { 0, 0 } };
//
//	//Calculate the Nbr of steps in 128bit counter
//	unsigned char slotId = abs(e) / 32;
//	unsigned char remider = abs(e) % 32;
//
//	if (e != 0)
//	{
//		unsigned int value = (1 << remider) + (e > 0 ? 1 : -1)*c;
//
//		if (slotId == 0)
//			Steps.L.lsb = value;
//		else if (slotId == 1)
//			Steps.L.msb = value;
//		else if (slotId == 2)
//			Steps.H.lsb = value;
//		else
//			Steps.H.msb = value;
//	}
//	else
//		Steps.L.lsb = c; 
//
//	//Update the stream counter
//	for (size_t i = 0; i < count; i++)
//	{
//		if (e >= 0)
//			streams[i].current.ctr = hcrngPhilox432Add(streams[i].current.ctr, Steps);
//		else streams[i].current.ctr = hcrngPhilox432Substract(streams[i].current.ctr, Steps);
//	}
//
//	return HCRNG_SUCCESS;
//}

void hcrngPhilox432AdvanceStream_(hcrngPhilox432Stream* stream, int e, int c)
{
	unsigned char slotId = 0, remider = 0;
	int slide = 0, push = 0, pull = 0, c2 = 0;
	hcrngPhilox432Counter Steps = { { 0, 0 }, { 0, 0 } };

	if (e >= 0)
	{
		//Slide the counter
		if (e < 2) {
			if (e == 1)	c += 2;
		}
		else{
			int e2 = e - 2;

			slotId = e2 / 32;
			remider = e2 % 32;
			slide = 1 << remider;
		}

		//Push/Pull the counter
		c2 = stream->current.deckIndex + c;
		if (c >= 0){
			push = c2 / 4;
		}
		else{
			if (c2 < 0){
				pull = (c2 / 4) - 1;
				if ((c2 % 4) == 0) pull += 1;
			}
		}

		//Advance by counter value
		unsigned int ctr_value = abs(slide + push + pull);

		if (slotId == 0)
			Steps.L.lsb = ctr_value;
		else if (slotId == 1)
			Steps.L.msb = ctr_value;
		else if (slotId == 2)
			Steps.H.lsb = ctr_value;
		else
			Steps.H.msb = ctr_value;

		if ((slide + push) > abs(pull))
			stream->current.ctr = hcrngPhilox432Add(stream->current.ctr, Steps);
		else stream->current.ctr = hcrngPhilox432Substract(stream->current.ctr, Steps);

		//Adjusting the DeckIndex
		if (c >= 0)	stream->current.deckIndex = c2 % 4;
		else{
			if (c2 < 0){
				if ((abs(c2) % 4) == 0) stream->current.deckIndex = 0;
				else stream->current.deckIndex = 4 - (abs(c2) % 4);
			}
			else
				stream->current.deckIndex = c2;
		}
	}
	//negative e
	else{
		//Slide the counter
		if (e > -2) {
			if (e == -1) c -= 2;
		}
		else{
			int e2 = abs(e) - 2;

			slotId = e2 / 32;
			remider = e2 % 32;
			slide = 1 << remider;
		}

		//Push/Pull the counter
		c2 = stream->current.deckIndex + c;

		if (c < 0){
			if (c2 < 0)	{
				push = -(c2 / 4) + 1;
				if ((c2 % 4) == 0) push -= 1;
			}
		}
		else
		{
			pull = c2 / 4;
		}

		//Advance by counter value
		unsigned int ctr_value = abs(slide + push - pull);

		if (slotId == 0)
			Steps.L.lsb = ctr_value;
		else if (slotId == 1)
			Steps.L.msb = ctr_value;
		else if (slotId == 2)
			Steps.H.lsb = ctr_value;
		else
			Steps.H.msb = ctr_value;

		if ((slide + push) > abs(pull))
			stream->current.ctr = hcrngPhilox432Substract(stream->current.ctr, Steps);
		else stream->current.ctr = hcrngPhilox432Add(stream->current.ctr, Steps);

		//Adjusting the DeckIndex
		if (c > 0) stream->current.deckIndex = abs(c2) % 4;
		else
		{
			if (c2 < 0)
			{
				if ((abs(c2) % 4) == 0) stream->current.deckIndex = 0;
				else stream->current.deckIndex = 4 - (abs(c2) % 4);
			}
			else
				stream->current.deckIndex = c2;
		}

	}
	
	hcrngPhilox432GenerateDeck(&stream->current);
}

hcrngStatus hcrngPhilox432AdvanceStreams(size_t count, hcrngPhilox432Stream* streams, int e, int c)
{
	//Check params
	if (streams == NULL)
		return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): streams cannot be NULL", __func__);
	if (e > 127)
		return hcrngSetErrorString(HCRNG_INVALID_VALUE, "%s(): 'e' can not exceed 127", __func__);

	//Advance streams
	for (size_t i = 0; i < count; i++)
	{
		hcrngPhilox432AdvanceStream_(&streams[i], e, c);
	}

	return HCRNG_SUCCESS;
}

hcrngStatus hcrngPhilox432DeviceRandomU01Array_single(size_t streamCount, hcrngPhilox432Stream* streams,
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
           if(gid < (streamCount/streams_per_thread)) {
           for(int i =0; i < numberCount/streamCount; i++) {
              if ((i > 0) && (streamlength > 0) && (i % streamlength == 0)) {
               hcrngPhilox432ForwardToNextSubstreams(streams_per_thread, &streams[streams_per_thread * gid]);
              }
              if ((i > 0) && (streamlength < 0) && (i % streamlength == 0)) {
               hcrngPhilox432RewindSubstreams(streams_per_thread, &streams[streams_per_thread * gid]);
              }
              for (int j = 0; j < streams_per_thread; j++)
               outBuffer[streams_per_thread * (i * (streamCount/streams_per_thread) + gid) + j] = hcrngPhilox432RandomU01(&streams[streams_per_thread * gid + j]);
              }
           }
        }).wait();
#undef HCRNG_SINGLE_PRECISION
        return status;
}

hcrngStatus hcrngPhilox432DeviceRandomNArray_single(size_t streamCount, hcrngPhilox432Stream *streams,
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
        hcrngStatus status = hcrngPhilox432DeviceRandomU01Array_single(streamCount, streams,numberCount, outBuffer, streamlength, streams_per_thread);
        if (status == HCRNG_SUCCESS){
	    	status = box_muller_transform_single(accl_view, mu, sigma, outBuffer, numberCount);
                return status;
                }
#undef HCRNG_SINGLE_PRECISION
        return status;
}


hcrngStatus hcrngPhilox432DeviceRandomU01Array_double(size_t streamCount, hcrngPhilox432Stream* streams,
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
           if(gid < (streamCount/streams_per_thread)) {
           for(int i =0; i < numberCount/streamCount; i++) {
              if ((i > 0) && (streamlength > 0) && (i % streamlength == 0)) {
               hcrngPhilox432ForwardToNextSubstreams(streams_per_thread, &streams[streams_per_thread * gid]);
              }
              if ((i > 0) && (streamlength < 0) && (i % streamlength == 0)) {
               hcrngPhilox432RewindSubstreams(streams_per_thread, &streams[streams_per_thread * gid]);
              }
              for (int j = 0; j < streams_per_thread; j++)
               outBuffer[streams_per_thread * (i * (streamCount/streams_per_thread) + gid) + j] = hcrngPhilox432RandomU01(&streams[streams_per_thread * gid + j]);
              }
           }
        }).wait();
        return status;
}

hcrngStatus hcrngPhilox432DeviceRandomNArray_double(size_t streamCount, hcrngPhilox432Stream *streams,
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
        hcrngStatus status = hcrngPhilox432DeviceRandomU01Array_double(streamCount, streams,numberCount, outBuffer, streamlength, streams_per_thread);
        if (status == HCRNG_SUCCESS){
	    	status = box_muller_transform_double(accl_view, mu, sigma, outBuffer, numberCount);
                return status;
                }
        return status;
}
