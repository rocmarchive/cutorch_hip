/*  @file Philox432.h
*  @brief Specific interface for the Philox432 generator
*/

#pragma once
#ifndef PHILOX432_H
#define PHILOX432_H
#include "hcRNG.h"
#include <stdio.h>


#ifdef __cplusplus 
//extern "C" {
#endif

/*  @brief State type of a Philox432 stream
*  The state is a seed consisting of a 128bits counter
*  @see hcrngStreamState
*/

typedef struct hcrngPhilox432SB_ {
	unsigned int msb, lsb;   //most significant bits, and the least significant bits
}hcrngPhilox432SB;

typedef struct hcrngPhilox432Counter_ {
	hcrngPhilox432SB H, L;
} hcrngPhilox432Counter;


typedef struct {
	hcrngPhilox432Counter  ctr;  // 128 bits counter
	unsigned int  deck[4];            // this table hold the 4x32 generated uint from philox4x32(ctr,kry) function
	unsigned int deckIndex;           //the index of actual pregenerated integer to give to the user
} hcrngPhilox432StreamState;


struct hcrngPhilox432Stream_ {
	union {
		struct {
			hcrngPhilox432StreamState states[3];
		};
		struct {
			hcrngPhilox432StreamState current;
			hcrngPhilox432StreamState initial;
			hcrngPhilox432StreamState substream;
		};
	};
};

/*! @copybrief hcrngStream
*  @see hcrngStream
*/
typedef struct hcrngPhilox432Stream_ hcrngPhilox432Stream;

//struct hcrngPhilox432StreamCreator_;
/*! @copybrief hcrngStreamCreator
*  @see hcrngStreamCreator
*/

struct hcrngPhilox432StreamCreator_ {
        hcrngPhilox432StreamState initialState;
        hcrngPhilox432StreamState nextState;
        hcrngPhilox432Counter JumpDistance;
};


typedef struct hcrngPhilox432StreamCreator_ hcrngPhilox432StreamCreator;


/*! @brief Default initial seed of the first stream
*/

#define BASE_CREATOR_STATE { \
        {{ 0, 0},{ 0, 1}}, \
        { 0, 0, 0, 0 }, \
        0 }
/*! @brief Jump Struc for \f$2^{100}\f$ steps forward
*/
#define BASE_CREATOR_JUMP_DISTANCE {{ 16, 0},{ 0, 0 }}

/*! @brief Default stream creator (defaults to \f$2^{100}\f$ steps forward)
*
*  Contains the default seed;
*  adjacent streams are spaced nu steps apart.
*  The default is \f$nu = 2^{100}\f$.
*  The default seed is \f$({{0,0},{0,0}})\f$.
*/
static  hcrngPhilox432StreamCreator defaultStreamCreator_Philox432 = { BASE_CREATOR_STATE, BASE_CREATOR_STATE, BASE_CREATOR_JUMP_DISTANCE };


	/*! @copybrief hcrngCopyStreamCreator()
	*  @see hcrngCopyStreamCreator()
	*/
	HCRNGAPI hcrngPhilox432StreamCreator* hcrngPhilox432CopyStreamCreator(const hcrngPhilox432StreamCreator* creator, hcrngStatus* err);

	/*! @copybrief hcrngDestroyStreamCreator()
	*  @see hcrngDestroyStreamCreator()
	*/
	HCRNGAPI hcrngStatus hcrngPhilox432DestroyStreamCreator(hcrngPhilox432StreamCreator* creator);

	/*! @copybrief hcrngRewindStreamCreator()
	 *  @see hcrngRewindStreamCreator()
	 */
	HCRNGAPI hcrngStatus hcrngPhilox432RewindStreamCreator(hcrngPhilox432StreamCreator* creator);

	/*! @copybrief hcrngSetBaseCreatorState()
	*  @see hcrngSetBaseCreatorState()
	*/
	HCRNGAPI hcrngStatus hcrngPhilox432SetBaseCreatorState(hcrngPhilox432StreamCreator* creator, const hcrngPhilox432StreamState* baseState);

	/*! @copybrief hcrngChangeStreamsSpacing()
	*  @see hcrngChangeStreamsSpacing()
	*/
	HCRNGAPI hcrngStatus hcrngPhilox432ChangeStreamsSpacing(hcrngPhilox432StreamCreator* creator, int e, int c);

	/*! @copybrief hcrngAllocStreams()
	*  @see hcrngAllocStreams()
	*/
	HCRNGAPI hcrngPhilox432Stream* hcrngPhilox432AllocStreams(size_t count, size_t* bufSize, hcrngStatus* err);

	/*! @copybrief hcrngDestroyStreams()
	*  @see hcrngDestroyStreams()
	*/
	HCRNGAPI hcrngStatus hcrngPhilox432DestroyStreams(hcrngPhilox432Stream* streams);

	/*! @copybrief hcrngCreateOverStreams()
	*  @see hcrngCreateOverStreams()
	*/
	HCRNGAPI hcrngStatus hcrngPhilox432CreateOverStreams(hcrngPhilox432StreamCreator* creator, size_t count, hcrngPhilox432Stream* streams);

	/*! @copybrief hcrngCreateStreams()
	*  @see hcrngCreateStreams()
	*/
	HCRNGAPI hcrngPhilox432Stream* hcrngPhilox432CreateStreams(hcrngPhilox432StreamCreator* creator, size_t count, size_t* bufSize, hcrngStatus* err);

	/*! @copybrief hcrngCopyOverStreams()
	*  @see hcrngCopyOverStreams()
	*/
	HCRNGAPI hcrngStatus hcrngPhilox432CopyOverStreams(size_t count, hcrngPhilox432Stream* destStreams, const hcrngPhilox432Stream* srcStreams);

	/*! @copybrief hcrngCopyStreams()
	*  @see hcrngCopyStreams()
	*/
	HCRNGAPI hcrngPhilox432Stream* hcrngPhilox432CopyStreams(size_t count, const hcrngPhilox432Stream* streams, hcrngStatus* err);

#define hcrngPhilox432RandomU01          _HCRNG_TAG_FPTYPE(hcrngPhilox432RandomU01)
#define hcrngPhilox432RandomN            _HCRNG_TAG_FPTYPE(hcrngPhilox432RandomN)
#define hcrngPhilox432RandomInteger      _HCRNG_TAG_FPTYPE(hcrngPhilox432RandomInteger)
#define hcrngPhilox432RandomUnsignedInteger      _HCRNG_TAG_FPTYPE(hcrngPhilox432RandomUnsignedInteger)
#define hcrngPhilox432RandomU01Array     _HCRNG_TAG_FPTYPE(hcrngPhilox432RandomU01Array)
#define hcrngPhilox432RandomIntegerArray _HCRNG_TAG_FPTYPE(hcrngPhilox432RandomIntegerArray)
#define hcrngPhilox432RandomUnsignedIntegerArray _HCRNG_TAG_FPTYPE(hcrngPhilox432RandomUnsignedIntegerArray)


	/*! @copybrief hcrngRandomU01()
	*  @see hcrngRandomU01()
	*/
	HCRNGAPI _HCRNG_FPTYPE hcrngPhilox432RandomU01(hcrngPhilox432Stream* stream);
	HCRNGAPI float  hcrngPhilox432RandomU01_float (hcrngPhilox432Stream* stream);
	HCRNGAPI double hcrngPhilox432RandomU01_double(hcrngPhilox432Stream* stream);

        
	HCRNGAPI _HCRNG_FPTYPE hcrngPhilox432RandomN(hcrngPhilox432Stream* stream1, hcrngPhilox432Stream* stream2, _HCRNG_FPTYPE mu, _HCRNG_FPTYPE sigma);
	HCRNGAPI float  hcrngPhilox432RandomN_float (hcrngPhilox432Stream* stream, hcrngPhilox432Stream* stream2, float mu, float sigma);
	HCRNGAPI double hcrngPhilox432RandomN_double(hcrngPhilox432Stream* stream, hcrngPhilox432Stream* stream2, double mu, double sigma);

	/*! @copybrief hcrngRandomInteger()
	*  @see hcrngRandomInteger()
	*/
	HCRNGAPI int hcrngPhilox432RandomInteger(hcrngPhilox432Stream* stream, int i, int j);
	HCRNGAPI int hcrngPhilox432RandomInteger_float (hcrngPhilox432Stream* stream, int i, int j);
	HCRNGAPI int hcrngPhilox432RandomInteger_double(hcrngPhilox432Stream* stream, int i, int j);

        HCRNGAPI unsigned int hcrngPhilox432RandomUnsignedInteger(hcrngPhilox432Stream* stream, unsigned int i, unsigned int j);
        HCRNGAPI unsigned int hcrngPhilox432RandomUnsignedInteger_float (hcrngPhilox432Stream* stream, unsigned int i, unsigned int j);
        HCRNGAPI unsigned int hcrngPhilox432RandomUnsignedInteger_double(hcrngPhilox432Stream* stream, unsigned int i, unsigned int j);



	/*! @copybrief hcrngRandomU01Array()
	*  @see hcrngRandomU01Array()
	*/
	HCRNGAPI hcrngStatus hcrngPhilox432RandomU01Array(hcrngPhilox432Stream* stream, size_t count, _HCRNG_FPTYPE* buffer);
	HCRNGAPI hcrngStatus hcrngPhilox432RandomU01Array_float (hcrngPhilox432Stream* stream, size_t count, float * buffer);
	HCRNGAPI hcrngStatus hcrngPhilox432RandomU01Array_double(hcrngPhilox432Stream* stream, size_t count, double* buffer);

	/*! @copybrief hcrngRandomIntegerArray()
	*  @see hcrngRandomIntegerArray()
	*/
	HCRNGAPI hcrngStatus hcrngPhilox432RandomIntegerArray(hcrngPhilox432Stream* stream, int i, int j, size_t count, int* buffer);
	HCRNGAPI hcrngStatus hcrngPhilox432RandomIntegerArray_float (hcrngPhilox432Stream* stream, int i, int j, size_t count, int* buffer);
	HCRNGAPI hcrngStatus hcrngPhilox432RandomIntegerArray_double(hcrngPhilox432Stream* stream, int i, int j, size_t count, int* buffer);


        HCRNGAPI hcrngStatus hcrngPhilox432RandomUnsignedIntegerArray(hcrngPhilox432Stream* stream, unsigned int i, unsigned int j, size_t count, unsigned int* buffer);
        HCRNGAPI hcrngStatus hcrngPhilox432RandomUnsignedIntegerArray_float (hcrngPhilox432Stream* stream, unsigned int i, unsigned int j, size_t count, unsigned int* buffer);
        HCRNGAPI hcrngStatus hcrngPhilox432RandomUnsignedIntegerArray_double(hcrngPhilox432Stream* stream, unsigned int i, unsigned int j, size_t count, unsigned int* buffer);

	/*! @copybrief hcrngRewindStreams()
	*  @see hcrngRewindStreams()
	*/
	HCRNGAPI hcrngStatus hcrngPhilox432RewindStreams(size_t count, hcrngPhilox432Stream* streams);

	/*! @copybrief hcrngRewindSubstreams()
	*  @see hcrngRewindSubstreams()
	*/
	HCRNGAPI hcrngStatus hcrngPhilox432RewindSubstreams(size_t count, hcrngPhilox432Stream* streams);

	/*! @copybrief hcrngForwardToNextSubstreams()
	*  @see hcrngForwardToNextSubstreams()
	*/
	HCRNGAPI hcrngStatus hcrngPhilox432ForwardToNextSubstreams(size_t count, hcrngPhilox432Stream* streams);

	/*! @copybrief hcrngMakeSubstreams()
	*  @see hcrngMakeSubstreams()
	*/
	HCRNGAPI hcrngPhilox432Stream* hcrngPhilox432MakeSubstreams(hcrngPhilox432Stream* stream, size_t count, size_t* bufSize, hcrngStatus* err);

	/*! @copybrief hcrngMakeOverSubstreams()
	*  @see hcrngMakeOverSubstreams()
	*/
	HCRNGAPI hcrngStatus hcrngPhilox432MakeOverSubstreams(hcrngPhilox432Stream* stream, size_t count, hcrngPhilox432Stream* substreams);

	/*! @copybrief hcrngAdvanceStreams()
	*  @see hcrngAdvanceStreams()
	*/
	HCRNGAPI hcrngStatus hcrngPhilox432AdvanceStreams(size_t count, hcrngPhilox432Stream* streams, int e, int c);

	/*! @copybrief hcrngDeviceRandomU01Array()
	*  @see hcrngDeviceRandomU01Array()
	*/
#ifdef HCRNG_SINGLE_PRECISION
#define hcrngPhilox432DeviceRandomU01Array(...) hcrngPhilox432DeviceRandomU01Array_(__VA_ARGS__, HC_TRUE)
#else
#define hcrngPhilox432DeviceRandomU01Array(...) hcrngPhilox432DeviceRandomU01Array_(__VA_ARGS__, HC_FALSE)
#endif

	/** \internal
	 *  @brief Helper function for hcrngPhilox432DeviceRandomU01Array()
	 */
	HCRNGAPI hcrngStatus hcrngPhilox432DeviceRandomU01Array_single(size_t streamCount, hcrngPhilox432Stream* streams,
		size_t numberCount, float* outBuffer, int streamlength = 0, size_t streams_per_thread = 1);
        HCRNGAPI hcrngStatus hcrngPhilox432DeviceRandomU01Array_double(size_t streamCount, hcrngPhilox432Stream* streams,
                size_t numberCount, double* outBuffer, int streamlength = 0, size_t streams_per_thread = 1);
               
//Normal distribution             
        HCRNGAPI hcrngStatus hcrngPhilox432DeviceRandomNArray_single(size_t streamCount, hcrngPhilox432Stream *streams,
	        size_t numberCount, float mu, float sigma, float *outBuffer, int streamlength = 0, size_t streams_per_thread = 1);
        HCRNGAPI hcrngStatus hcrngPhilox432DeviceRandomNArray_double(size_t streamCount, hcrngPhilox432Stream *streams,
	        size_t numberCount, double mu, double sigma, double *outBuffer, int streamlength = 0, size_t streams_per_thread = 1);
	/*! @copybrief hcrngWriteStreamInfo()
	*  @see hcrngWriteStreamInfo()
	*/
	HCRNGAPI hcrngStatus hcrngPhilox432WriteStreamInfo(const hcrngPhilox432Stream* stream, FILE *file);

#ifdef __cplusplus 
//}
#endif // __cplusplus
#endif // PHILOX432_H
