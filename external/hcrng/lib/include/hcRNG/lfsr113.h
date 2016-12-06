/*  @file Lfsr113.h
*  @brief Specific interface for the Lfsr113 generator
*/

#pragma once
#ifndef LFSR113_H
#define LFSR113_H
#include "hcRNG.h"
#include <stdio.h>

#ifdef __cplusplus
//extern "C" {
#endif 

/*  @brief State type of a Lfsr113 stream
*  The state is a seed consisting of six unsigned 32-bit integers.
*  @see hcrngStreamState
*/
typedef struct {
	/*! @brief Seed for the first LFSR component
	*/
	unsigned int g[4];
} hcrngLfsr113StreamState;


struct hcrngLfsr113Stream_ {
	union {
		struct {
			hcrngLfsr113StreamState states[3];
		};
		struct {
			hcrngLfsr113StreamState current;
			hcrngLfsr113StreamState initial;
			hcrngLfsr113StreamState substream;
		};
	};
};

/*! @copybrief hcrngStream
*  @see hcrngStream
*/
typedef struct hcrngLfsr113Stream_ hcrngLfsr113Stream;

//struct hcrngLfsr113StreamCreator_;
/*! @copybrief hcrngStreamCreator
*  @see hcrngStreamCreator
*/

struct hcrngLfsr113StreamCreator_ {
        hcrngLfsr113StreamState initialState;
        hcrngLfsr113StreamState nextState;
};


typedef struct hcrngLfsr113StreamCreator_ hcrngLfsr113StreamCreator;

/*! @brief Default initial seed of the first stream
*/
#define BASE_CREATOR_STATE_LFSR113 { 987654321, 987654321, 987654321, 987654321 }


/*! @brief Default stream creator (defaults to \f$2^{134}\f$ steps forward)
*
*  Contains the default seed and the transition matrices to jump \f$\nu\f$ steps forward;
*  adjacent streams are spaced nu steps apart.
*  The default is \f$nu = 2^{134}\f$.
*  The default seed is \f$(12345,12345,12345,12345,12345,12345)\f$.
*/
static hcrngLfsr113StreamCreator defaultStreamCreator_Lfsr113 = {
        { BASE_CREATOR_STATE_LFSR113 },
        { BASE_CREATOR_STATE_LFSR113 }
};



	/*! @copybrief hcrngCopyStreamCreator()
	*  @see hcrngCopyStreamCreator()
	*/
	HCRNGAPI hcrngLfsr113StreamCreator* hcrngLfsr113CopyStreamCreator(const hcrngLfsr113StreamCreator* creator, hcrngStatus* err);

	/*! @copybrief hcrngDestroyStreamCreator()
	*  @see hcrngDestroyStreamCreator()
	*/
	HCRNGAPI hcrngStatus hcrngLfsr113DestroyStreamCreator(hcrngLfsr113StreamCreator* creator);

	/*! @copybrief hcrngRewindStreamCreator()
	 *  @see hcrngRewindStreamCreator()
	 */
	HCRNGAPI hcrngStatus hcrngLfsr113RewindStreamCreator(hcrngLfsr113StreamCreator* creator);

	/*! @copybrief hcrngSetBaseCreatorState()
	*  @see hcrngSetBaseCreatorState()
	*/
	HCRNGAPI hcrngStatus hcrngLfsr113SetBaseCreatorState(hcrngLfsr113StreamCreator* creator, const hcrngLfsr113StreamState* baseState);

	/*! @copybrief hcrngChangeStreamsSpacing()
	*  @see hcrngChangeStreamsSpacing()
	*/
	HCRNGAPI hcrngStatus hcrngLfsr113ChangeStreamsSpacing(hcrngLfsr113StreamCreator* creator, int e, int c);

	/*! @copybrief hcrngAllocStreams()
	*  @see hcrngAllocStreams()
	*/
	HCRNGAPI hcrngLfsr113Stream* hcrngLfsr113AllocStreams(size_t count, size_t* bufSize, hcrngStatus* err);

	/*! @copybrief hcrngDestroyStreams()
	*  @see hcrngDestroyStreams()
	*/
	HCRNGAPI hcrngStatus hcrngLfsr113DestroyStreams(hcrngLfsr113Stream* streams);

	/*! @copybrief hcrngCreateOverStreams()
	*  @see hcrngCreateOverStreams()
	*/
	HCRNGAPI hcrngStatus hcrngLfsr113CreateOverStreams(hcrngLfsr113StreamCreator* creator, size_t count, hcrngLfsr113Stream* streams);

	/*! @copybrief hcrngCreateStreams()
	*  @see hcrngCreateStreams()
	*/
	HCRNGAPI hcrngLfsr113Stream* hcrngLfsr113CreateStreams(hcrngLfsr113StreamCreator* creator, size_t count, size_t* bufSize, hcrngStatus* err);

	/*! @copybrief hcrngCopyOverStreams()
	*  @see hcrngCopyOverStreams()
	*/
	HCRNGAPI hcrngStatus hcrngLfsr113CopyOverStreams(size_t count, hcrngLfsr113Stream* destStreams, const hcrngLfsr113Stream* srcStreams);

	/*! @copybrief hcrngCopyStreams()
	*  @see hcrngCopyStreams()
	*/
	HCRNGAPI hcrngLfsr113Stream* hcrngLfsr113CopyStreams(size_t count, const hcrngLfsr113Stream* streams, hcrngStatus* err);

#define hcrngLfsr113RandomU01          _HCRNG_TAG_FPTYPE(hcrngLfsr113RandomU01)
#define hcrngLfsr113RandomN            _HCRNG_TAG_FPTYPE(hcrngLfsr113RandomN)
#define hcrngLfsr113RandomInteger      _HCRNG_TAG_FPTYPE(hcrngLfsr113RandomInteger)
#define hcrngLfsr113RandomUnsignedInteger      _HCRNG_TAG_FPTYPE(hcrngLfsr113RandomUnsignedInteger)
#define hcrngLfsr113RandomU01Array     _HCRNG_TAG_FPTYPE(hcrngLfsr113RandomU01Array)
#define hcrngLfsr113RandomIntegerArray _HCRNG_TAG_FPTYPE(hcrngLfsr113RandomIntegerArray)
#define hcrngLfsr113RandomUnsignedIntegerArray _HCRNG_TAG_FPTYPE(hcrngLfsr113RandomUnsignedIntegerArray)

	/*! @copybrief hcrngRandomU01()
	*  @see hcrngRandomU01()
	*/
	HCRNGAPI _HCRNG_FPTYPE hcrngLfsr113RandomU01(hcrngLfsr113Stream* stream);
	HCRNGAPI float  hcrngLfsr113RandomU01_float (hcrngLfsr113Stream* stream);
	HCRNGAPI double hcrngLfsr113RandomU01_double(hcrngLfsr113Stream* stream);

 // Normal distribution
        
	HCRNGAPI _HCRNG_FPTYPE hcrngLfsr113RandomN(hcrngLfsr113Stream* stream1, hcrngLfsr113Stream* stream2, _HCRNG_FPTYPE mu, _HCRNG_FPTYPE sigma);
	HCRNGAPI float  hcrngLfsr113RandomN_float (hcrngLfsr113Stream* stream, hcrngLfsr113Stream* stream2, float mu, float sigma);
	HCRNGAPI double hcrngLfsr113RandomN_double(hcrngLfsr113Stream* stream, hcrngLfsr113Stream* stream2, double mu, double sigma);

	/*! @copybrief hcrngRandomInteger()
	*  @see hcrngRandomInteger()
	*/
	HCRNGAPI int hcrngLfsr113RandomInteger(hcrngLfsr113Stream* stream, int i, int j);
	HCRNGAPI int hcrngLfsr113RandomInteger_float (hcrngLfsr113Stream* stream, int i, int j);
	HCRNGAPI int hcrngLfsr113RandomInteger_double(hcrngLfsr113Stream* stream, int i, int j);

        HCRNGAPI unsigned int hcrngLfsr113RandomUnsignedInteger(hcrngLfsr113Stream* stream, unsigned int i, unsigned int j);
        HCRNGAPI unsigned int hcrngLfsr113RandomUnsignedInteger_float (hcrngLfsr113Stream* stream, unsigned int i, unsigned int j);
        HCRNGAPI unsigned int hcrngLfsr113RandomUnsignedInteger_double(hcrngLfsr113Stream* stream, unsigned int i, unsigned int j);



	/*! @copybrief hcrngRandomU01Array()
	*  @see hcrngRandomU01Array()
	*/
	HCRNGAPI hcrngStatus hcrngLfsr113RandomU01Array(hcrngLfsr113Stream* stream, size_t count, _HCRNG_FPTYPE* buffer);
	HCRNGAPI hcrngStatus hcrngLfsr113RandomU01Array_float (hcrngLfsr113Stream* stream, size_t count, float * buffer);
	HCRNGAPI hcrngStatus hcrngLfsr113RandomU01Array_double(hcrngLfsr113Stream* stream, size_t count, double* buffer);

	/*! @copybrief hcrngRandomIntegerArray()
	*  @see hcrngRandomIntegerArray()
	*/
	HCRNGAPI hcrngStatus hcrngLfsr113RandomIntegerArray(hcrngLfsr113Stream* stream, int i, int j, size_t count, int* buffer);
	HCRNGAPI hcrngStatus hcrngLfsr113RandomIntegerArray_float (hcrngLfsr113Stream* stream, int i, int j, size_t count, int* buffer);
	HCRNGAPI hcrngStatus hcrngLfsr113RandomIntegerArray_double(hcrngLfsr113Stream* stream, int i, int j, size_t count, int* buffer);


        HCRNGAPI hcrngStatus hcrngLfsr113RandomUnsignedIntegerArray(hcrngLfsr113Stream* stream, unsigned int i, unsigned int j, size_t count, unsigned int* buffer);
        HCRNGAPI hcrngStatus hcrngLfsr113RandomUnsignedIntegerArray_float (hcrngLfsr113Stream* stream, unsigned int i, unsigned int j, size_t count, unsigned int* buffer);
        HCRNGAPI hcrngStatus hcrngLfsr113RandomUnsignedIntegerArray_double(hcrngLfsr113Stream* stream, unsigned int i, unsigned int j, size_t count, unsigned int* buffer);



	/*! @copybrief hcrngRewindStreams()
	*  @see hcrngRewindStreams()
	*/
	HCRNGAPI hcrngStatus hcrngLfsr113RewindStreams(size_t count, hcrngLfsr113Stream* streams);

	/*! @copybrief hcrngRewindSubstreams()
	*  @see hcrngRewindSubstreams()
	*/
	HCRNGAPI hcrngStatus hcrngLfsr113RewindSubstreams(size_t count, hcrngLfsr113Stream* streams);

	/*! @copybrief hcrngForwardToNextSubstreams()
	*  @see hcrngForwardToNextSubstreams()
	*/
	HCRNGAPI hcrngStatus hcrngLfsr113ForwardToNextSubstreams(size_t count, hcrngLfsr113Stream* streams);

	/*! @copybrief hcrngMakeSubstreams()
	 *  @see hcrngMakeSubstreams()
	 */
	HCRNGAPI hcrngLfsr113Stream* hcrngLfsr113MakeSubstreams(hcrngLfsr113Stream* stream, size_t count, size_t* bufSize, hcrngStatus* err);

	/*! @copybrief hcrngMakeOverSubstreams()
	 *  @see hcrngMakeOverSubstreams()
	 */
	HCRNGAPI hcrngStatus hcrngLfsr113MakeOverSubstreams(hcrngLfsr113Stream* stream, size_t count, hcrngLfsr113Stream* substreams);

	/*! @copybrief hcrngAdvanceStreams()
	*  @see hcrngAdvanceStreams()
	*/
	HCRNGAPI hcrngStatus hcrngLfsr113AdvanceStreams(size_t count, hcrngLfsr113Stream* streams, int e, int c);

	/*! @copybrief hcrngDeviceRandomU01Array()
	*  @see hcrngDeviceRandomU01Array()
	*/
#ifdef HCRNG_SINGLE_PRECISION
#define hcrngLfsr113DeviceRandomU01Array(...) hcrngLfsr113DeviceRandomU01Array_(__VA_ARGS__, HC_TRUE)
#else
#define hcrngLfsr113DeviceRandomU01Array(...) hcrngLfsr113DeviceRandomU01Array_(__VA_ARGS__, HC_FALSE)
#endif

	/** \internal
	 *  @brief Helper function for hcrngLfsr113DeviceRandomU01Array()
	 */
	HCRNGAPI hcrngStatus hcrngLfsr113DeviceRandomU01Array_single(size_t streamCount, hcrngLfsr113Stream* streams,
		size_t numberCount, float* outBuffer, int streamlength = 0, size_t streams_per_thread = 1);
        HCRNGAPI hcrngStatus hcrngLfsr113DeviceRandomU01Array_double(size_t streamCount, hcrngLfsr113Stream* streams,
                size_t numberCount, double* outBuffer, int streamlength = 0, size_t streams_per_thread = 1);
/** \endinternal
 */
 
//Normal distribution
        HCRNGAPI hcrngStatus hcrngLfsr113DeviceRandomNArray_single(size_t streamCount, hcrngLfsr113Stream *streams,
	        size_t numberCount, float mu, float sigma, float *outBuffer, int streamlength = 0, size_t streams_per_thread = 1);
        HCRNGAPI hcrngStatus hcrngLfsr113DeviceRandomNArray_double(size_t streamCount, hcrngLfsr113Stream *streams,
	        size_t numberCount, double mu, double sigma, double *outBuffer, int streamlength = 0, size_t streams_per_thread = 1);	

	/*! @copybrief hcrngWriteStreamInfo()
	*  @see hcrngWriteStreamInfo()
	*/
	HCRNGAPI hcrngStatus hcrngLfsr113WriteStreamInfo(const hcrngLfsr113Stream* stream, FILE *file);

#ifdef __cplusplus
//}
#endif //__cplusplus
#endif // LFSR113_H
