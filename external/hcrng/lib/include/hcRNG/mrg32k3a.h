/*  @file Mrg32k3a.h
*  @brief Specific interface for the Mrg32k3a generator
*/

#pragma once
#ifndef MRG32K3A_H
#define MRG32K3A_H
#include "hcRNG.h"
#include <stdio.h>

#ifdef __cplusplus

//extern "C" {

#endif
/*  @brief State type of a Mrg32k3a stream
*  The state is a seed consisting of six unsigned 32-bit integers.
*  @see hcrngStreamState
*/
typedef struct {
	/*! @brief Seed for the first MRG component
	*/
	unsigned long g1[3];
	/*! @brief Seed for the second MRG component
	*/
	unsigned long g2[3];
} hcrngMrg32k3aStreamState;


struct hcrngMrg32k3aStream_ {
	union {
		struct {
			hcrngMrg32k3aStreamState states[3];
		};
		struct {
			hcrngMrg32k3aStreamState current;
			hcrngMrg32k3aStreamState initial;
			hcrngMrg32k3aStreamState substream;
		};
	};
};

/*! @copybrief hcrngStream
*  @see hcrngStream
*/
typedef struct hcrngMrg32k3aStream_ hcrngMrg32k3aStream;

//struct hcrngMrg32k3aStreamCreator_;
/*! @copybrief hcrngStreamCreator
*  @see hcrngStreamCreator
*/

struct hcrngMrg32k3aStreamCreator_ {
        hcrngMrg32k3aStreamState initialState;
        hcrngMrg32k3aStreamState nextState;
        /*! @brief Jump matrices for advancing the initial seed of streams
        */
        unsigned long nuA1[3][3];
        unsigned long nuA2[3][3];
};

/*! @brief Default initial seed of the first stream
*/
#define BASE_CREATOR_STATE_MRG32K3A { { 12345, 12345, 12345 }, { 12345, 12345, 12345 } }
/*! @brief Jump matrices for \f$2^{127}\f$ steps forward
*/
#define BASE_CREATOR_JUMP_MATRIX_1_MRG32K3A { \
        {2427906178, 3580155704, 949770784}, \
        { 226153695, 1230515664, 3580155704}, \
        {1988835001, 986791581, 1230515664} }
#define BASE_CREATOR_JUMP_MATRIX_2_MRG32K3A { \
        { 1464411153, 277697599, 1610723613}, \
        {32183930, 1464411153, 1022607788}, \
        {2824425944, 32183930, 2093834863} }

/*! @brief Default stream creator (defaults to \f$2^{134}\f$ steps forward)
*
*  Contains the default seed and the transition matrices to jump \f$\nu\f$ steps forward;
*  adjacent streams are spaced nu steps apart.
*  The default is \f$nu = 2^{134}\f$.
*  The default seed is \f$(12345,12345,12345,12345,12345,12345)\f$.
*/
typedef struct hcrngMrg32k3aStreamCreator_ hcrngMrg32k3aStreamCreator;

static hcrngMrg32k3aStreamCreator defaultStreamCreator_Mrg32k3a = {
        BASE_CREATOR_STATE_MRG32K3A,
        BASE_CREATOR_STATE_MRG32K3A,
        BASE_CREATOR_JUMP_MATRIX_1_MRG32K3A,
        BASE_CREATOR_JUMP_MATRIX_2_MRG32K3A
};
	/*! @copybrief hcrngCopyStreamCreator()
	*  @see hcrngCopyStreamCreator()
	*/
	HCRNGAPI hcrngMrg32k3aStreamCreator* hcrngMrg32k3aCopyStreamCreator(const hcrngMrg32k3aStreamCreator* creator, hcrngStatus* err);

	/*! @copybrief hcrngDestroyStreamCreator()
	*  @see hcrngDestroyStreamCreator()
	*/
	HCRNGAPI hcrngStatus hcrngMrg32k3aDestroyStreamCreator(hcrngMrg32k3aStreamCreator* creator);

	/*! @copybrief hcrngRewindStreamCreator()
	 *  @see hcrngRewindStreamCreator()
	 */
	HCRNGAPI hcrngStatus hcrngMrg32k3aRewindStreamCreator(hcrngMrg32k3aStreamCreator* creator);

	/*! @copybrief hcrngSetBaseCreatorState()
	*  @see hcrngSetBaseCreatorState()
	*/
	HCRNGAPI hcrngStatus hcrngMrg32k3aSetBaseCreatorState(hcrngMrg32k3aStreamCreator* creator, const hcrngMrg32k3aStreamState* baseState);

	/*! @copybrief hcrngChangeStreamsSpacing()
	*  @see hcrngChangeStreamsSpacing()
	*/
	HCRNGAPI hcrngStatus hcrngMrg32k3aChangeStreamsSpacing(hcrngMrg32k3aStreamCreator* creator, int e, int c);

	/*! @copybrief hcrngAllocStreams()
	*  @see hcrngAllocStreams()
	*/
	HCRNGAPI hcrngMrg32k3aStream* hcrngMrg32k3aAllocStreams(size_t count, size_t* bufSize, hcrngStatus* err);

	/*! @copybrief hcrngDestroyStreams()
	*  @see hcrngDestroyStreams()
	*/
	HCRNGAPI hcrngStatus hcrngMrg32k3aDestroyStreams(hcrngMrg32k3aStream* streams);

	/*! @copybrief hcrngCreateOverStreams()
	*  @see hcrngCreateOverStreams()
	*/
	HCRNGAPI hcrngStatus hcrngMrg32k3aCreateOverStreams(hcrngMrg32k3aStreamCreator* creator, size_t count, hcrngMrg32k3aStream* streams);

	/*! @copybrief hcrngCreateStreams()
	*  @see hcrngCreateStreams()
	*/
	HCRNGAPI hcrngMrg32k3aStream* hcrngMrg32k3aCreateStreams(hcrngMrg32k3aStreamCreator* creator, size_t count, size_t* bufSize, hcrngStatus* err);

	/*! @copybrief hcrngCopyOverStreams()
	*  @see hcrngCopyOverStreams()
	*/
	HCRNGAPI hcrngStatus hcrngMrg32k3aCopyOverStreams(size_t count, hcrngMrg32k3aStream* destStreams, const hcrngMrg32k3aStream* srcStreams);

	/*! @copybrief hcrngCopyStreams()
	*  @see hcrngCopyStreams()
	*/
	HCRNGAPI hcrngMrg32k3aStream* hcrngMrg32k3aCopyStreams(size_t count, const hcrngMrg32k3aStream* streams, hcrngStatus* err);

#define hcrngMrg32k3aRandomU01          _HCRNG_TAG_FPTYPE(hcrngMrg32k3aRandomU01)
#define hcrngMrg32k3aRandomN            _HCRNG_TAG_FPTYPE(hcrngMrg32k3aRandomN)           //Normal distribution 
#define hcrngMrg32k3aRandomInteger      _HCRNG_TAG_FPTYPE(hcrngMrg32k3aRandomInteger)
#define hcrngMrg32k3aRandomUnsignedInteger      _HCRNG_TAG_FPTYPE(hcrngMrg32k3aRandomUnsignedInteger)
#define hcrngMrg32k3aRandomU01Array     _HCRNG_TAG_FPTYPE(hcrngMrg32k3aRandomU01Array)
#define hcrngMrg32k3aRandomIntegerArray _HCRNG_TAG_FPTYPE(hcrngMrg32k3aRandomIntegerArray)
#define hcrngMrg32k3aRandomUnsignedIntegerArray _HCRNG_TAG_FPTYPE(hcrngMrg32k3aRandomUnsignedIntegerArray)

	/*! @copybrief hcrngRandomU01()
	*  @see hcrngRandomU01()
	*/
	HCRNGAPI _HCRNG_FPTYPE hcrngMrg32k3aRandomU01(hcrngMrg32k3aStream* stream);
	HCRNGAPI float  hcrngMrg32k3aRandomU01_float (hcrngMrg32k3aStream* stream);
	HCRNGAPI double hcrngMrg32k3aRandomU01_double(hcrngMrg32k3aStream* stream);
        
        // Normal distribution
        
	HCRNGAPI _HCRNG_FPTYPE hcrngMrg32k3aRandomN(hcrngMrg32k3aStream* stream1, hcrngMrg32k3aStream* stream2, _HCRNG_FPTYPE mu, _HCRNG_FPTYPE sigma);
	HCRNGAPI float  hcrngMrg32k3aRandomN_float (hcrngMrg32k3aStream* stream, hcrngMrg32k3aStream* stream2, float mu, float sigma);
	HCRNGAPI double hcrngMrg32k3aRandomN_double(hcrngMrg32k3aStream* stream, hcrngMrg32k3aStream* stream2, double mu, double sigma);


	/*! @copybrief hcrngRandomInteger()
	*  @see hcrngRandomInteger()
	*/
	HCRNGAPI  int hcrngMrg32k3aRandomInteger(hcrngMrg32k3aStream* stream,  int i,  int j);
	HCRNGAPI  int hcrngMrg32k3aRandomInteger_float (hcrngMrg32k3aStream* stream, int i, int j);
	HCRNGAPI  int hcrngMrg32k3aRandomInteger_double(hcrngMrg32k3aStream* stream, int i, int j);
        

        HCRNGAPI unsigned int hcrngMrg32k3aRandomUnsignedInteger(hcrngMrg32k3aStream* stream, unsigned int i, unsigned int j);
        HCRNGAPI unsigned int hcrngMrg32k3aRandomUnsignedInteger_float (hcrngMrg32k3aStream* stream, unsigned int i, unsigned int j);
        HCRNGAPI unsigned int hcrngMrg32k3aRandomUnsignedInteger_double(hcrngMrg32k3aStream* stream, unsigned int i, unsigned int j);


	/*! @copybrief hcrngRandomU01Array()
	*  @see hcrngRandomU01Array()
	*/
	HCRNGAPI hcrngStatus hcrngMrg32k3aRandomU01Array(hcrngMrg32k3aStream* stream, size_t count, _HCRNG_FPTYPE* buffer);
	HCRNGAPI hcrngStatus hcrngMrg32k3aRandomU01Array_float (hcrngMrg32k3aStream* stream, size_t count, float * buffer);
	HCRNGAPI hcrngStatus hcrngMrg32k3aRandomU01Array_double(hcrngMrg32k3aStream* stream, size_t count, double* buffer);

	/*! @copybrief hcrngRandomIntegerArray()
	*  @see hcrngRandomIntegerArray()
	*/
	HCRNGAPI hcrngStatus hcrngMrg32k3aRandomIntegerArray(hcrngMrg32k3aStream* stream, int i, int j, size_t count, int* buffer);
	HCRNGAPI hcrngStatus hcrngMrg32k3aRandomIntegerArray_float (hcrngMrg32k3aStream* stream, int i, int j, size_t count, int* buffer);
	HCRNGAPI hcrngStatus hcrngMrg32k3aRandomIntegerArray_double(hcrngMrg32k3aStream* stream, int i, int j, size_t count, int* buffer);


        HCRNGAPI hcrngStatus hcrngMrg32k3aRandomUnsignedIntegerArray(hcrngMrg32k3aStream* stream, unsigned int i, unsigned int j, size_t count, unsigned int* buffer);
        HCRNGAPI hcrngStatus hcrngMrg32k3aRandomUnsignedIntegerArray_float (hcrngMrg32k3aStream* stream, unsigned int i, unsigned int j, size_t count, unsigned int* buffer);
        HCRNGAPI hcrngStatus hcrngMrg32k3aRandomUnsignedIntegerArray_double(hcrngMrg32k3aStream* stream, unsigned int i, unsigned int j, size_t count, unsigned int* buffer);
	/*! @copybrief hcrngRewindStreams()
	*  @see hcrngRewindStreams()
	*/
	HCRNGAPI hcrngStatus hcrngMrg32k3aRewindStreams(size_t count, hcrngMrg32k3aStream* streams);

	/*! @copybrief hcrngRewindSubstreams()
	*  @see hcrngRewindSubstreams()
	*/
	HCRNGAPI hcrngStatus hcrngMrg32k3aRewindSubstreams(size_t count, hcrngMrg32k3aStream* streams);

	/*! @copybrief hcrngForwardToNextSubstreams()
	*  @see hcrngForwardToNextSubstreams()
	*/
	HCRNGAPI hcrngStatus hcrngMrg32k3aForwardToNextSubstreams(size_t count, hcrngMrg32k3aStream* streams);

	/*! @copybrief hcrngMakeSubstreams()
	 *  @see hcrngMakeSubstreams()
	 */
	HCRNGAPI hcrngMrg32k3aStream* hcrngMrg32k3aMakeSubstreams(hcrngMrg32k3aStream* stream, size_t count, size_t* bufSize, hcrngStatus* err);

	/*! @copybrief hcrngMakeOverSubstreams()
	 *  @see hcrngMakeOverSubstreams()
	 */
	HCRNGAPI hcrngStatus hcrngMrg32k3aMakeOverSubstreams(hcrngMrg32k3aStream* stream, size_t count, hcrngMrg32k3aStream* substreams);

	/*! @copybrief hcrngAdvanceStreams()
	*  @see hcrngAdvanceStreams()
	*/
	HCRNGAPI hcrngStatus hcrngMrg32k3aAdvanceStreams(size_t count, hcrngMrg32k3aStream* streams, int e, int c);

	/*! @copybrief hcrngDeviceRandomU01Array()
	*  @see hcrngDeviceRandomU01Array()
	*/
#ifdef HCRNG_SINGLE_PRECISION
#define hcrngMrg32k3aDeviceRandomU01Array(...) hcrngMrg32k3aDeviceRandomU01Array_(__VA_ARGS__, HC_TRUE)
#else
#define hcrngMrg32k3aDeviceRandomU01Array(...) hcrngMrg32k3aDeviceRandomU01Array_(__VA_ARGS__, HC_FALSE)
#endif

	/** \internal
	 *  @brief Helper function for hcrngMrg32k3aDeviceRandomU01Array()
	 */
	HCRNGAPI hcrngStatus hcrngMrg32k3aDeviceRandomU01Array_single(size_t streamCount,  hcrngMrg32k3aStream* streams,
		size_t numberCount, float* outBuffer, int streamlength = 0, size_t streams_per_thread = 1);
        HCRNGAPI hcrngStatus hcrngMrg32k3aDeviceRandomU01Array_double(size_t streamCount,  hcrngMrg32k3aStream* streams,
                size_t numberCount, double* outBuffer, int streamlength = 0, size_t streams_per_thread = 1);
/** \endinternal
 */
        HCRNGAPI hcrngStatus hcrngMrg32k3aDeviceRandomNArray_single(size_t streamCount, hcrngMrg32k3aStream *streams,
	       size_t numberCount, float mu, float sigma, float *outBuffer, int streamlength = 0, size_t streams_per_thread = 1);
        HCRNGAPI hcrngStatus hcrngMrg32k3aDeviceRandomNArray_double(size_t streamCount, hcrngMrg32k3aStream *streams,
	       size_t numberCount, double mu, double sigma, double *outBuffer, int streamlength = 0, size_t streams_per_thread = 1);
	/*! @copybrief hcrngWriteStreamInfo()
	*  @see hcrngWriteStreamInfo()
	*/
	HCRNGAPI hcrngStatus hcrngMrg32k3aWriteStreamInfo(const hcrngMrg32k3aStream* stream, FILE *file);

#ifdef __cplusplus
//}
#endif //__cplusplus
#endif // MRG32K3A_H
