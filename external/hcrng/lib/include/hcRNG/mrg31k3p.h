/*  @file mrg31k3p.h
 *  @brief Specific interface for the MRG31k3p generator
 */

#pragma once
#ifndef MRG31K3P_H
#define MRG31K3P_H
#include "hcRNG.h"
#include <stdio.h>

#ifdef __cplusplus 

//extern "C" {

#endif 

/*  @brief State type of a MRG31k3p stream
 *  The state is a seed consisting of six unsigned 32-bit integers.
 *  @see hcrngStreamState
 */
typedef struct {
    /*! @brief Seed for the first MRG component
     */
    unsigned int g1[3];
    /*! @brief Seed for the second MRG component
     */
    unsigned int g2[3];
} hcrngMrg31k3pStreamState;


struct hcrngMrg31k3pStream_ {
	union {
		struct {
			hcrngMrg31k3pStreamState states[3];
		};
		struct {
			hcrngMrg31k3pStreamState current;
			hcrngMrg31k3pStreamState initial;
			hcrngMrg31k3pStreamState substream;
		};
	};
};

/*! @copybrief hcrngStream
 *  @see hcrngStream
 */
typedef struct hcrngMrg31k3pStream_ hcrngMrg31k3pStream;

//struct hcrngMrg31k3pStreamCreator_;
/*! @copybrief hcrngStreamCreator
 *  @see hcrngStreamCreator
 */

struct hcrngMrg31k3pStreamCreator_ {
        hcrngMrg31k3pStreamState initialState;
        hcrngMrg31k3pStreamState nextState;
        /*! @brief Jump matrices for advancing the initial seed of streams
         */
        unsigned int nuA1[3][3];
        unsigned int nuA2[3][3];
};



typedef struct hcrngMrg31k3pStreamCreator_ hcrngMrg31k3pStreamCreator;

/*! @brief Default initial seed of the first stream
 */
#define BASE_CREATOR_STATE_MRG31K3P { { 12345, 12345, 12345 }, { 12345, 12345, 12345 } }
/*! @brief Jump matrices for \f$2^{134}\f$ steps forward
 */
#define BASE_CREATOR_JUMP_MATRIX_1_MRG31K3P { \
        {1702500920, 1849582496, 1656874625}, \
        { 828554832, 1702500920, 1512419905}, \
        {1143731069,  828554832,  102237247} }
#define BASE_CREATOR_JUMP_MATRIX_2_MRG31K3P { \
        { 796789021, 1464208080,  607337906}, \
        {1241679051, 1431130166, 1464208080}, \
        {1401213391, 1178684362, 1431130166} }

/*! @brief Default stream creator (defaults to \f$2^{134}\f$ steps forward)
 *
 *  Contains the default seed and the transition matrices to jump \f$\nu\f$ steps forward;
 *  adjacent streams are spaced nu steps apart.
 *  The default is \f$nu = 2^{134}\f$.
 *  The default seed is \f$(12345,12345,12345,12345,12345,12345)\f$.
 */
static hcrngMrg31k3pStreamCreator defaultStreamCreator_Mrg31k3p = {
        BASE_CREATOR_STATE_MRG31K3P,
        BASE_CREATOR_STATE_MRG31K3P,
        BASE_CREATOR_JUMP_MATRIX_1_MRG31K3P,
        BASE_CREATOR_JUMP_MATRIX_2_MRG31K3P
};





/*! @copybrief hcrngCopyStreamCreator()
 *  @see hcrngCopyStreamCreator()
 */
HCRNGAPI hcrngMrg31k3pStreamCreator* hcrngMrg31k3pCopyStreamCreator(const hcrngMrg31k3pStreamCreator* creator, hcrngStatus* err);

/*! @copybrief hcrngDestroyStreamCreator()
 *  @see hcrngDestroyStreamCreator()
 */
HCRNGAPI hcrngStatus hcrngMrg31k3pDestroyStreamCreator(hcrngMrg31k3pStreamCreator* creator);

/*! @copybrief hcrngRewindStreamCreator()
 *  @see hcrngRewindStreamCreator()
 */
HCRNGAPI hcrngStatus hcrngMrg31k3pRewindStreamCreator(hcrngMrg31k3pStreamCreator* creator);

/*! @copybrief hcrngSetBaseCreatorState()
 *  @see hcrngSetBaseCreatorState()
 */
HCRNGAPI hcrngStatus hcrngMrg31k3pSetBaseCreatorState(hcrngMrg31k3pStreamCreator* creator, const hcrngMrg31k3pStreamState* baseState);

/*! @copybrief hcrngChangeStreamsSpacing()
 *  @see hcrngChangeStreamsSpacing()
 */
HCRNGAPI hcrngStatus hcrngMrg31k3pChangeStreamsSpacing(hcrngMrg31k3pStreamCreator* creator, int e, int c);

/*! @copybrief hcrngAllocStreams()
 *  @see hcrngAllocStreams()
 */
HCRNGAPI hcrngMrg31k3pStream* hcrngMrg31k3pAllocStreams(size_t count, size_t* bufSize, hcrngStatus* err);

/*! @copybrief hcrngDestroyStreams()
 *  @see hcrngDestroyStreams()
 */
HCRNGAPI hcrngStatus hcrngMrg31k3pDestroyStreams(hcrngMrg31k3pStream* streams);

/*! @copybrief hcrngCreateOverStreams()
 *  @see hcrngCreateOverStreams()
 */
HCRNGAPI hcrngStatus hcrngMrg31k3pCreateOverStreams(hcrngMrg31k3pStreamCreator* creator, size_t count, hcrngMrg31k3pStream* streams);

/*! @copybrief hcrngCreateStreams()
 *  @see hcrngCreateStreams()
 */
HCRNGAPI hcrngMrg31k3pStream* hcrngMrg31k3pCreateStreams(hcrngMrg31k3pStreamCreator* creator, size_t count, size_t* bufSize, hcrngStatus* err);

/*! @copybrief hcrngCopyOverStreams()
 *  @see hcrngCopyOverStreams()
 */
HCRNGAPI hcrngStatus hcrngMrg31k3pCopyOverStreams(size_t count, hcrngMrg31k3pStream* destStreams, const hcrngMrg31k3pStream* srcStreams);

/*! @copybrief hcrngCopyStreams()
 *  @see hcrngCopyStreams()
 */
HCRNGAPI hcrngMrg31k3pStream* hcrngMrg31k3pCopyStreams(size_t count, const hcrngMrg31k3pStream* streams, hcrngStatus* err);

#define hcrngMrg31k3pRandomU01          _HCRNG_TAG_FPTYPE(hcrngMrg31k3pRandomU01)
#define hcrngMrg31k3pRandomN            _HCRNG_TAG_FPTYPE(hcrngMrg31k3pRandomN)
#define hcrngMrg31k3pRandomInteger      _HCRNG_TAG_FPTYPE(hcrngMrg31k3pRandomInteger)
#define hcrngMrg31k3pRandomUnsignedInteger      _HCRNG_TAG_FPTYPE(hcrngMrg31k3pRandomUnsignedInteger)
#define hcrngMrg31k3pRandomU01Array     _HCRNG_TAG_FPTYPE(hcrngMrg31k3pRandomU01Array)
#define hcrngMrg31k3pRandomIntegerArray _HCRNG_TAG_FPTYPE(hcrngMrg31k3pRandomIntegerArray)
#define hcrngMrg31k3pRandomUnsignedIntegerArray _HCRNG_TAG_FPTYPE(hcrngMrg31k3pRandomUnsignedIntegerArray)

/*! @copybrief hcrngRandomU01()
 *  @see hcrngRandomU01()
 */
HCRNGAPI _HCRNG_FPTYPE hcrngMrg31k3pRandomU01(hcrngMrg31k3pStream* stream);
HCRNGAPI float  hcrngMrg31k3pRandomU01_float (hcrngMrg31k3pStream* stream);
HCRNGAPI double hcrngMrg31k3pRandomU01_double(hcrngMrg31k3pStream* stream);

        
	HCRNGAPI _HCRNG_FPTYPE hcrngMrg31k3pRandomN(hcrngMrg31k3pStream* stream1, hcrngMrg31k3pStream* stream2, _HCRNG_FPTYPE mu, _HCRNG_FPTYPE sigma);
	HCRNGAPI float  hcrngMrg31k3pRandomN_float (hcrngMrg31k3pStream* stream, hcrngMrg31k3pStream* stream2, float mu, float sigma);
	HCRNGAPI double hcrngMrg31k3pRandomN_double(hcrngMrg31k3pStream* stream, hcrngMrg31k3pStream* stream2, double mu, double sigma);


/*! @copybrief hcrngRandomInteger()
 *  @see hcrngRandomInteger()
 */
HCRNGAPI int hcrngMrg31k3pRandomInteger(hcrngMrg31k3pStream* stream, int i, int j);
HCRNGAPI int hcrngMrg31k3pRandomInteger_float (hcrngMrg31k3pStream* stream, int i, int j);
HCRNGAPI int hcrngMrg31k3pRandomInteger_double(hcrngMrg31k3pStream* stream, int i, int j);

HCRNGAPI unsigned int hcrngMrg31k3pRandomUnsignedInteger(hcrngMrg31k3pStream* stream, unsigned int i, unsigned int j);
HCRNGAPI unsigned int hcrngMrg31k3pRandomUnsignedInteger_float (hcrngMrg31k3pStream* stream, unsigned int i, unsigned int j);
HCRNGAPI unsigned int hcrngMrg31k3pRandomUnsignedInteger_double(hcrngMrg31k3pStream* stream, unsigned int i, unsigned int j);



/*! @copybrief hcrngRandomU01Array()
 *  @see hcrngRandomU01Array()
 */
HCRNGAPI hcrngStatus hcrngMrg31k3pRandomU01Array(hcrngMrg31k3pStream* stream, size_t count, _HCRNG_FPTYPE* buffer);
HCRNGAPI hcrngStatus hcrngMrg31k3pRandomU01Array_float (hcrngMrg31k3pStream* stream, size_t count, float * buffer);
HCRNGAPI hcrngStatus hcrngMrg31k3pRandomU01Array_double(hcrngMrg31k3pStream* stream, size_t count, double* buffer);

/*! @copybrief hcrngRandomIntegerArray()
 *  @see hcrngRandomIntegerArray()
 */
HCRNGAPI hcrngStatus hcrngMrg31k3pRandomIntegerArray(hcrngMrg31k3pStream* stream, int i, int j, size_t count, int* buffer);
HCRNGAPI hcrngStatus hcrngMrg31k3pRandomIntegerArray_float (hcrngMrg31k3pStream* stream, int i, int j, size_t count, int* buffer);
HCRNGAPI hcrngStatus hcrngMrg31k3pRandomIntegerArray_double(hcrngMrg31k3pStream* stream, int i, int j, size_t count, int* buffer);

HCRNGAPI hcrngStatus hcrngMrg31k3pRandomUnsignedIntegerArray(hcrngMrg31k3pStream* stream, unsigned int i, unsigned int j, size_t count, unsigned int* buffer);
HCRNGAPI hcrngStatus hcrngMrg31k3pRandomUnsignedIntegerArray_float (hcrngMrg31k3pStream* stream, unsigned int i, unsigned int j, size_t count, unsigned int* buffer);
HCRNGAPI hcrngStatus hcrngMrg31k3pRandomUnsignedIntegerArray_double(hcrngMrg31k3pStream* stream, unsigned int i, unsigned int j, size_t count, unsigned int* buffer);


/*! @copybrief hcrngRewindStreams()
 *  @see hcrngRewindStreams()
 */
HCRNGAPI hcrngStatus hcrngMrg31k3pRewindStreams(size_t count, hcrngMrg31k3pStream* streams);

/*! @copybrief hcrngRewindSubstreams()
 *  @see hcrngRewindSubstreams()
 */
HCRNGAPI hcrngStatus hcrngMrg31k3pRewindSubstreams(size_t count, hcrngMrg31k3pStream* streams);

/*! @copybrief hcrngForwardToNextSubstreams()
 *  @see hcrngForwardToNextSubstreams()
 */
HCRNGAPI hcrngStatus hcrngMrg31k3pForwardToNextSubstreams(size_t count, hcrngMrg31k3pStream* streams);

/*! @copybrief hcrngMakeSubstreams()
 *  @see hcrngMakeSubstreams()
 */
HCRNGAPI hcrngMrg31k3pStream* hcrngMrg31k3pMakeSubstreams(hcrngMrg31k3pStream* stream, size_t count, size_t* bufSize, hcrngStatus* err);

/*! @copybrief hcrngMakeOverSubstreams()
 *  @see hcrngMakeOverSubstreams()
 */
HCRNGAPI hcrngStatus hcrngMrg31k3pMakeOverSubstreams(hcrngMrg31k3pStream* stream, size_t count, hcrngMrg31k3pStream* substreams);

/*! @copybrief hcrngAdvanceStreams()
 *  @see hcrngAdvanceStreams()
 */
HCRNGAPI hcrngStatus hcrngMrg31k3pAdvanceStreams(size_t count, hcrngMrg31k3pStream* streams, int e, int c);

/*! @copybrief hcrngDeviceRandomU01Array()
 *  @see hcrngDeviceRandomU01Array()
 */
#ifdef HCRNG_SINGLE_PRECISION
#define hcrngMrg31k3pDeviceRandomU01Array(...) hcrngMrg31k3pDeviceRandomU01Array_(__VA_ARGS__, HC_TRUE)
#else
#define hcrngMrg31k3pDeviceRandomU01Array(...) hcrngMrg31k3pDeviceRandomU01Array_(__VA_ARGS__, HC_FALSE)
#endif

/** \internal
 *  @brief Helper function for hcrngMrg31k3pDeviceRandomU01Array()
 */
HCRNGAPI hcrngStatus hcrngMrg31k3pDeviceRandomU01Array_single(size_t streamCount, hcrngMrg31k3pStream *streams,
	size_t numberCount, float *outBuffer, int streamlength = 0, size_t streams_per_thread = 1);
HCRNGAPI hcrngStatus hcrngMrg31k3pDeviceRandomU01Array_double(size_t streamCount, hcrngMrg31k3pStream *streams,
        size_t numberCount, double *outBuffer, int streamlength = 0, size_t streams_per_thread = 1);
/** \endinternal
 */

// Normal distribution
HCRNGAPI hcrngStatus hcrngMrg31k3pDeviceRandomNArray_single(size_t streamCount, hcrngMrg31k3pStream *streams,
	size_t numberCount, float mu, float sigma, float *outBuffer, int streamlength = 0, size_t streams_per_thread = 1);
HCRNGAPI hcrngStatus hcrngMrg31k3pDeviceRandomNArray_double(size_t streamCount, hcrngMrg31k3pStream *streams,
	size_t numberCount, double mu, double sigma, double *outBuffer, int streamlength = 0, size_t streams_per_thread = 1);
	
/*! @copybrief hcrngWriteStreamInfo()
 *  @see hcrngWriteStreamInfo()
 */
HCRNGAPI hcrngStatus hcrngMrg31k3pWriteStreamInfo(const hcrngMrg31k3pStream* stream, FILE *file);

#ifdef __cplusplus 
//}
#endif //__cplusplus
#endif // MRG31K3P_H

