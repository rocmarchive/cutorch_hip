/*! @file Mrg32k3a.c.h
*  @brief Code for the Mrg32k3a generator common to the host and device
*/
#pragma once
#ifndef PRIVATE_MRG32K3A_CH
#define PRIVATE_MRG32K3A_CH

#define Mrg32k3a_M1 4294967087            
#define Mrg32k3a_M2 4294944443             

#define Mrg32k3a_NORM_double 2.328306549295727688e-10
#define Mrg32k3a_NORM_float  2.3283064e-10

#include <hc.hpp>
#include <hc_math.hpp>

// hcrngMrg32k3a_A1p76 and hcrngMrg32k3a_A2p76 jump 2^76 steps forward
static
unsigned long hcrngMrg32k3a_A1p76[3][3] = {
	{ 82758667, 1871391091, 4127413238 },
	{ 3672831523, 69195019, 1871391091 },
	{ 3672091415, 3528743235, 69195019 }
};

static
unsigned long hcrngMrg32k3a_A2p76[3][3] = {
	{ 1511326704, 3759209742, 1610795712 },
	{ 4292754251, 1511326704, 3889917532 },
	{ 3859662829, 4292754251, 3708466080 }
};

hcrngStatus hcrngMrg32k3aCopyOverStreams(size_t count, hcrngMrg32k3aStream* destStreams, const hcrngMrg32k3aStream* srcStreams) __attribute__((hc, cpu)) 
{
        //Check params
        if (!destStreams)
                return HCRNG_INVALID_VALUE;
        if (!srcStreams)
                return HCRNG_INVALID_VALUE;

	for (size_t i = 0; i < count; i++)
		destStreams[i] = srcStreams[i];

	return HCRNG_SUCCESS;
}

/*! @brief Advance the rng one step and returns z such that 1 <= z <= Mrg32k3a_M1
*/
static unsigned long hcrngMrg32k3aNextState(hcrngMrg32k3aStreamState* currentState) __attribute__((hc, cpu))
{

	unsigned long* g1 = currentState->g1;
	unsigned long* g2 = currentState->g2;

	long p0, p1;

	/* component 1 */
	p0 = 1403580 * g1[1] - 810728 * g1[0];
	p0 %= Mrg32k3a_M1;
	if (p0 < 0)
		p0 += Mrg32k3a_M1;
	g1[0] = g1[1];
	g1[1] = g1[2];
	g1[2] = p0;

	/* component 2 */
	p1 = 527612 * g2[2] - 1370589 * g2[0];
	p1 %= Mrg32k3a_M2;
	if (p1 < 0)
		p1 += Mrg32k3a_M2;
	g2[0] = g2[1];
	g2[1] = g2[2];
	g2[2] = p1;

	/* combinations */
	if (p0 > p1)
		return (p0 - p1);
	else return (p0 - p1 + Mrg32k3a_M1);
}



// We use an underscore on the r.h.s. to avoid potential recursion with certain
// preprocessors.
#define IMPLEMENT_GENERATE_FOR_TYPE(fptype) \
	\
	fptype hcrngMrg32k3aRandomU01_##fptype(hcrngMrg32k3aStream* stream) __attribute__((hc, cpu)) { \
	    return hcrngMrg32k3aNextState(&stream->current) * Mrg32k3a_NORM_##fptype; \
	} \
	\
        fptype hcrngMrg32k3aRandomN_##fptype(hcrngMrg32k3aStream* stream1, hcrngMrg32k3aStream* stream2, fptype mu, fptype sigma) __attribute__((hc,cpu)) { \
            static fptype z0, z1, i;\
            i++;\
            const fptype two_pi = 2.0 * 3.14159265358979323846;\
            static bool generate;\
            generate =! generate;\
            if (!generate) return z1 * sigma +mu;\
            fptype u1, u2;\
            u1 = hcrngMrg32k3aRandomU01_##fptype(stream1);\
            u2 = hcrngMrg32k3aRandomU01_##fptype(stream2);\
            z0 = sqrt(-2.0 * log((float)u1)) * cos(two_pi * (float)u2);\
            z1 = sqrt(-2.0 * log((float)u1)) * sin(two_pi * (float)u2);\
	    return z0 * sigma + mu; \
	} \
	\
        int hcrngMrg32k3aRandomInteger_##fptype(hcrngMrg32k3aStream* stream,  int i, int j) __attribute__((hc, cpu)) { \
	    return i + (int)((j - i + 1) * hcrngMrg32k3aRandomU01_##fptype(stream)); \
	} \
	\
        unsigned int hcrngMrg32k3aRandomUnsignedInteger_##fptype(hcrngMrg32k3aStream* stream, unsigned int i, unsigned int j) __attribute__((hc, cpu)) { \
            return i + (unsigned int)((j - i + 1) * hcrngMrg32k3aRandomU01_##fptype(stream)); \
        } \
	hcrngStatus hcrngMrg32k3aRandomU01Array_##fptype(hcrngMrg32k3aStream* stream, size_t count, fptype* buffer) __attribute__((hc, cpu)) { \
		for (size_t i = 0; i < count; i++)  \
			buffer[i] = hcrngMrg32k3aRandomU01_##fptype(stream); \
		return HCRNG_SUCCESS; \
	} \
	\
	hcrngStatus hcrngMrg32k3aRandomIntegerArray_##fptype(hcrngMrg32k3aStream* stream, int i, int j, size_t count, int* buffer) __attribute__((hc, cpu)) { \
		for (size_t k = 0; k < count; k++) \
			buffer[k] = hcrngMrg32k3aRandomInteger_##fptype(stream, i, j); \
		return HCRNG_SUCCESS; \
	}\
       hcrngStatus hcrngMrg32k3aRandomUnsignedIntegerArray_##fptype(hcrngMrg32k3aStream* stream, unsigned int i, unsigned int j, size_t count, unsigned int* buffer) __attribute__((hc, cpu)) { \
                for (size_t k = 0; k < count; k++) \
                  buffer[k] = hcrngMrg32k3aRandomUnsignedInteger_##fptype(stream, i, j); \
                return HCRNG_SUCCESS; \
        }

// On the host, implement everything.
// On the device, implement only what is required to avoid hcuttering memory.
#if defined(HCRNG_SINGLE_PRECISION)  
IMPLEMENT_GENERATE_FOR_TYPE(float)
#endif
#if !defined(HCRNG_SINGLE_PRECISION) 
IMPLEMENT_GENERATE_FOR_TYPE(double)
#endif

// Clean up macros, especially to avoid polluting device code.
#undef IMPLEMENT_GENERATE_FOR_TYPE



hcrngStatus hcrngMrg32k3aRewindStreams(size_t count, hcrngMrg32k3aStream* streams) __attribute__((hc, cpu))
{
        if (!streams)
                return HCRNG_INVALID_VALUE;
	//Reset current state to the stream initial state
	for (size_t j = 0; j < count; j++) {
		streams[j].current = streams[j].substream = streams[j].initial;
	}

	return HCRNG_SUCCESS;
}

hcrngStatus hcrngMrg32k3aRewindSubstreams(size_t count, hcrngMrg32k3aStream* streams) __attribute__((hc, cpu))
{
        if (!streams)
                return HCRNG_INVALID_VALUE;
	//Reset current state to the subStream initial state
	for (size_t j = 0; j < count; j++) {
		streams[j].current = streams[j].substream;
	}

	return HCRNG_SUCCESS;
}

hcrngStatus hcrngMrg32k3aForwardToNextSubstreams(size_t count, hcrngMrg32k3aStream* streams) __attribute__((hc, cpu))
{
        if (!streams)
                return HCRNG_INVALID_VALUE;

	for (size_t k = 0; k < count; k++) {
		modMatVec(hcrngMrg32k3a_A1p76, streams[k].substream.g1, streams[k].substream.g1, Mrg32k3a_M1);
		modMatVec(hcrngMrg32k3a_A2p76, streams[k].substream.g2, streams[k].substream.g2, Mrg32k3a_M2);
		streams[k].current = streams[k].substream;
	}

	return HCRNG_SUCCESS;
}

hcrngStatus hcrngMrg32k3aMakeOverSubstreams(hcrngMrg32k3aStream* stream, size_t count, hcrngMrg32k3aStream* substreams) __attribute__((hc, cpu))
{
	for (size_t i = 0; i < count; i++) {
		hcrngStatus err;
		// snapshot current stream into substreams[i]
		err = hcrngMrg32k3aCopyOverStreams(1, &substreams[i], stream);
		if (err != HCRNG_SUCCESS)
		    return err;
		// advance to next substream
		err = hcrngMrg32k3aForwardToNextSubstreams(1, stream);
		if (err != HCRNG_SUCCESS)
		    return err;
	}
	return HCRNG_SUCCESS;
}

#endif // PRIVATE_Mrg32k3a_CH
