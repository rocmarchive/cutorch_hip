/*! @file Lfsr113.c.h
*  @brief Code for the Lfsr113 generator common to the host and device
*/

#pragma once
#ifndef PRIVATE_LFSR113_CH
#define PRIVATE_LFSR113_CH

#define Lfsr113_NORM_double 1.0 / 0x100000001L   // 2^32 + 1    // 2.328306436538696e-10
#define Lfsr113_NORM_float  2.3283063e-10
#include <hc_math.hpp>

hcrngStatus hcrngLfsr113CopyOverStreams(size_t count, hcrngLfsr113Stream* destStreams, const hcrngLfsr113Stream* srcStreams) __attribute__((hc, cpu)) 
{
        if (!destStreams)
                return HCRNG_INVALID_VALUE;
        if (!srcStreams)
                return HCRNG_INVALID_VALUE;

	for (size_t i = 0; i < count; i++)
		destStreams[i] = srcStreams[i];

	return HCRNG_SUCCESS;
}

/*! @brief Advance the rng one step and returns z such that 1 <= z <= lfsr113_M1
*/
static unsigned long hcrngLfsr113NextState(hcrngLfsr113StreamState *currentState) __attribute__((hc, cpu)) {

	unsigned long b;

	b = (((currentState->g[0] << 6) ^ currentState->g[0]) >> 13);
	currentState->g[0] = (((currentState->g[0] & 4294967294U) << 18) ^ b);

	b = (((currentState->g[1] << 2) ^ currentState->g[1]) >> 27);
	currentState->g[1] = (((currentState->g[1] & 4294967288U) << 2) ^ b);

	b = (((currentState->g[2] << 13) ^ currentState->g[2]) >> 21);
	currentState->g[2] = (((currentState->g[2] & 4294967280U) << 7) ^ b);

	b = (((currentState->g[3] << 3) ^ currentState->g[3]) >> 12);
	currentState->g[3] = (((currentState->g[3] & 4294967168U) << 13) ^ b);

	return (currentState->g[0] ^ currentState->g[1] ^ currentState->g[2] ^ currentState->g[3]);

}

// The following would be much hceaner with C++ templates instead of macros.

// We use an underscore on the r.h.s. to avoid potential recursion with certain
// preprocessors.
#define IMPLEMENT_GENERATE_FOR_TYPE(fptype) \
	\
	fptype hcrngLfsr113RandomU01_##fptype(hcrngLfsr113Stream* stream) __attribute__((hc, cpu)) { \
	    return hcrngLfsr113NextState(&stream->current) * Lfsr113_NORM_##fptype; \
	} \
	\
        fptype hcrngLfsr113RandomN_##fptype(hcrngLfsr113Stream* stream1, hcrngLfsr113Stream* stream2, fptype mu, fptype sigma) __attribute__((hc,cpu)) { \
            static fptype z0, z1;\
            const fptype two_pi = 2.0 * 3.14159265358979323846;\
            static bool generate;\
            generate =! generate;\
            if (!generate) return z1 * sigma +mu;\
            fptype u1, u2;\
            u1 = (fptype)hcrngLfsr113RandomU01_##fptype(stream1);\
            u2 = (fptype)hcrngLfsr113RandomU01_##fptype(stream2);\
            z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);\
            z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);\
	    return z0 * sigma + mu; \
	} \
	\
        unsigned int hcrngLfsr113RandomUnsignedInteger_##fptype(hcrngLfsr113Stream* stream, unsigned int i, unsigned int j) __attribute__((hc, cpu)) { \
            return i + (unsigned int)((j - i + 1) * hcrngLfsr113RandomU01_##fptype(stream)); \
        } \
        \
	int hcrngLfsr113RandomInteger_##fptype(hcrngLfsr113Stream* stream, int i, int j) __attribute__((hc, cpu)) { \
	    return i + (int)((j - i + 1) * hcrngLfsr113RandomU01_##fptype(stream)); \
	} \
	\
	hcrngStatus hcrngLfsr113RandomU01Array_##fptype(hcrngLfsr113Stream* stream, size_t count, fptype* buffer) __attribute__((hc, cpu)) { \
		for (size_t i = 0; i < count; i++)  \
			buffer[i] = hcrngLfsr113RandomU01_##fptype(stream); \
		return HCRNG_SUCCESS; \
	} \
	\
	hcrngStatus hcrngLfsr113RandomIntegerArray_##fptype(hcrngLfsr113Stream* stream, int i, int j, size_t count, int* buffer) __attribute__((hc, cpu)) { \
		for (size_t k = 0; k < count; k++) \
			buffer[k] = hcrngLfsr113RandomInteger_##fptype(stream, i, j); \
		return HCRNG_SUCCESS; \
	}\
        hcrngStatus hcrngLfsr113RandomUnsignedIntegerArray_##fptype(hcrngLfsr113Stream* stream, unsigned int i, unsigned int j, size_t count, unsigned int* buffer) __attribute__((hc, cpu)) { \
                for (size_t k = 0; k < count; k++) \
                        buffer[k] = hcrngLfsr113RandomUnsignedInteger_##fptype(stream, i, j); \
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



hcrngStatus hcrngLfsr113RewindStreams(size_t count, hcrngLfsr113Stream* streams) __attribute__((hc, cpu))
{
        if (!streams)
                return HCRNG_INVALID_VALUE;

	//Reset current state to the stream initial state
	for (size_t j = 0; j < count; j++) {
		streams[j].current = streams[j].substream = streams[j].initial;
	}

	return HCRNG_SUCCESS;
}

hcrngStatus hcrngLfsr113RewindSubstreams(size_t count, hcrngLfsr113Stream* streams) __attribute__((hc, cpu))
{
        if (!streams)
                return HCRNG_INVALID_VALUE;

	//Reset current state to the subStream initial state
	for (size_t j = 0; j < count; j++) {
		streams[j].current = streams[j].substream;
	}

	return HCRNG_SUCCESS;
}
void lfsr113ResetNextSubStream(hcrngLfsr113Stream* stream) __attribute__((hc, cpu)) {

	/* The following operations make the jump ahead with
	2 ^ 55 iterations for every component of the generator.
	The internal state after the jump, however, is slightly different
	from 2 ^ 55 iterations since it ignores the state in
	which are found the first bits of each components,
	since they are ignored in the recurrence.The state becomes
	identical to what one would with normal iterations
	after a call nextValue().*/

	int z, b;

	unsigned int* subStreamState = stream->substream.g;

	//Calculate the first component
	z = subStreamState[0] & (unsigned int)-2;
	b = (z << 6) ^ z;

	z = (z) ^ (z << 3) ^ (z << 4) ^ (z << 6) ^ (z << 7) ^
		(z << 8) ^ (z << 10) ^ (z << 11) ^ (z << 13) ^ (z << 14) ^
		(z << 16) ^ (z << 17) ^ (z << 18) ^ (z << 22) ^
		(z << 24) ^ (z << 25) ^ (z << 26) ^ (z << 28) ^ (z << 30);

	z ^= ((b >> 1) & 0x7FFFFFFF) ^
		((b >> 3) & 0x1FFFFFFF) ^
		((b >> 5) & 0x07FFFFFF) ^
		((b >> 6) & 0x03FFFFFF) ^
		((b >> 7) & 0x01FFFFFF) ^
		((b >> 9) & 0x007FFFFF) ^
		((b >> 13) & 0x0007FFFF) ^
		((b >> 14) & 0x0003FFFF) ^
		((b >> 15) & 0x0001FFFF) ^
		((b >> 17) & 0x00007FFF) ^
		((b >> 18) & 0x00003FFF) ^
		((b >> 20) & 0x00000FFF) ^
		((b >> 21) & 0x000007FF) ^
		((b >> 23) & 0x000001FF) ^
		((b >> 24) & 0x000000FF) ^
		((b >> 25) & 0x0000007F) ^
		((b >> 26) & 0x0000003F) ^
		((b >> 27) & 0x0000001F) ^
		((b >> 30) & 0x00000003);
	subStreamState[0] = z;

	//Calculate the second component
	z = subStreamState[1] & (unsigned int)-8;
	b = z ^ (z << 1);
	b ^= (b << 2);
	b ^= (b << 4);
	b ^= (b << 8);

	b <<= 8;
	b ^= (z << 22) ^ (z << 25) ^ (z << 27);
	if ((z & 0x80000000) != 0) b ^= 0xABFFF000;
	if ((z & 0x40000000) != 0) b ^= 0x55FFF800;

	z = b ^ ((z >> 7) & 0x01FFFFFF) ^
		((z >> 20) & 0x00000FFF) ^
		((z >> 21) & 0x000007FF);

	subStreamState[1] = z;

	//Calculate the third component
	z = subStreamState[2] & (unsigned int)-16;
	b = (z << 13) ^ z;
	z = ((b >> 3) & 0x1FFFFFFF) ^
		((b >> 17) & 0x00007FFF) ^
		(z << 10) ^ (z << 11) ^ (z << 25);
	subStreamState[2] = z;

	//Calculate the forth component
	z = subStreamState[3] & (unsigned int)-128;
	b = (z << 3) ^ z;
	z = (z << 14) ^ (z << 16) ^ (z << 20) ^
		((b >> 5) & 0x07FFFFFF) ^
		((b >> 9) & 0x007FFFFF) ^
		((b >> 11) & 0x001FFFFF);
	subStreamState[3] = z;

	hcrngLfsr113RewindSubstreams(1, stream);
}
hcrngStatus hcrngLfsr113ForwardToNextSubstreams(size_t count, hcrngLfsr113Stream* streams) __attribute__((hc, cpu))
{
        if (!streams)
                return HCRNG_INVALID_VALUE;

	for (size_t k = 0; k < count; k++) {

		lfsr113ResetNextSubStream(&streams[k]);
	}

	return HCRNG_SUCCESS;
}

hcrngStatus hcrngLfsr113MakeOverSubstreams(hcrngLfsr113Stream* stream, size_t count, hcrngLfsr113Stream* substreams) __attribute__((hc, cpu))
{
	for (size_t i = 0; i < count; i++) {
		hcrngStatus err;
		// snapshot current stream into substreams[i]
		err = hcrngLfsr113CopyOverStreams(1, &substreams[i], stream);
		if (err != HCRNG_SUCCESS)
		    return err;
		// advance to next substream
		err = hcrngLfsr113ForwardToNextSubstreams(1, stream);
		if (err != HCRNG_SUCCESS)
		    return err;
	}
	return HCRNG_SUCCESS;
}

#endif // PRIVATE_Lfsr113_CH
