#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCThrustAlternate.h"
#else

// Unary transforms
template <typename UnaryFunction>
void transform(THCState* state, THCTensor* first, THCTensor* result, UnaryFunction op);

// Binary transform
template <typename BinaryFunction>
void transform(THCState* state, THCTensor* first1, THCTensor* first2, THCTensor* result, BinaryFunction op);

// Binary Transform kernel code
template <typename BinaryFunction>
void binary_transform_kernel(THCState* state, real*& first1, long first1Offset,
                      real*& first2, long first2Offset,
                      real*& result, long resultOffset, long size,  BinaryFunction f);


// Unary Transform kernel
template <typename UnaryFunction>
void unary_transform_kernel(THCState* state, real*& first, long firstOffset,
                     real*& result, long resultOffset, long size, UnaryFunction f);


 
#endif
