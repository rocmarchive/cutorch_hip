#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCThrustAlternate.h"
#else

// Unary transforms
template <typename UnaryFunction>
void transform(THCState* state, THCTensor* first, THCTensor* result, UnaryFunction op);

// Binary transform
template <typename BinaryFunction>
void transform(THCState* state, THCTensor* first1, THCTensor* first2, THCTensor* result, BinaryFunction op); 

// Reduce function
template<class T, typename BinaryFunction>
T reduce(THCState* state, THCTensor* input, T init, BinaryFunction f);
 
// Innerproduct
template <class T, typename BinaryFunction1, typename BinaryFunction2>
T inner_product(THCState* state, THCTensor* first1, THCTensor* first2, T init, BinaryFunction1 op1, BinaryFunction2 op2);

#endif
