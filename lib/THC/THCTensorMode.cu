#include "THC.h"
#include "THCThrustAllocator.cuh"
#include "THCTensorTypeUtils.cuh"
#include "THCReduceApplyUtils.cuh"
#ifdef THRUST_PATH
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#else
#include <bolt/amp/iterator/ubiquitous_iterator.h>
#include <bolt/amp/iterator/counting_iterator.h>
#include <bolt/amp/transform.h>
//#include <bolt/amp/sort_by_key.h>
#include <bolt/amp/inner_product.h>

template<typename T>
struct sequence_functor
{
  T init, step;

  __host__ __device__
  sequence_functor(T init, T step)
    : init(init), step(step)
  {}

  template<typename Index>
  __host__ __device__
  T operator()(Index i) const
  {
    return init + step * i;
  }
};

template<typename T>
struct not_equal_to
{
 __host__ __device__
 bool operator()(const T& x, const T& y) const {return x!=y;}
};

#endif

#include "THCTensorMode.cuh"

#include "generic/THCTensorMode.cu"
#include "THCGenerateAllTypes.h"
