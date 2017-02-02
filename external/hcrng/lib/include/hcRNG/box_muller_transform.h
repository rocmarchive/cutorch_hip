#include <cstdlib>
#include <cmath>
#include <limits>
#include "hcRNG.h"
#include <hc.hpp>
#include <hc_am.hpp>
HCRNGAPI hcrngStatus box_muller_transform_single(hc::accelerator_view &accl_view, float mu, float sigma, float *OutBuffer, size_t numberCount);
HCRNGAPI hcrngStatus box_muller_transform_double(hc::accelerator_view &accl_view, double mu, double sigma, double *OutBuffer, size_t numberCount);
