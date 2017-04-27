#pragma once

// FIXME: Better to inline every amp-restricted functions and make it static
#include <hc.hpp>
#include <hc_math.hpp>
#include "hc_am.hpp"
#include "mtgp32-fast.h"

using namespace hc;

#define MTGP32_MEXP 11213
#define MTGP32_N 351
#define MTGP32_FLOOR_2P 256
#define MTGP32_CEIL_2P 512
#define MTGP32_TN MTGP32_FLOOR_2P // TBL_NUMBER, do not exceed 256, i.e. WAVEFRONT_SIZE
#define MTGP32_LS (MTGP32_TN * 3)
#define MTGP32_TS 16              // TBL_SIZE
#define HcRAND_GROUP_NUM 200     // < MTGP32_TN, user specifies USER_GROUP_NUM
#define MTGP32_STATE_SIZE 1024    // The effective state number is =351
#define MTGP32_STATE_MASK 1023
#define USER_GROUP_NUM 64         // < HcRAND_GROUP_NUM

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif
#define BLOCK_SIZE 256
#define MAX_NUM_BLOCKS 64

// Structure of HipRandStateMtgp32
typedef struct HipRandStateMtgp32 {
  uint32_t* offset;     // size: USER_GROUP_NUM
  uint32_t* index;      // size: USER_GROUP_NUM
  uint32_t* d_status;   // extent<2>(USER_GROUP_NUM, MTGP32_STATE_SIZE)
  // mtgp32 kernel params
  uint32_t* mexp_tbl;   // size: 1. Redundant
  uint32_t* param_tbl;  // extent<2>(HcRAND_GROUP_NUM, MTGP32_TN)
  uint32_t* temper_tbl; // extent<2>(HcRAND_GROUP_NUM, MTGP32_TN)
  uint32_t* single_temper_tbl; // extent<2>(HcRAND_GROUP_NUM, MTGP32_TN)
  uint32_t* pos_tbl;    // size: MTGP32_TN
  uint32_t* sh1_tbl;    // size: MTGP32_TN
  uint32_t* sh2_tbl;    // size: MTGP32_TN
  uint32_t* mask;       // size: 1
} HipRandStateMtgp32;

// host array
typedef struct HOSTRandStateMtgp32 {
  uint32_t offset[USER_GROUP_NUM]; // size: USER_GROUP_NUM
  uint32_t index[USER_GROUP_NUM];  // size: USER_GROUP_NUM
  uint32_t d_status[USER_GROUP_NUM * MTGP32_STATE_SIZE]; // extent<2>(USER_GROUP_NUM, MTGP32_STATE_SIZE)
  // mtgp32 kernel params
  uint32_t mexp_tbl[HcRAND_GROUP_NUM];                      // size: 1. Redundant
  uint32_t param_tbl[HcRAND_GROUP_NUM * MTGP32_TN];         // extent<2>(HcRAND_GROUP_NUM, MTGP32_TN)
  uint32_t temper_tbl[HcRAND_GROUP_NUM * MTGP32_TN];        // extent<2>(HcRAND_GROUP_NUM, MTGP32_TN)
  uint32_t single_temper_tbl[HcRAND_GROUP_NUM * MTGP32_TN]; // extent<2>(HcRAND_GROUP_NUM, MTGP32_TN)
  uint32_t pos_tbl[MTGP32_TN];  // size: MTGP32_TN
  uint32_t sh1_tbl[MTGP32_TN];  // size: MTGP32_TN
  uint32_t sh2_tbl[MTGP32_TN];  // size: MTGP32_TN
  uint32_t mask[1];             // size: 1
} HOSTRandStateMtgp32;

void HipRandStateMtgp32_init(
    hc::accelerator_view accl_view, HipRandStateMtgp32* s);
void HipRandStateMtgp32_release(HipRandStateMtgp32* s);
void HipRandStateMtgp32_copy_D2H(void* src, void* dst);
void HipRandStateMtgp32_copy_H2D(void* src, void* dst);

// Device APIs
int mtgp32_init_params_kernel(
    hc::accelerator_view accl_view,
    const mtgp32_params_fast_t* params,
    HipRandStateMtgp32*& s);
int mtgp32_init_seed_kernel(
    hc::accelerator_view accl_view,
    const HipRandStateMtgp32* s,
    unsigned long seed);
/**
 * The function of the recursion formula calculation.
 *
 * @param[in] X1 the farthest part of state array.
 * @param[in] X2 the second farthest part of state array.
 * @param[in] Y a part of state array.
 * @return output
 */
static
inline
uint32_t para_rec(
    uint32_t* param_tbl,
    uint32_t sh1,
    uint32_t sh2,
    uint32_t X1,
    uint32_t X2,
    uint32_t Y,
    uint32_t mask) [[cpu]][[hc]]
{
  uint32_t X = (X1 & mask) ^ X2;
  uint32_t MAT;

  X ^= X << sh1;
  Y = X ^ (Y >> sh2);
  MAT = param_tbl[Y & 0x0f];
  return Y ^ MAT;
}

/**
 * The tempering function.
 *
 * @param[in] mtgp mtgp32 structure
 * @param[in] V the output value should be tempered.
 * @param[in] T the tempering helper value.
 * @return the tempered value.
 */
static
inline
uint32_t temper(uint32_t* temper_tbl, uint32_t V, uint32_t T) [[cpu]][[hc]]
{
  uint32_t MAT;

  T ^= T >> 16;
  T ^= T >> 8;
  MAT = temper_tbl[T & 0x0f];
  return V ^ MAT;
}

/**
 * The tempering and converting function.
 * By using the preset-ted table, converting to IEEE format
 * and tempering are done simultaneously.
 *
 * @param[in] V the output value should be tempered.
 * @param[in] T the tempering helper value.
 * @return the tempered and converted value.
 */
#if 0
static
inline
uint32_t temper_single(
  uint32_t* single_temper_tbl, uint32_t V, uint32_t T) [[cpu]][[hc]]
{
  uint32_t MAT;
  uint32_t r;

  T ^= T >> 16;
  T ^= T >> 8;
  MAT = single_temper_tbl[T & 0x0f]; // from group
  r = (V >> 9) ^ MAT;
  return r;
}
#endif
// in one workgroup
static
inline
unsigned int hiprand(
    const uint32_t* av_param_tbl,
    const uint32_t* av_temper_tbl,
    const uint32_t* av_sh1_tbl,
    const uint32_t* av_sh2_tbl,
    uint32_t* av_offset,
    const uint32_t* av_index,
    const uint32_t* av_pos_tbl,
    const uint32_t* av_mask,
    const uint32_t* av_d_status,
    const hc::tiled_index<1>& tidx) [[hc]]
{
  int groupId = tidx.tile[0];
  unsigned int t = tidx.local[0];
  unsigned int d = BLOCK_SIZE * 1 * 1;
  //assert( d <= 255);
  uint32_t* pParam_tbl = const_cast<uint32_t*>(&av_param_tbl[groupId * MTGP32_TS + 0 ]);
  uint32_t* pTemper_tbl = const_cast<uint32_t*>(&av_temper_tbl[groupId * MTGP32_TS + 0 ]);
  uint32_t* d_status = const_cast<uint32_t*>(&av_d_status[groupId * MTGP32_STATE_SIZE + 0 ]);
  uint32_t offset = av_offset[groupId];
  uint32_t sh1 = av_sh1_tbl[groupId];
  uint32_t sh2 = av_sh2_tbl[groupId];
  int index = av_index[groupId];
  int pos = av_pos_tbl[index];
  unsigned int r;
  unsigned int o;
  r = para_rec(pParam_tbl, sh1, sh2, d_status[(t + offset) & MTGP32_STATE_MASK],
           d_status[(t + offset + 1) & MTGP32_STATE_MASK],
           d_status[(t + offset + pos) & MTGP32_STATE_MASK],
           av_mask[0]);
  d_status[(t + offset + MTGP32_N) & MTGP32_STATE_MASK] = r;
  o = temper(pTemper_tbl, r, d_status[(t + offset + pos -1) & MTGP32_STATE_MASK]);
  tidx.barrier.wait();
  if (t == 0) {
    av_offset[groupId] = (av_offset[groupId] + d) & MTGP32_STATE_MASK;
  }
  tidx.barrier.wait();
  return o;
}

#define HcRAND_2POW32_INV (2.3283064e-10f)
#define HcRAND_SQRT2 (-1.4142135f)
#define HcRAND_SQRT2_DOUBLE (-1.4142135623730951)
static
inline
float _hiprand_uniform(unsigned int x) [[cpu]][[hc]]
{
  return x * HcRAND_2POW32_INV + (HcRAND_2POW32_INV/2.0f);
}

inline
float hiprand_uniform(
    const uint32_t* av_param_tbl,
    const uint32_t* av_temper_tbl,
    const uint32_t* av_sh1_tbl,
    const uint32_t* av_sh2_tbl,
    const uint32_t* av_offset,
    const uint32_t* av_index,
    const uint32_t* av_pos_tbl,
    const uint32_t* av_mask,
    const uint32_t* av_d_status,
    const hc::tiled_index<1>& tidx) [[hc]]
{
  unsigned int x = hiprand(
      av_param_tbl,
      av_temper_tbl,
      av_sh1_tbl,
      av_sh2_tbl,
      const_cast<uint32_t*>(av_offset),
      av_index,
      av_pos_tbl,
      av_mask,
      av_d_status,
      tidx);
  return _hiprand_uniform(x);
}
// http://hg.savannah.gnu.org/hgweb/octave/rev/cb85e836d035
static
inline
double do_erfcinv(float x, bool refine) [[cpu]][[hc]]
{
  // Coefficients of rational approximation.
  const double a[] =
    { -2.806989788730439e+01,  1.562324844726888e+02,
      -1.951109208597547e+02,  9.783370457507161e+01,
      -2.168328665628878e+01,  1.772453852905383e+00 };
  const double b[] =
    { -5.447609879822406e+01,  1.615858368580409e+02,
      -1.556989798598866e+02,  6.680131188771972e+01,
      -1.328068155288572e+01 };
  const double c[] =
    { -5.504751339936943e-03, -2.279687217114118e-01,
      -1.697592457770869e+00, -1.802933168781950e+00,
       3.093354679843505e+00,  2.077595676404383e+00 };
  const double d[] =
    {  7.784695709041462e-03,  3.224671290700398e-01,
       2.445134137142996e+00,  3.754408661907416e+00 };

  static const double spi2 =  8.862269254527579e-01; // sqrt(pi)/2.
  //static const double pi = 3.14159265358979323846;
  static const double pbreak = 0.95150;
  double y;

  // Select case.
  if (x <= 1+pbreak && x >= 1-pbreak) {
    // Middle region.
    const double q = 0.5*(1-x), r = q*q;
    const double yn =
        (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q;
    const double yd =
        ((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0;
    y = yn / yd;
  } else if (x < 2.0 && x > 0.0) {
    // Tail region.
    const double q =
        x < 1 ? hc::precise_math::sqrt(-2*hc::precise_math::log((double)0.5*x))
              : hc::precise_math::sqrt(-2*hc::precise_math::log((double)0.5*(2-x)));
    const double yn = ((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5];
    const double yd = (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0;
    y = yn / yd;
    if (x < 1-pbreak)
      y *= -1;
  } else if (x == 0.0) {
    return 0; //octave_Inf;
  } else if (x == 2.0) {
    return 0; //-octave_Inf;
  } else {
    return 1; //octave_NaN;
  }

  if (refine) {
    // One iteration of Halley's method gives full precision.
    double u = (erf(y) - (1-x)) * spi2 * exp (y*y);
    y -= u / (1 + y*u);
  }

 return y;
}


static
inline
float my_erfcinvf(float x) [[cpu]][[hc]]
{
  return do_erfcinv (x, false);
}

static
inline
float _hiprand_normal_icdf(unsigned int x) [[cpu]][[hc]]
{
  float s = HcRAND_SQRT2;
  // Mirror to avoid loss of precision
  if (x > 0x80000000UL) {
    x = 0xffffffffUL - x;
    s = -s;
  }
  float p = x * HcRAND_2POW32_INV + (HcRAND_2POW32_INV/2.0f);
  // p is in (0, 0.5], 2p is in (0, 1]
  return s * my_erfcinvf(2.0f * p);
}

static
inline
float hiprand_normal(
    const uint32_t* av_param_tbl,
    const uint32_t* av_temper_tbl,
    const uint32_t* av_sh1_tbl,
    const uint32_t* av_sh2_tbl,
    const uint32_t* av_offset,
    const uint32_t* av_index,
    const uint32_t* av_pos_tbl,
    const uint32_t* av_mask,
    const uint32_t* av_d_status,
    const hc::tiled_index<1>& tidx) [[hc]]
{
  unsigned int x = hiprand(
      av_param_tbl,
      av_temper_tbl,
      av_sh1_tbl,
      av_sh2_tbl,
      const_cast<uint32_t*>(av_offset),
      av_index,
      av_pos_tbl,
      av_mask,
      av_d_status,
      tidx);
  return _hiprand_normal_icdf(x);
}

static
inline
double hiprand_log_normal(
    const uint32_t* av_param_tbl,
    const uint32_t* av_temper_tbl,
    const uint32_t* av_sh1_tbl,
    const uint32_t* av_sh2_tbl,
    const uint32_t* av_offset,
    const uint32_t* av_index,
    const uint32_t* av_pos_tbl,
    const uint32_t* av_mask,
    const uint32_t* av_d_status,
    const hc::tiled_index<1>& tidx,
    double mean,
    double stddev) [[hc]]
{
  unsigned int x = hiprand(
      av_param_tbl,
      av_temper_tbl,
      av_sh1_tbl,
      av_sh2_tbl,
      const_cast<uint32_t*>(av_offset),
      av_index,
      av_pos_tbl,
      av_mask,
      av_d_status,
      tidx);
  return hc::precise_math::exp(mean + ((double)stddev * _hiprand_normal_icdf(x)));
}

// User defined wrappers
static
inline
void user_log_normal_kernel(
    hc::accelerator_view accl_view,
    HipRandStateMtgp32 *s,
    float* &av_result,
    double mean,
    double stddev)
{
  hc::accelerator accl = accl_view.get_accelerator();
  hc::AmPointerInfo resInfo(0, 0, 0, accl, 0, 0);
  hc::am_memtracker_getinfo(&resInfo, av_result);
  const int  size = resInfo._sizeBytes/sizeof(float);
  int rounded_size = DIVUP(size, BLOCK_SIZE) * BLOCK_SIZE;
  int blocks = std::min((int)DIVUP(size, BLOCK_SIZE), MAX_NUM_BLOCKS);
  hc::extent<1> ext(blocks*BLOCK_SIZE);
  hc::tiled_extent<1> t_ext = ext.tile(BLOCK_SIZE);
  const uint32_t* av_param_tbl = (s->param_tbl);
  const uint32_t* av_temper_tbl = (s->temper_tbl);
  const uint32_t* av_sh1_tbl = (s->sh1_tbl);
  const uint32_t* av_sh2_tbl = (s->sh2_tbl);
  const uint32_t* av_offset = (s->offset);
  const uint32_t* av_index = (s->index);
  const uint32_t* av_pos_tbl = (s->pos_tbl);
  const uint32_t* av_mask = (s->mask);
  const uint32_t* av_d_status = (s->d_status);

  hc::parallel_for_each(
      accl_view, t_ext, [=] (const hc::tiled_index<1>& tidx) [[hc]] {
    int threadId = tidx.global[0];
    int groupId = tidx.tile[0];
    if (groupId >= USER_GROUP_NUM)
      return;
    for (int i = threadId; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {
      double x = hiprand_log_normal(
          av_param_tbl,
          av_temper_tbl,
          av_sh1_tbl,
          av_sh2_tbl,
          av_offset,
          av_index,
          av_pos_tbl,
          av_mask,
          av_d_status,
          tidx,
          mean,
          stddev);
      if (i < size) {
        av_result[i] = x;
      }
    }
  }).wait();
}

// User defined wrappers
template <typename UnaryFunction>
inline
void user_uniform_kernel(
    hc::accelerator_view accl_view,
    HipRandStateMtgp32 *s,
    float* &av_result,
    UnaryFunction f)
{
  hc::accelerator accl = accl_view.get_accelerator();
  hc::AmPointerInfo resInfo(0, 0, 0, accl, 0, 0);
  hc::am_memtracker_getinfo(&resInfo, av_result);
  const int  size = resInfo._sizeBytes/sizeof(float);
  int rounded_size = DIVUP(size, BLOCK_SIZE) * BLOCK_SIZE;
  int blocks = std::min((int)DIVUP(size, BLOCK_SIZE), MAX_NUM_BLOCKS);
  hc::extent<1> ext(blocks*BLOCK_SIZE);
  hc::tiled_extent<1> t_ext = ext.tile(BLOCK_SIZE);
  const uint32_t* av_param_tbl = (s->param_tbl);
  const uint32_t* av_temper_tbl = (s->temper_tbl);
  const uint32_t* av_sh1_tbl = (s->sh1_tbl);
  const uint32_t* av_sh2_tbl = (s->sh2_tbl);
  const uint32_t* av_offset = (s->offset);
  const uint32_t* av_index = (s->index);
  const uint32_t* av_pos_tbl = (s->pos_tbl);
  const uint32_t* av_mask = (s->mask);
  const uint32_t* av_d_status = (s->d_status);
  hc::parallel_for_each(
      accl_view, t_ext, [=] (const hc::tiled_index<1>& tidx) [[hc]] {
    int threadId = tidx.global[0];
    int groupId = tidx.tile[0];
    if (groupId >= USER_GROUP_NUM)
      return;
    for (int i = threadId; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {
      float x = hiprand_uniform(
          av_param_tbl,
          av_temper_tbl,
          av_sh1_tbl,
          av_sh2_tbl,
          av_offset,
          av_index,
          av_pos_tbl,
          av_mask,
          av_d_status,
          tidx);
      if (i < size) {
        double y = f(x);
        av_result[i] = y;
      }
    }
  }).wait();
}

template <typename UnaryFunction>
void user_normal_kernel(
    hc::accelerator_view accl_view,
    HipRandStateMtgp32 *s,
    float* &av_result,
    UnaryFunction f)
{
  hc::accelerator accl = accl_view.get_accelerator();
  hc::AmPointerInfo resInfo(0, 0, 0, accl, 0, 0);
  hc::am_memtracker_getinfo(&resInfo, av_result);
  const int  size = resInfo._sizeBytes/sizeof(float);
  int rounded_size = DIVUP(size, BLOCK_SIZE) * BLOCK_SIZE;
  int blocks = std::min((int)DIVUP(size, BLOCK_SIZE), MAX_NUM_BLOCKS);
  hc::extent<1> ext(blocks*BLOCK_SIZE);
  hc::tiled_extent<1> t_ext = ext.tile(BLOCK_SIZE);
  const uint32_t* av_param_tbl = (s->param_tbl);
  const uint32_t* av_temper_tbl = (s->temper_tbl);
  const uint32_t* av_sh1_tbl = (s->sh1_tbl);
  const uint32_t* av_sh2_tbl = (s->sh2_tbl);
  const uint32_t* av_offset = (s->offset);
  const uint32_t* av_index = (s->index);
  const uint32_t* av_pos_tbl = (s->pos_tbl);
  const uint32_t* av_mask = (s->mask);
  const uint32_t* av_d_status = (s->d_status);
  hc::parallel_for_each(
      accl_view, t_ext, [=] (const hc::tiled_index<1>& tidx) [[hc]] {
    int threadId = tidx.global[0];
    int groupId = tidx.tile[0];
    if (groupId >= USER_GROUP_NUM)
      return;
    for (int i = threadId; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {
      float x = hiprand_normal(
          av_param_tbl,
          av_temper_tbl,
          av_sh1_tbl,
          av_sh2_tbl,
          av_offset,
          av_index,
          av_pos_tbl,
          av_mask,
          av_d_status,
          tidx);
      if (i < size) {
        double y = f(x);
        av_result[i] = y;
      }
    }
  }).wait();
}
