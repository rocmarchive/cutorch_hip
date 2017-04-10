#include <hc.hpp>
#include "hiprand_mtgp32.h"

using namespace hc;

//for debug purpose
//#define DEBUG_HcRAND
#ifdef DEBUG_HcRAND
#include <iostream>
template <typename T>
void VerifyData(T expected, T actual, const char* s) {
  if (expected != actual) {
    std::cout << s << " expected=" << expected << "  actual=" << actual << std::endl;
    exit(1);
  }
}
// Host APIs
/**
 * This function initializes the internal state array with a 32-bit
 * integer seed. The allocated memory should be freed by calling
 * mtgp32_free(). \b para should be one of the elements in the
 * parameter table (mtgp32-param-ref.c).
 *
 * @param[out] array MTGP internal status vector.
 * @param[in] para parameter structure
 * @param[in] seed a 32-bit integer used as the seed.
 */
void mtgp32_init_state(uint32_t array[], const mtgp32_params_fast_t* para,
                       uint32_t seed) {
  int i;
  int size = para->mexp / 32 + 1;
  uint32_t hidden_seed;
  uint32_t tmp;
  hidden_seed = para->tbl[4] ^ (para->tbl[8] << 16);
  tmp = hidden_seed;
  tmp += tmp >> 16;
  tmp += tmp >> 8;
  memset(array, tmp & 0xff, sizeof(uint32_t) * size);
  array[0] = seed;
  array[1] = hidden_seed;

  for (i = 1; i < size; i++) {
    array[i] ^= UINT32_C(1812433253) * (array[i - 1]
                                        ^ (array[i - 1] >> 30)) + i;
  }

  for (i = size; i < MTGP32_STATE_SIZE; i++) {
    array[i] = 0;
  }
}
#endif


// C-style
void HipRandStateMtgp32_init(hc::accelerator_view accl_view,
                             HipRandStateMtgp32* s) {
  // kernel params
  // get target accelerator
  hc::accelerator accl = accl_view.get_accelerator();
  uint32_t* arrP = hc::am_alloc((HcRAND_GROUP_NUM * MTGP32_TS * sizeof(uint32_t)), accl, 0); 
  s->param_tbl = arrP;
  uint32_t* arrT = hc::am_alloc((HcRAND_GROUP_NUM * MTGP32_TS * sizeof(uint32_t)), accl, 0); 
  s->temper_tbl = arrT;
  uint32_t* arrS = hc::am_alloc((HcRAND_GROUP_NUM * MTGP32_TS * sizeof(uint32_t)), accl, 0); 
  s->single_temper_tbl = arrS;
  uint32_t* arrPos = hc::am_alloc(HcRAND_GROUP_NUM * sizeof(uint32_t), accl, 0); 
  s->pos_tbl = arrPos;
  uint32_t* arrSh1 = hc::am_alloc(HcRAND_GROUP_NUM * sizeof(uint32_t), accl, 0); 
  s->sh1_tbl = arrSh1;
  uint32_t* arrSh2 = hc::am_alloc(HcRAND_GROUP_NUM * sizeof(uint32_t), accl, 0); 
  s->sh2_tbl = arrSh2;
  uint32_t* arrMask = hc::am_alloc(1 * sizeof(uint32_t), accl, 0); 
  s->mask = arrMask;
  // Redundant member
  uint32_t* arrMexp = hc::am_alloc((HcRAND_GROUP_NUM * sizeof(uint32_t)), accl, 0); 
  s->mexp_tbl = arrMexp;
  // states
  uint32_t* arrStatus = hc::am_alloc((USER_GROUP_NUM * MTGP32_STATE_SIZE * sizeof(uint32_t)), accl, 0); 
  s->d_status = arrStatus;
  uint32_t* arrOffset = hc::am_alloc((USER_GROUP_NUM * sizeof(uint32_t)), accl, 0); 
  s->offset = arrOffset;
  uint32_t* arrIndex = hc::am_alloc((USER_GROUP_NUM * sizeof(uint32_t)), accl, 0); 
  s->index = arrIndex;
}

#define FREE_MEMBER(s, member) \
  if (s->member) {             \
    hc::am_free(s->member);  \
    s->member = NULL;          \
  }
// TODO: Big memory leak and cause device buffer allocation fail if exits abnormally
void HipRandStateMtgp32_release(HipRandStateMtgp32* s) {
  FREE_MEMBER(s, param_tbl);
  FREE_MEMBER(s, temper_tbl);
  FREE_MEMBER(s, single_temper_tbl);
  FREE_MEMBER(s, pos_tbl);
  FREE_MEMBER(s, sh1_tbl);
  FREE_MEMBER(s, sh2_tbl);
  FREE_MEMBER(s, mask);
  FREE_MEMBER(s, mexp_tbl);
  FREE_MEMBER(s, d_status);
  FREE_MEMBER(s, offset);
  FREE_MEMBER(s, index);
}

// The following are device APIs

// Copy param constants onto device
int mtgp32_init_params_kernel(hc::accelerator_view accl_view,
                              mtgp32_params_fast_t* params,
                              HipRandStateMtgp32*& s) {
  const uint32_t* av_param_tbl = (s->param_tbl);
  const uint32_t* av_temper_tbl = (s->temper_tbl);
  const uint32_t* av_single_temper_tbl = (s->single_temper_tbl);
  const uint32_t* av_pos_tbl = (s->pos_tbl);
  const uint32_t* av_sh1_tbl = (s->sh1_tbl);
  const uint32_t* av_sh2_tbl = (s->sh2_tbl);
  const uint32_t* av_mask = (s->mask);
  const uint32_t* av_mexp_tbl = (s->mexp_tbl);
  // Prepare data source
  uint32_t vec_param[HcRAND_GROUP_NUM * MTGP32_TS] = {0x0};
  uint32_t vec_temper[HcRAND_GROUP_NUM * MTGP32_TS] = {0x0};
  uint32_t vec_single_temper[HcRAND_GROUP_NUM * MTGP32_TS] = {0x0};
  uint32_t vec_pos[HcRAND_GROUP_NUM] = {0x0};
  uint32_t vec_sh1[HcRAND_GROUP_NUM] = {0x0};
  uint32_t vec_sh2[HcRAND_GROUP_NUM] = {0x0};
  uint32_t vec_mexp[HcRAND_GROUP_NUM] = {0x0};

  for (int i = 0; i < HcRAND_GROUP_NUM; i++) {
    vec_pos[i] = params[i].pos;
    vec_sh1[i] = params[i].sh1;
    vec_sh2[i] = params[i].sh2;
    vec_mexp[i] = params[i].mexp;

    for (int j = 0; j < MTGP32_TS; j++) {
      vec_param[i * MTGP32_TS + j] = params[i].tbl[j];
      vec_temper[i * MTGP32_TS + j] = params[i].tmp_tbl[j];
      vec_single_temper[i * MTGP32_TS + j] = params[i].flt_tmp_tbl[j];
    }
  }

  accl_view.copy(&params[0].mask, (void*)av_mask, 1 * sizeof(uint32_t));
  accl_view.copy(vec_param,(void*)av_param_tbl,  HcRAND_GROUP_NUM * MTGP32_TS * sizeof(uint32_t));
  accl_view.copy(vec_temper, (void*)av_temper_tbl, HcRAND_GROUP_NUM * MTGP32_TS * sizeof(uint32_t));
  accl_view.copy(vec_single_temper, (void*)av_single_temper_tbl, HcRAND_GROUP_NUM * MTGP32_TS * sizeof(uint32_t));
  accl_view.copy(vec_pos, (void*)av_pos_tbl, HcRAND_GROUP_NUM * sizeof(uint32_t));
  accl_view.copy(vec_sh1, (void*)av_sh1_tbl, HcRAND_GROUP_NUM * sizeof(uint32_t));
  accl_view.copy(vec_sh2, (void*)av_sh2_tbl, HcRAND_GROUP_NUM * sizeof(uint32_t));
  accl_view.copy(vec_mexp, (void*)av_mexp_tbl, HcRAND_GROUP_NUM * sizeof(uint32_t));

  return 0;
}

// Initialize HipRandStateMtgp32 by seed
int mtgp32_init_seed_kernel(hc::accelerator_view accl_view,
                            HipRandStateMtgp32* s, unsigned long seed) {
  seed = seed ^ (seed >> 32);
  int nGroups = USER_GROUP_NUM;
  const uint32_t* av_param_tbl = (s->param_tbl);
  uint32_t* av_offset = (s->offset);
  uint32_t* av_index = (s->index);
  const uint32_t* av_mexp_tbl = (s->mexp_tbl);
  uint32_t* av_d_status = (s->d_status);
  hc::extent<1> ext(nGroups);
  hc::parallel_for_each(accl_view, ext, [ = ] (hc::index<1> idx) __attribute__((hc, cpu)) {
    const int id = idx[0];

    if (id >= nGroups) {
      return;
    }

    uint32_t* status = &av_d_status[id * MTGP32_STATE_SIZE + 0];
    uint32_t mexp = av_mexp_tbl[id];
    int size = mexp / 32 + 1;
    // Initialize state
    int i;
    uint32_t hidden_seed;
    uint32_t tmp;
    hidden_seed = av_param_tbl[id * MTGP32_TS + 4] ^ (av_param_tbl[id * MTGP32_TS + 8] << 16);
    tmp = hidden_seed;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    tmp = tmp & 0xff;
    tmp |= tmp << 8;
    tmp |= tmp << 16;

    for (i = 0; i < size; i++) {
      status[i] = tmp;
    }

    status[0] = (unsigned int) seed + 1 + id;
    status[1] = hidden_seed;

    for (i = 1; i < size; i++) {
      status[i] ^= UINT32_C(1812433253) * (status[i - 1] ^ (status[i - 1] >> 30)) + i;
    }

    av_offset[id] = 0;
    av_index[id] = id;
  }).wait();
  return 0;
}
