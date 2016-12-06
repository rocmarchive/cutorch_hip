#include "hcblaslib.h"
#include <assert.h>
#include <vector>
#include <hc.hpp>
#include <iostream>
#include "hc_short_vector.hpp"
#include <hc_math.hpp>
using namespace hc;
using namespace hc::short_vector;
#define TILE_SZ_A 64
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A/TILE_SZ_B) 
#define TOTMICROTILEPROD (TILESIZE*TILESIZE*MICROTILESIZE)
#define MICROTILEPROD (TILESIZE*MICROTILESIZE)
#define BANKMICROTILESIZE (TILESIZE*MICROTILESIZE+1)

#define  M1x1(offset)			\
            rAreal[0][0] = lAreal[offA + 0];	\
            rBreal[0][0] = lBreal[offB + 0];	\
            rAimg[0][0] = lAimg[offA + 0];	\
            rBimg[0][0] = lBimg[offB + 0];	\
            offA += offset;			\
            offB += offset;			\
            rCreal[0][0] = rCreal[0][0] + (rAreal[0][0] *rBreal[0][0]) - (rAimg[0][0] * rBimg[0][0]) ; \
            rCimg[0][0] = rCimg[0][0] + (rAreal[0][0] *rBimg[0][0]) + (rAimg[0][0] * rBreal[0][0]) ; \

#define  MS1x1(offset)			\
            for(int iter = 0; iter < STEPSIZE/TILESIZE; ++iter) \
            {\
              rAreal[0][iter] = lAreal[offA + (TILESIZE * TILESIZE) * iter];	\
              rBreal[0][iter] = lBreal[offB + (TILESIZE * TILESIZE) * iter];	\
              rAimg[0][iter] = lAimg[offA + (TILESIZE * TILESIZE) * iter];	\
              rBimg[0][iter] = lBimg[offB + (TILESIZE * TILESIZE) * iter];	\
              rCreal[0][0] = rCreal[0][0] + (rAreal[0][iter] *rBreal[0][iter]) - (rAimg[0][iter] * rBimg[0][iter]) ; \
              rCimg[0][0] = rCimg[0][0] + (rAreal[0][iter] *rBimg[0][iter]) + (rAimg[0][iter] * rBreal[0][iter]) ; \
            }\
            offA += offset;			\
            offB += offset;			\

#define  MTS                                                                                \
           for(int iter = 0; iter < MICROTILESIZE ; iter++)                                 \
           {                                                                                \
             rAreal[0][iter] = lAreal[offA + (iter * TILESIZE)];                            \
             rBreal[0][iter] = lBreal[offB + (iter * TILESIZE)];                            \
             rAimg[0][iter] = lAimg[offA + (iter * TILESIZE)];                              \
             rBimg[0][iter] = lBimg[offB + (iter * TILESIZE)];                              \
           }                                                                                \
           for(int rowIndex = 0; rowIndex < MICROTILESIZE ; rowIndex++)                     \
           {                                                                                \
           for(int colIndex = 0; colIndex < MICROTILESIZE ; colIndex++)                     \
           {                                                                                \
           rCreal[rowIndex][colIndex] = (rAreal[0][rowIndex] * rBreal[0][colIndex] -        \
	                                 rAimg[0][rowIndex] * rBimg[0][colIndex]) +         \
                                         rCreal[rowIndex][colIndex];                        \
           rCimg[rowIndex][colIndex] = (rAreal[0][rowIndex] * rBimg[0][colIndex] +          \
                                        rAimg[0][rowIndex] * rBreal[0][colIndex]) +         \
                                        rCimg[rowIndex][colIndex];                          \
           }                                                                                \
           }                                                                                \
           offA += BANKMICROTILESIZE;                                                       \
           offB += BANKMICROTILESIZE;                                                       \


/* 
*  ZGEMM kernels - column major Order
*/
hcblasStatus zgemm_NoTransAB(hc::accelerator_view accl_view,
		             double_2 *A, long aOffset,
                             double_2 *B, long bOffset,
                             double_2 *C, long cOffset,
                             int M, int N, int K, int lda, int ldb, int ldc,
                             double_2 alpha, double_2 beta);

hcblasStatus zgemm_NoTransA(hc::accelerator_view accl_view,
		            double_2 *A, long aOffset,
                            double_2 *B, long bOffset,
                            double_2 *C, long cOffset,
                            int M, int N, int K, int lda, int ldb, int ldc,
                            double_2 alpha, double_2 beta);

hcblasStatus zgemm_NoTransB(hc::accelerator_view accl_view,
		            double_2 *A, long aOffset,
                            double_2 *B, long bOffset,
                            double_2 *C, long cOffset,
                            int M, int N, int K, int lda, int ldb, int ldc,
                            double_2 alpha, double_2 beta);

hcblasStatus zgemm_TransAB(hc::accelerator_view accl_view,
		           double_2 *A, long aOffset,
                           double_2 *B, long bOffset,
                           double_2 *C, long cOffset,
                           int M, int N, int K, int lda, int ldb, int ldc,
                           double_2 alpha, double_2 beta);


/*
* ZGEMM Kernels for Batch processing in column major order
*/

hcblasStatus zgemm_NoTransAB(hc::accelerator_view accl_view,
		             double_2 *A, long aOffset, long A_batchOffset,
                             double_2 *B, long bOffset, long B_batchOffset,
                             double_2 *C, long cOffset, long C_batchOffset,
                             int M, int N, int K, int lda, int ldb, int ldc,
                             double_2 alpha, double_2 beta, int batchSize);

hcblasStatus zgemm_NoTransA(hc::accelerator_view accl_view,
		            double_2 *A, long aOffset, long A_batchOffset,
                            double_2 *B, long bOffset, long B_batchOffset,
                            double_2 *C, long cOffset, long C_batchOffset,
                            int M, int N, int K, int lda, int ldb, int ldc,
                            double_2 alpha, double_2 beta, int batchSize);

hcblasStatus zgemm_NoTransB(hc::accelerator_view accl_view,
		            double_2 *A, long aOffset, long A_batchOffset,
                            double_2 *B, long bOffset, long B_batchOffset,
                            double_2 *C, long cOffset, long C_batchOffset,
                            int M, int N, int K, int lda, int ldb, int ldc,
                            double_2 alpha, double_2 beta, int batchSize);

hcblasStatus zgemm_TransAB(hc::accelerator_view accl_view,
		           double_2 *A, long aOffset, long A_batchOffset,
                           double_2 *B, long bOffset, long B_batchOffset,
                           double_2 *C, long cOffset, long C_batchOffset,
                           int M, int N, int K, int lda, int ldb, int ldc,
                           double_2 alpha, double_2 beta, int batchSize);


/*
* ZGEMM Kernels - Row major order
*/

hcblasStatus zgemm_NoTransAB_rMajor(hc::accelerator_view accl_view,
                                    double_2 *A, long aOffset,
                                    double_2 *B, long bOffset,
                                    double_2 *C, long cOffset,
                                    int M, int N, int K, int lda, int ldb, int ldc,
                                    double_2 alpha, double_2 beta);

hcblasStatus zgemm_NoTransA_rMajor(hc::accelerator_view accl_view,
                                   double_2 *A, long aOffset,
                                   double_2 *B, long bOffset,
                                   double_2 *C, long cOffset,
                                   int M, int N, int K, int lda, int ldb, int ldc,
                                   double_2 alpha, double_2 beta);

hcblasStatus zgemm_NoTransB_rMajor(hc::accelerator_view accl_view,
                                   double_2 *A, long aOffset,
                                   double_2 *B, long bOffset,
                                   double_2 *C, long cOffset,
                                   int M, int N, int K, int lda, int ldb, int ldc,
                                   double_2 alpha, double_2 beta);

hcblasStatus zgemm_TransAB_rMajor(hc::accelerator_view accl_view,
                                  double_2 *A, long aOffset,
                                  double_2 *B, long bOffset,
                                  double_2 *C, long cOffset,
                                  int M, int N, int K, int lda, int ldb, int ldc,
                                  double_2 alpha, double_2 beta);


/*
* ZGEMM Kernels for Batch-processing in Row major order
*/

hcblasStatus zgemm_NoTransAB_rMajor(hc::accelerator_view accl_view,
                                    double_2 *A, long aOffset, long A_batchOffset,
                                    double_2 *B, long bOffset, long B_batchOffset,
                                    double_2 *C, long cOffset, long C_batchOffset,
                                    int M, int N, int K, int lda, int ldb, int ldc,
                                    double_2 alpha, double_2 beta, int batchSize);

hcblasStatus zgemm_NoTransA_rMajor(hc::accelerator_view accl_view,
                                   double_2 *A, long aOffset, long A_batchOffset,
                                   double_2 *B, long bOffset, long B_batchOffset,
                                   double_2 *C, long cOffset, long C_batchOffset,
                                   int M, int N, int K, int lda, int ldb, int ldc,
                                   double_2 alpha, double_2 beta, int batchSize);

hcblasStatus zgemm_NoTransB_rMajor(hc::accelerator_view accl_view,
                                   double_2 *A, long aOffset, long A_batchOffset,
                                   double_2 *B, long bOffset, long B_batchOffset,
                                   double_2 *C, long cOffset, long C_batchOffset,
                                   int M, int N, int K, int lda, int ldb, int ldc,
                                   double_2 alpha, double_2 beta, int batchSize);

hcblasStatus zgemm_TransAB_rMajor(hc::accelerator_view accl_view,
                                  double_2 *A, long aOffset, long A_batchOffset,
                                  double_2 *B, long bOffset, long B_batchOffset,
                                  double_2 *C, long cOffset, long C_batchOffset,
                                  int M, int N, int K, int lda, int ldb, int ldc,
                                  double_2 alpha, double_2 beta, int batchSize);

