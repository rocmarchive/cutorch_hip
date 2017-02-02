#include "hcblaslib.h"
#include <hc.hpp>
#include <hc_math.hpp>
using namespace hc;
#define TILESIZE 8
#define STEPSIZE 8
#define MICROTILESIZE 2
#define BANKTILESIZE (TILESIZE + 1)
#define STEPTILERATIO STEPSIZE/TILESIZE 
#define STEPTILEPROD STEPSIZE*TILESIZE
#define BANKNUMTILEELMTS TILESIZE*BANKTILESIZE
#define NUMTILEELMTS TILESIZE*TILESIZE
#define TOTMICROTILEPROD (TILESIZE*TILESIZE*MICROTILESIZE)
#define MICROTILEPROD (TILESIZE*MICROTILESIZE)
#define BANKMICROTILESIZE (TILESIZE*MICROTILESIZE+1)

#define  MS1x1(offsetA, offsetB)			\
            for(int iter = 0; iter < STEPSIZE/TILESIZE; ++iter) \
            {\
              rA[0][iter] = lA[offA + (TILESIZE * TILESIZE) * iter];	\
              rB[0][iter] = lB[offB + (TILESIZE * TILESIZE) * iter];	\
              rC[0][0] +=rA[0][iter] *rB[0][iter]; \
            }\
            offA += offsetA;			\
            offB += offsetB;			\

#define  MS1x1_NOBANK(offset)			\
            for(int iter = 0; iter < STEPTILERATIO; ++iter) \
            {\
              rA[0][iter] = lA[offA + (BANKNUMTILEELMTS) * iter];	\
              rB[0][iter] = lB[offB + (BANKNUMTILEELMTS) * iter];	\
              rC[0][0] +=rA[0][iter] *rB[0][iter]; \
            }\
            offA += offset;			\
            offB += offset;			\

#define  MTS                                                                \
           for(int iter = 0; iter < MICROTILESIZE ; iter++)                 \
           {                                                                \
             rA[0][iter] = lA[offA + (iter * TILESIZE)];                    \
             rB[0][iter] = lB[offB + (iter * TILESIZE)];                    \
           }                                                                \
           for(int rowIndex = 0; rowIndex < MICROTILESIZE ; rowIndex++)     \
           {                                                                \
           for(int colIndex = 0; colIndex < MICROTILESIZE ; colIndex++)     \
           {                                                                \
           rC[rowIndex][colIndex] = rA[0][rowIndex] * rB[0][colIndex] +     \
                                    rC[rowIndex][colIndex];                 \
           }                                                                \
           }                                                                \
           offA += (MICROTILESIZE * TILESIZE);                              \
           offB += (MICROTILESIZE * TILESIZE);                              \

#define  MTS_NOBANK                                                         \
           for(int iter = 0; iter < MICROTILESIZE ; iter++)                 \
           {                                                                \
             rA[0][iter] = lA[offA + (iter * TILESIZE)];                    \
             rB[0][iter] = lB[offB + (iter * TILESIZE)];                    \
           }                                                                \
           for(int rowIndex = 0; rowIndex < MICROTILESIZE ; rowIndex++)     \
           {                                                                \
           for(int colIndex = 0; colIndex < MICROTILESIZE ; colIndex++)     \
           {                                                                \
           rC[rowIndex][colIndex] = rA[0][rowIndex] * rB[0][colIndex] +     \
                                    rC[rowIndex][colIndex];                 \
           }                                                                \
           }                                                                \
           offA += BANKMICROTILESIZE;                              \
           offB += BANKMICROTILESIZE;                              \




#define  MTS_AB                                                                \
           for (int iter = 0; iter < MICROTILESIZE_A ; iter++) {               \
             rA[0][iter] = lA[offA + (iter * TILESIZE_A)];                     \
           }                                                                   \
           for (int iter = 0; iter < MICROTILESIZE_B ; iter++) {               \
             rB[0][iter] = lB[offB + (iter * TILESIZE_B)];                     \
           }                                                                   \
           for (int rowIndex = 0; rowIndex < MICROTILESIZE_A ; rowIndex++) {   \
             for (int colIndex = 0; colIndex < MICROTILESIZE_B ; colIndex++) { \
               rC[rowIndex][colIndex] = rA[0][rowIndex] * rB[0][colIndex] +    \
                                        rC[rowIndex][colIndex];                \
             }                                                                 \
           }                                                                   \
           offA += (MICROTILESIZE_A * TILESIZE_A);                             \
           offB += (MICROTILESIZE_B * TILESIZE_B);                             \
	/* 
*  SGEMM kernels - column major Order
*/
#define mad(a, b, c) a * b + c

#define  M2x2                            \
             rA[0][0] = lA[offA + 0];          \
             rA[0][1] = lA[offA + 16];         \
             rB[0][0] = lB[offB + 0];          \
             rB[0][1] = lB[offB + 16];         \
             offA += 33;                       \
             offB += 33;  \
            rC[0][0]=mad(rA[0][0],rB[0][0],rC[0][0]); \
            rC[1][0]=mad(rA[0][1],rB[0][0],rC[1][0]); \
            rC[0][1]=mad(rA[0][0],rB[0][1],rC[0][1]); \
            rC[1][1]=mad(rA[0][1],rB[0][1],rC[1][1]); \

#define  M2x2_MB                            \
             rA[0][0] = lA[offA + 0 * 2 + 0];          \
             rA[0][1] = lA[offA + 0 * 2 + 1];          \
             rA[0][2] = lA[offA + 16 * 2 + 0];         \
             rA[0][3] = lA[offA + 16 * 2 + 1];         \
             rB[0][0] = lB[offB + 0 * 2 + 0];          \
             rB[0][1] = lB[offB + 0 * 2 + 1];          \
             rB[0][2] = lB[offB + 16 * 2 + 0];         \
             rB[0][3] = lB[offB + 16 * 2 + 1];         \
             offA += 32 * 2 + 1;                       \
             offB += 32 * 2 + 1;  \
            rC[0][0]=mad(rA[0][0],rB[0][0],rC[0][0]); \
            rC[1][0]=mad(rA[0][1],rB[0][0],rC[1][0]); \
            rC[2][0]=mad(rA[0][2],rB[0][0],rC[2][0]); \
            rC[3][0]=mad(rA[0][3],rB[0][0],rC[3][0]); \
            rC[0][1]=mad(rA[0][0],rB[0][1],rC[0][1]); \
            rC[1][1]=mad(rA[0][1],rB[0][1],rC[1][1]); \
            rC[2][1]=mad(rA[0][2],rB[0][1],rC[2][1]); \
            rC[3][1]=mad(rA[0][3],rB[0][1],rC[3][1]); \
            rC[0][2]=mad(rA[0][0],rB[0][2],rC[0][2]); \
            rC[1][2]=mad(rA[0][1],rB[0][2],rC[1][2]); \
            rC[2][2]=mad(rA[0][2],rB[0][2],rC[2][2]); \
            rC[3][2]=mad(rA[0][3],rB[0][2],rC[3][2]); \
            rC[0][3]=mad(rA[0][0],rB[0][3],rC[0][3]); \
            rC[1][3]=mad(rA[0][1],rB[0][3],rC[1][3]); \
            rC[2][3]=mad(rA[0][2],rB[0][3],rC[2][3]); \
            rC[3][3]=mad(rA[0][3],rB[0][3],rC[3][3]); \

#define  M4x4                            \
             rA[0][0] = lA[offA + 0];          \
             rA[0][1] = lA[offA + 16];         \
             rA[0][2] = lA[offA + 32];         \
             rA[0][3] = lA[offA + 48];         \
             rB[0][0] = lB[offB + 0];          \
             rB[0][1] = lB[offB + 16];         \
             rB[0][2] = lB[offB + 32];         \
             rB[0][3] = lB[offB + 48];         \
             offA += 65;                       \
             offB += 65;  \
            rC[0][0]=mad(rA[0][0],rB[0][0],rC[0][0]); \
            rC[1][0]=mad(rA[0][1],rB[0][0],rC[1][0]); \
            rC[2][0]=mad(rA[0][2],rB[0][0],rC[2][0]); \
            rC[3][0]=mad(rA[0][3],rB[0][0],rC[3][0]); \
            rC[0][1]=mad(rA[0][0],rB[0][1],rC[0][1]); \
            rC[1][1]=mad(rA[0][1],rB[0][1],rC[1][1]); \
            rC[2][1]=mad(rA[0][2],rB[0][1],rC[2][1]); \
            rC[3][1]=mad(rA[0][3],rB[0][1],rC[3][1]); \
            rC[0][2]=mad(rA[0][0],rB[0][2],rC[0][2]); \
            rC[1][2]=mad(rA[0][1],rB[0][2],rC[1][2]); \
            rC[2][2]=mad(rA[0][2],rB[0][2],rC[2][2]); \
            rC[3][2]=mad(rA[0][3],rB[0][2],rC[3][2]); \
            rC[0][3]=mad(rA[0][0],rB[0][3],rC[0][3]); \
            rC[1][3]=mad(rA[0][1],rB[0][3],rC[1][3]); \
            rC[2][3]=mad(rA[0][2],rB[0][3],rC[2][3]); \
            rC[3][3]=mad(rA[0][3],rB[0][3],rC[3][3]); \

#define  M4x4_MB                            \
             rA[0][0] = lA[offA + 0 * 2 + 0];          \
             rA[0][1] = lA[offA + 0 * 2 + 1];          \
             rA[0][2] = lA[offA + 16 * 2 + 0];         \
             rA[0][3] = lA[offA + 16 * 2 + 1];         \
             rA[0][4] = lA[offA + 32 * 2 + 0];         \
             rA[0][5] = lA[offA + 32 * 2 + 1];         \
             rA[0][6] = lA[offA + 48 * 2 + 0];         \
             rA[0][7] = lA[offA + 48 * 2 + 1];         \
             rB[0][0] = lB[offB + 0 * 2 + 0];          \
             rB[0][1] = lB[offB + 0 * 2 + 1];          \
             rB[0][2] = lB[offB + 16 * 2 + 0];         \
             rB[0][3] = lB[offB + 16 * 2 + 1];         \
             rB[0][4] = lB[offB + 32 * 2 + 0];         \
             rB[0][5] = lB[offB + 32 * 2 + 1];         \
             rB[0][6] = lB[offB + 48 * 2 + 0];         \
             rB[0][7] = lB[offB + 48 * 2 + 1];         \
             offA += 64 * 2 + 1;                       \
             offB += 64 * 2 + 1;  \
            rC[0][0]=mad(rA[0][0],rB[0][0],rC[0][0]); \
            rC[1][0]=mad(rA[0][1],rB[0][0],rC[1][0]); \
            rC[2][0]=mad(rA[0][2],rB[0][0],rC[2][0]); \
            rC[3][0]=mad(rA[0][3],rB[0][0],rC[3][0]); \
            rC[4][0]=mad(rA[0][4],rB[0][0],rC[4][0]); \
            rC[5][0]=mad(rA[0][5],rB[0][0],rC[5][0]); \
            rC[6][0]=mad(rA[0][6],rB[0][0],rC[6][0]); \
            rC[7][0]=mad(rA[0][7],rB[0][0],rC[7][0]); \
            rC[0][1]=mad(rA[0][0],rB[0][1],rC[0][1]); \
            rC[1][1]=mad(rA[0][1],rB[0][1],rC[1][1]); \
            rC[2][1]=mad(rA[0][2],rB[0][1],rC[2][1]); \
            rC[3][1]=mad(rA[0][3],rB[0][1],rC[3][1]); \
            rC[4][1]=mad(rA[0][4],rB[0][1],rC[4][1]); \
            rC[5][1]=mad(rA[0][5],rB[0][1],rC[5][1]); \
            rC[6][1]=mad(rA[0][6],rB[0][1],rC[6][1]); \
            rC[7][1]=mad(rA[0][7],rB[0][1],rC[7][1]); \
            rC[0][2]=mad(rA[0][0],rB[0][2],rC[0][2]); \
            rC[1][2]=mad(rA[0][1],rB[0][2],rC[1][2]); \
            rC[2][2]=mad(rA[0][2],rB[0][2],rC[2][2]); \
            rC[3][2]=mad(rA[0][3],rB[0][2],rC[3][2]); \
            rC[4][2]=mad(rA[0][4],rB[0][2],rC[4][2]); \
            rC[5][2]=mad(rA[0][5],rB[0][2],rC[5][2]); \
            rC[6][2]=mad(rA[0][6],rB[0][2],rC[6][2]); \
            rC[7][2]=mad(rA[0][7],rB[0][2],rC[7][2]); \
            rC[0][3]=mad(rA[0][0],rB[0][3],rC[0][3]); \
            rC[1][3]=mad(rA[0][1],rB[0][3],rC[1][3]); \
            rC[2][3]=mad(rA[0][2],rB[0][3],rC[2][3]); \
            rC[3][3]=mad(rA[0][3],rB[0][3],rC[3][3]); \
            rC[4][3]=mad(rA[0][4],rB[0][3],rC[4][3]); \
            rC[5][3]=mad(rA[0][5],rB[0][3],rC[5][3]); \
            rC[6][3]=mad(rA[0][6],rB[0][3],rC[6][3]); \
            rC[7][3]=mad(rA[0][7],rB[0][3],rC[7][3]); \
            rC[0][4]=mad(rA[0][0],rB[0][4],rC[0][4]); \
            rC[1][4]=mad(rA[0][1],rB[0][4],rC[1][4]); \
            rC[2][4]=mad(rA[0][2],rB[0][4],rC[2][4]); \
            rC[3][4]=mad(rA[0][3],rB[0][4],rC[3][4]); \
            rC[4][4]=mad(rA[0][4],rB[0][4],rC[4][4]); \
            rC[5][4]=mad(rA[0][5],rB[0][4],rC[5][4]); \
            rC[6][4]=mad(rA[0][6],rB[0][4],rC[6][4]); \
            rC[7][4]=mad(rA[0][7],rB[0][4],rC[7][4]); \
            rC[0][5]=mad(rA[0][0],rB[0][5],rC[0][5]); \
            rC[1][5]=mad(rA[0][1],rB[0][5],rC[1][5]); \
            rC[2][5]=mad(rA[0][2],rB[0][5],rC[2][5]); \
            rC[3][5]=mad(rA[0][3],rB[0][5],rC[3][5]); \
            rC[4][5]=mad(rA[0][4],rB[0][5],rC[4][5]); \
            rC[5][5]=mad(rA[0][5],rB[0][5],rC[5][5]); \
            rC[6][5]=mad(rA[0][6],rB[0][5],rC[6][5]); \
            rC[7][5]=mad(rA[0][7],rB[0][5],rC[7][5]); \
            rC[0][6]=mad(rA[0][0],rB[0][6],rC[0][6]); \
            rC[1][6]=mad(rA[0][1],rB[0][6],rC[1][6]); \
            rC[2][6]=mad(rA[0][2],rB[0][6],rC[2][6]); \
            rC[3][6]=mad(rA[0][3],rB[0][6],rC[3][6]); \
            rC[4][6]=mad(rA[0][4],rB[0][6],rC[4][6]); \
            rC[5][6]=mad(rA[0][5],rB[0][6],rC[5][6]); \
            rC[6][6]=mad(rA[0][6],rB[0][6],rC[6][6]); \
            rC[7][6]=mad(rA[0][7],rB[0][6],rC[7][6]); \
            rC[0][7]=mad(rA[0][0],rB[0][7],rC[0][7]); \
            rC[1][7]=mad(rA[0][1],rB[0][7],rC[1][7]); \
            rC[2][7]=mad(rA[0][2],rB[0][7],rC[2][7]); \
            rC[3][7]=mad(rA[0][3],rB[0][7],rC[3][7]); \
            rC[4][7]=mad(rA[0][4],rB[0][7],rC[4][7]); \
            rC[5][7]=mad(rA[0][5],rB[0][7],rC[5][7]); \
            rC[6][7]=mad(rA[0][6],rB[0][7],rC[6][7]); \
            rC[7][7]=mad(rA[0][7],rB[0][7],rC[7][7]); \



#define  M6x6 \
            rA[0][0] = lA[offA + 0];                              \
            rA[0][1] = lA[offA + 16];                             \
            rA[0][2] = lA[offA + 32];                             \
            rA[0][3] = lA[offA + 48];                             \
            rA[0][4] = lA[offA + 64];                             \
            rA[0][5] = lA[offA + 80];                             \
            rB[0][0] = lB[offB + 0];                              \
            rB[0][1] = lB[offB + 16];                             \
            rB[0][2] = lB[offB + 32];                             \
            rB[0][3] = lB[offB + 48];                             \
            rB[0][4] = lB[offB + 64];                             \
            rB[0][5] = lB[offB + 80];                             \
            offA += 97;                                                           \
            offB += 97;                                                           \
            rC[0][0]=mad(rA[0][0],rB[0][0],rC[0][0]); \
            rC[1][0]=mad(rA[0][1],rB[0][0],rC[1][0]); \
            rC[2][0]=mad(rA[0][2],rB[0][0],rC[2][0]); \
            rC[3][0]=mad(rA[0][3],rB[0][0],rC[3][0]); \
            rC[4][0]=mad(rA[0][4],rB[0][0],rC[4][0]); \
            rC[5][0]=mad(rA[0][5],rB[0][0],rC[5][0]); \
            rC[0][1]=mad(rA[0][0],rB[0][1],rC[0][1]); \
            rC[1][1]=mad(rA[0][1],rB[0][1],rC[1][1]); \
            rC[2][1]=mad(rA[0][2],rB[0][1],rC[2][1]); \
            rC[3][1]=mad(rA[0][3],rB[0][1],rC[3][1]); \
            rC[4][1]=mad(rA[0][4],rB[0][1],rC[4][1]); \
            rC[5][1]=mad(rA[0][5],rB[0][1],rC[5][1]); \
            rC[0][2]=mad(rA[0][0],rB[0][2],rC[0][2]); \
            rC[1][2]=mad(rA[0][1],rB[0][2],rC[1][2]); \
            rC[2][2]=mad(rA[0][2],rB[0][2],rC[2][2]); \
            rC[3][2]=mad(rA[0][3],rB[0][2],rC[3][2]); \
            rC[4][2]=mad(rA[0][4],rB[0][2],rC[4][2]); \
            rC[5][2]=mad(rA[0][5],rB[0][2],rC[5][2]); \
            rC[0][3]=mad(rA[0][0],rB[0][3],rC[0][3]); \
            rC[1][3]=mad(rA[0][1],rB[0][3],rC[1][3]); \
            rC[2][3]=mad(rA[0][2],rB[0][3],rC[2][3]); \
            rC[3][3]=mad(rA[0][3],rB[0][3],rC[3][3]); \
            rC[4][3]=mad(rA[0][4],rB[0][3],rC[4][3]); \
            rC[5][3]=mad(rA[0][5],rB[0][3],rC[5][3]); \
            rC[0][4]=mad(rA[0][0],rB[0][4],rC[0][4]); \
            rC[1][4]=mad(rA[0][1],rB[0][4],rC[1][4]); \
            rC[2][4]=mad(rA[0][2],rB[0][4],rC[2][4]); \
            rC[3][4]=mad(rA[0][3],rB[0][4],rC[3][4]); \
            rC[4][4]=mad(rA[0][4],rB[0][4],rC[4][4]); \
            rC[5][4]=mad(rA[0][5],rB[0][4],rC[5][4]); \
            rC[0][5]=mad(rA[0][0],rB[0][5],rC[0][5]); \
            rC[1][5]=mad(rA[0][1],rB[0][5],rC[1][5]); \
            rC[2][5]=mad(rA[0][2],rB[0][5],rC[2][5]); \
            rC[3][5]=mad(rA[0][3],rB[0][5],rC[3][5]); \
            rC[4][5]=mad(rA[0][4],rB[0][5],rC[4][5]); \
            rC[5][5]=mad(rA[0][5],rB[0][5],rC[5][5]); 


#define MSS4X4					      \
           rA[0][0] = lA[offA];                       \
           rA[0][1] = lA[offA + 272];		      \
           rA[0][2] = lA[offA + 544];		      \
           rA[0][3] = lA[offA + 816];		      \
           rB[0][0] = lB[offB];	 		      \
           rB[0][1] = lB[offB + 272];		      \
           rB[0][2] = lB[offB + 544];		      \
           rB[0][3] = lB[offB + 816];		      \
           rC[0][0]=mad(rA[0][0],rB[0][0],rC[0][0]);  \
           rC[0][0]=mad(rA[0][1],rB[0][1],rC[0][0]);  \
           rC[0][0]=mad(rA[0][2],rB[0][2],rC[0][0]);  \
           rC[0][0]=mad(rA[0][3],rB[0][3],rC[0][0]);  \
           offA++;       			      \
           offB++;       			      

#define MSS6X6					      \
           rA[0][0] = lA[offA];                       \
           rA[0][1] = lA[offA + 272];		      \
           rA[0][2] = lA[offA + 544];		      \
           rA[0][3] = lA[offA + 816];		      \
           rA[0][4] = lA[offA + 1088];		      \
           rA[0][5] = lA[offA + 1360];		      \
           rB[0][0] = lB[offB];	 		      \
           rB[0][1] = lB[offB + 272];		      \
           rB[0][2] = lB[offB + 544];		      \
           rB[0][3] = lB[offB + 816];		      \
           rB[0][4] = lB[offB + 1088];		      \
           rB[0][5] = lB[offB + 1360];		      \
           rC[0][0]=mad(rA[0][0],rB[0][0],rC[0][0]);  \
           rC[0][0]=mad(rA[0][1],rB[0][1],rC[0][0]);  \
           rC[0][0]=mad(rA[0][2],rB[0][2],rC[0][0]);  \
           rC[0][0]=mad(rA[0][3],rB[0][3],rC[0][0]);  \
           rC[0][0]=mad(rA[0][4],rB[0][4],rC[0][0]);  \
           rC[0][0]=mad(rA[0][5],rB[0][5],rC[0][0]);  \
           offA++;       			      \
           offB++;       			      

hcblasStatus gemm_NoTransAB_STEP_NBK_Mx16_NX16_KX64_TS16XMS4(hc::accelerator_view accl_view,
					     float *A, long aOffset,
					     float *B, long bOffset,
					     float *C, long cOffset,
					     int M, int N, int K, int lda, int ldb, int ldc,
					     float alpha, float beta);

hcblasStatus gemm_NoTransAB_STEP_NBK_M_N_K_TS16XMS4(hc::accelerator_view accl_view,
                                             float *A, long aOffset,
                                             float *B, long bOffset,
                                             float *C, long cOffset,
                                             int M, int N, int K, int lda, int ldb, int ldc,
                                             float alpha, float beta);

hcblasStatus gemm_NoTransAB_STEP_NBK_Mx16_NX16_KX96_TS16XMS6(hc::accelerator_view accl_view,
					     float *A, long aOffset,
					     float *B, long bOffset,
					     float *C, long cOffset,
					     int M, int N, int K, int lda, int ldb, int ldc,
					     float alpha, float beta);

hcblasStatus gemm_NoTransAB_STEP_NBK_M_N_K_TS16XMS6(hc::accelerator_view accl_view,
                                             float *A, long aOffset,
                                             float *B, long bOffset,
                                             float *C, long cOffset,
                                             int M, int N, int K, int lda, int ldb, int ldc,
                                             float alpha, float beta);


hcblasStatus gemm_NoTransAB_MICRO_NBK_MX064_NX064_KX16_TS16XMTS4(hc::accelerator_view accl_view,
					       const  float *A, long aOffset,
					       const  float *B, long bOffset,
					        float *C, long cOffset,
					        int M, int N, int K, int lda, int ldb, int ldc,
					        float alpha, float beta); 

hcblasStatus gemm_NoTransAB_MICRO_NBK_Mini_Batch_M128_N128_K16_TS16XMTS2_MB2(hc::accelerator_view accl_view,
					       const  float *A, long aOffset,
					       const  float *B, long bOffset,
					        float *C, long cOffset,
					        int M, int N, int K, int lda, int ldb, int ldc,
					        float alpha, float beta); 

hcblasStatus gemm_NoTransAB_MICRO_NBK_Mini_Batch_M128_N128_K16_TS16XMTS4_MB2(hc::accelerator_view accl_view,
					       const  float *A, long aOffset,
					       const  float *B, long bOffset,
					        float *C, long cOffset,
					        int M, int N, int K, int lda, int ldb, int ldc,
					        float alpha, float beta); 

hcblasStatus gemm_NoTransAB_MICRO_NBK_Mini_Batch_M_N_K_TS16XMTS2_MB2(hc::accelerator_view accl_view,
					       const  float *A, long aOffset,
					       const  float *B, long bOffset,
					        float *C, long cOffset,
					        int M, int N, int K, int lda, int ldb, int ldc,
					        float alpha, float beta); 

hcblasStatus gemm_NoTransAB_MICRO_NBK_Mini_Batch_M_N_K_TS16XMTS4_MB2(hc::accelerator_view accl_view,
					       const  float *A, long aOffset,
					       const  float *B, long bOffset,
					        float *C, long cOffset,
					        int M, int N, int K, int lda, int ldb, int ldc,
					        float alpha, float beta); 

hcblasStatus gemm_NoTransAB_MICRO_NBK_M_N_K_TS16XMTS2(hc::accelerator_view accl_view,
					       const  float *A, long aOffset,
					       const  float *B, long bOffset,
					        float *C, long cOffset,
					        int M, int N, int K, int lda, int ldb, int ldc,
					        float alpha, float beta); 

hcblasStatus gemm_NoTransAB_MICRO_NBK_M_N_K_TS8XMTS4(hc::accelerator_view accl_view,
					       const  float *A, long aOffset,
					       const  float *B, long bOffset,
					        float *C, long cOffset,
					        int M, int N, int K, int lda, int ldb, int ldc,
					        float alpha, float beta); 

hcblasStatus gemm_NoTransAB_MICRO_NBK_M_N_K_TS16XMTS4(hc::accelerator_view accl_view,
					       const  float *A, long aOffset,
					       const  float *B, long bOffset,
					        float *C, long cOffset,
					        int M, int N, int K, int lda, int ldb, int ldc,
					        float alpha, float beta); 

hcblasStatus gemm_NoTransAB_MICRO_NBK_M_N_K_TS16XMTS6(hc::accelerator_view accl_view,
					       const  float *A, long aOffset,
					       const  float *B, long bOffset,
					        float *C, long cOffset,
					        int M, int N, int K, int lda, int ldb, int ldc,
					        float alpha, float beta); 

hcblasStatus gemm_NoTransAB_MICRO_NBK_MX096_NX096_KX16_TS16XMTS6(hc::accelerator_view accl_view,
					       const  float *A, long aOffset,
					       const  float *B, long bOffset,
					        float *C, long cOffset,
					        int M, int N, int K, int lda, int ldb, int ldc,
					        float alpha, float beta); 

hcblasStatus gemm_NoTransAB(hc::accelerator_view accl_view,
                            float *A, long aOffset,
                            float *B, long bOffset,
                            float *C, long cOffset,
                            int M, int N, int K, int lda, int ldb, int ldc,
                            float alpha, float beta);

hcblasStatus gemm_NoTransA_MICRO_NBK_M096_N096_K096_TS16XMTS6(hc::accelerator_view accl_view,
					       const float *A, long aOffset,
					       const float *B, long bOffset,
					       float *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       float alpha, float beta);

hcblasStatus gemm_NoTransA_MICRO_NBK_M064_N064_K064_TS16XMTS4(hc::accelerator_view accl_view,
					       const float *A, long aOffset,
					       const float *B, long bOffset,
					       float *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       float alpha, float beta);

hcblasStatus gemm_NoTransA_MICRO_NBK_M_N_K_TS16XMTS2(hc::accelerator_view accl_view,
					       const float *A, long aOffset,
					       const float *B, long bOffset,
					       float *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       float alpha, float beta);

hcblasStatus gemm_NoTransA_MICRO_NBK_M_N_K_TS16XMTS4(hc::accelerator_view accl_view,
					       const float *A, long aOffset,
					       const float *B, long bOffset,
					       float *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       float alpha, float beta);

hcblasStatus gemm_NoTransA_MICRO_NBK_M_N_K_TS16XMTS6(hc::accelerator_view accl_view,
					       const float *A, long aOffset,
					       const float *B, long bOffset,
					       float *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       float alpha, float beta);

hcblasStatus gemm_NoTransA_MICRO_NBK_Mini_Batch_M128_N128_K16_TS16XMTS2_MB2(hc::accelerator_view accl_view,
					       const float *A, long aOffset,
					       const float *B, long bOffset,
					       float *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       float alpha, float beta);

hcblasStatus gemm_NoTransA_MICRO_NBK_Mini_Batch_M128_N128_K16_TS16XMTS4_MB2(hc::accelerator_view accl_view,
					       const float *A, long aOffset,
					       const float *B, long bOffset,
					       float *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       float alpha, float beta);

hcblasStatus gemm_NoTransA_MICRO_NBK_Mini_Batch_M_N_K_TS16XMTS2_MB2(hc::accelerator_view accl_view,
					       const  float *A, long aOffset,
					       const  float *B, long bOffset,
					        float *C, long cOffset,
					        int M, int N, int K, int lda, int ldb, int ldc,
					        float alpha, float beta); 

hcblasStatus gemm_NoTransA_MICRO_NBK_Mini_Batch_M_N_K_TS16XMTS4_MB2(hc::accelerator_view accl_view,
					       const  float *A, long aOffset,
					       const  float *B, long bOffset,
					        float *C, long cOffset,
					        int M, int N, int K, int lda, int ldb, int ldc,
					        float alpha, float beta); 

hcblasStatus gemm_NoTransA(hc::accelerator_view accl_view,
	                   float *A, long aOffset,
                           float *B, long bOffset,
                           float *C, long cOffset,
                           int M, int N, int K, int lda, int ldb, int ldc,
                           float alpha, float beta);

hcblasStatus gemm_NoTransB_MICRO_NBK_M_N_K_TS16XMTS2(hc::accelerator_view accl_view,
					       const float *A, long aOffset,
					       const float *B, long bOffset,
					       float *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       float alpha, float beta);

hcblasStatus gemm_NoTransB_MICRO_NBK_M_N_K_TS16XMTS4(hc::accelerator_view accl_view,
					       const float *A, long aOffset,
					       const float *B, long bOffset,
					       float *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       float alpha, float beta);

hcblasStatus gemm_NoTransB_MICRO_NBK_M_N_K_TS16XMTS6(hc::accelerator_view accl_view,
					       const float *A, long aOffset,
					       const float *B, long bOffset,
					       float *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       float alpha, float beta);

hcblasStatus gemm_NoTransB_MICRO_NBK_M064_N064_K064_TS16XMTS4(hc::accelerator_view accl_view,
					       const float *A, long aOffset,
					       const float *B, long bOffset,
					       float *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       float alpha, float beta);

hcblasStatus gemm_NoTransB_MICRO_NBK_M096_N096_K096_TS16XMTS6(hc::accelerator_view accl_view,
					       const float *A, long aOffset,
					       const float *B, long bOffset,
					       float *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       float alpha, float beta);

hcblasStatus gemm_NoTransB_MICRO_NBK_Mini_Batch_M128_N128_K16_TS16XMTS2_MB2(hc::accelerator_view accl_view,
					       const float *A, long aOffset,
					       const float *B, long bOffset,
					       float *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       float alpha, float beta);

hcblasStatus gemm_NoTransB_MICRO_NBK_Mini_Batch_M128_N128_K16_TS16XMTS4_MB2(hc::accelerator_view accl_view,
					       const float *A, long aOffset,
					       const float *B, long bOffset,
					       float *C, long cOffset,
					       int M, int N, int K, int lda, int ldb, int ldc,
					       float alpha, float beta);

hcblasStatus gemm_NoTransB_MICRO_NBK_Mini_Batch_M_N_K_TS16XMTS2_MB2(hc::accelerator_view accl_view,
					       const  float *A, long aOffset,
					       const  float *B, long bOffset,
					        float *C, long cOffset,
					        int M, int N, int K, int lda, int ldb, int ldc,
					        float alpha, float beta); 

hcblasStatus gemm_NoTransB_MICRO_NBK_Mini_Batch_M_N_K_TS16XMTS4_MB2(hc::accelerator_view accl_view,
					       const  float *A, long aOffset,
					       const  float *B, long bOffset,
					        float *C, long cOffset,
					        int M, int N, int K, int lda, int ldb, int ldc,
					        float alpha, float beta); 

hcblasStatus gemm_NoTransB(hc::accelerator_view accl_view,
          		   float *A, long aOffset,
                           float *B, long bOffset,
                           float *C, long cOffset,
                           int M, int N, int K, int lda, int ldb, int ldc,
                           float alpha, float beta);

hcblasStatus gemm_TransAB(hc::accelerator_view accl_view,
                          float *A, long aOffset,
                          float *B, long bOffset,
                          float *C, long cOffset,
                          int M, int N, int K, int lda, int ldb, int ldc,
                          float alpha, float beta);


/*
* SGEMM Kernels for Batch processing in column major order
*/

hcblasStatus gemm_NoTransAB(hc::accelerator_view accl_view,
                            float *A, long aOffset, long A_batchOffset,
                            float *B, long bOffset, long B_batchOffset,
                            float *C, long cOffset, long C_batchOffset,
                            int M, int N, int K, int lda, int ldb, int ldc,
                            float alpha, float beta, int batchSize);

hcblasStatus gemm_NoTransA(hc::accelerator_view accl_view,
	                   float *A, long aOffset, long A_batchOffset,
                           float *B, long bOffset, long B_batchOffset,
                           float *C, long cOffset, long C_batchOffset,
                           int M, int N, int K, int lda, int ldb, int ldc,
                           float alpha, float beta, int batchSize);

hcblasStatus gemm_NoTransB(hc::accelerator_view accl_view,
          		   float *A, long aOffset, long A_batchOffset,
                           float *B, long bOffset, long B_batchOffset,
                           float *C, long cOffset, long C_batchOffset,
                           int M, int N, int K, int lda, int ldb, int ldc,
                           float alpha, float beta, int batchSize);

hcblasStatus gemm_TransAB(hc::accelerator_view accl_view,
                          float *A, long aOffset, long A_batchOffset,
                          float *B, long bOffset, long B_batchOffset,
                          float *C, long cOffset, long C_batchOffset,
                          int M, int N, int K, int lda, int ldb, int ldc,
                          float alpha, float beta, int batchSize);


/*
* SGEMM Kernels - Row major order
*/

hcblasStatus gemm_NoTransAB_rMajor(hc::accelerator_view accl_view,
                                   float *A, long aOffset,
                                   float *B, long bOffset,
                                   float *C, long cOffset,
                                   int M, int N, int K, int lda, int ldb, int ldc,
                                   float alpha, float beta);

hcblasStatus gemm_NoTransA_rMajor(hc::accelerator_view accl_view,
                                  float *A, long aOffset,
                                  float *B, long bOffset,
                                  float *C, long cOffset,
                                  int M, int N, int K, int lda, int ldb, int ldc,
                                  float alpha, float beta);

hcblasStatus gemm_NoTransB_rMajor(hc::accelerator_view accl_view,
                                  float *A, long aOffset,
                                  float *B, long bOffset,
                                  float *C, long cOffset,
                                  int M, int N, int K, int lda, int ldb, int ldc,
                                  float alpha, float beta);

hcblasStatus gemm_TransAB_rMajor(hc::accelerator_view accl_view,
                                 float *A, long aOffset,
                                 float *B, long bOffset,
                                 float *C, long cOffset,
                                 int M, int N, int K, int lda, int ldb, int ldc,
                                 float alpha, float beta);



/*
* SGEMM Kernels for Batch-processing in Row major order
*/

hcblasStatus gemm_NoTransAB_rMajor(hc::accelerator_view accl_view,
                                   float *A, long aOffset, long A_batchOffset,
                                   float *B, long bOffset, long B_batchOffset,
                                   float *C, long cOffset, long C_batchOffset,
                                   int M, int N, int K, int lda, int ldb, int ldc,
                                   float alpha, float beta, int batchSize);

hcblasStatus gemm_NoTransA_rMajor(hc::accelerator_view accl_view,
                                  float *A, long aOffset, long A_batchOffset,
                                  float *B, long bOffset, long B_batchOffset,
                                  float *C, long cOffset, long C_batchOffset,
                                  int M, int N, int K, int lda, int ldb, int ldc,
                                  float alpha, float beta, int batchSize);

hcblasStatus gemm_NoTransB_rMajor(hc::accelerator_view accl_view,
                                  float *A, long aOffset, long A_batchOffset,
                                  float *B, long bOffset, long B_batchOffset,
                                  float *C, long cOffset, long C_batchOffset,
                                  int M, int N, int K, int lda, int ldb, int ldc,
                                  float alpha, float beta, int batchSize);

hcblasStatus gemm_TransAB_rMajor(hc::accelerator_view accl_view,
                                 float *A, long aOffset, long A_batchOffset,
                                 float *B, long bOffset, long B_batchOffset,
                                 float *C, long cOffset, long C_batchOffset,
                                 int M, int N, int K, int lda, int ldb, int ldc,
                                 float alpha, float beta, int batchSize);

