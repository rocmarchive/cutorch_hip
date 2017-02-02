#pragma once
#ifndef MODULAR_CH
#define MODULAR_CH

/*! @file Modular arithmetic and linear algebra
 *
 *  This file provides the code specific to the host.  There is another file
 *  included by this file, modular.c.h, for the code shared by the host and the
 *  device.
 *
 *  The preprocessor symbol `MODULAR_NUMBER_TYPE` must be defined as the type
 *  of number (unsigned int, unsigned long, etc.) on which the modular functions operate.
 *
 *  To use the fixed size variant, the preprocessor symbol
 *  `MODULAR_FIXED_SIZE` must be set to the size (number of rows or of columns)
 *  of the matrix.
 *
 *  @note If the project is migrated to C++, this could be rewritten much more
 *  clearly using templates.
 */

// code that is common to host and device
//#include "hcRNG/private/modular.c.h"

#ifndef MODULAR_NUMBER_TYPE
#error "MODULAR_NUMBER_TYPE must be defined"
#endif

#ifdef MODULAR_FIXED_SIZE
#  define N MODULAR_FIXED_SIZE
#  define MATRIX_ELEM(mat, i, j) (mat[i][j])
#else
#  define MATRIX_ELEM(mat, i, j) (mat[i * N + j])
#endif // MODULAR_FIXED_SIZE

//! Compute (a*s + c) % m
#if 1
#define modMult(a, s, c, m) ((MODULAR_NUMBER_TYPE)(((unsigned long) a * s + c) % m))
#else
static MODULAR_NUMBER_TYPE modMult(MODULAR_NUMBER_TYPE a, MODULAR_NUMBER_TYPE s, MODULAR_NUMBER_TYPE c, MODULAR_NUMBER_TYPE m)
{
    MODULAR_NUMBER_TYPE v;
    v = (MODULAR_NUMBER_TYPE) (((unsigned long) a * s + c) % m);
    return v;
}
#endif


//! @brief Matrix-vector modular multiplication
//  @details Also works if v = s.
//  @return v = A*s % m
#ifdef MODULAR_FIXED_SIZE
static void modMatVec (MODULAR_NUMBER_TYPE A[N][N], MODULAR_NUMBER_TYPE s[N], MODULAR_NUMBER_TYPE v[N], MODULAR_NUMBER_TYPE m) __attribute__((hc, cpu)) 
#else
void modMatVec (size_t N, MODULAR_NUMBER_TYPE* A, MODULAR_NUMBER_TYPE* s, MODULAR_NUMBER_TYPE* v, MODULAR_NUMBER_TYPE m) __attribute__((hc, cpu))
#endif
{
    MODULAR_NUMBER_TYPE x[MODULAR_FIXED_SIZE];     // Necessary if v = s
    for (size_t i = 0; i < N; ++i) {
        x[i] = 0;
        for (size_t j = 0; j < N; j++)
            x[i] = modMult(MATRIX_ELEM(A,i,j), s[j], x[i], m);
    }
    for (size_t i = 0; i < N; ++i)
        v[i] = x[i];
}

//! @brief Compute A*B % m
//  @details Also works if A = C or B = C or A = B = C.
//  @return C = A*B % m
#ifdef MODULAR_FIXED_SIZE
static void modMatMat (MODULAR_NUMBER_TYPE A[N][N], MODULAR_NUMBER_TYPE B[N][N], MODULAR_NUMBER_TYPE C[N][N], MODULAR_NUMBER_TYPE m)
#else
void modMatMat (size_t N, MODULAR_NUMBER_TYPE* A, MODULAR_NUMBER_TYPE* B, MODULAR_NUMBER_TYPE* C, MODULAR_NUMBER_TYPE m)
#endif
{
    MODULAR_NUMBER_TYPE V[N];
    MODULAR_NUMBER_TYPE W[N][N];
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j)
            V[j] = MATRIX_ELEM(B,j,i);
#ifdef MODULAR_FIXED_SIZE
        modMatVec (A, V, V, m);
#else
        modMatVec (N, A, V, V, m);
#endif
        for (size_t j = 0; j < N; ++j)
            W[j][i] = V[j];
    }
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j)
            MATRIX_ELEM(C,i,j) = W[i][j];
    }
}


//! @brief Compute matrix B = (A^(2^e) % m)
//  @details Also works if A = B.
#ifdef MODULAR_FIXED_SIZE
static void modMatPowLog2 (MODULAR_NUMBER_TYPE A[N][N], MODULAR_NUMBER_TYPE B[N][N], MODULAR_NUMBER_TYPE m, unsigned int e)
#else
void modMatPowLog2 (size_t N, MODULAR_NUMBER_TYPE* A, MODULAR_NUMBER_TYPE* B, MODULAR_NUMBER_TYPE m, unsigned int e)
#endif
{
    // initialize: B = A
    if (A != B) {
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; ++j)
                MATRIX_ELEM(B,i,j) = MATRIX_ELEM(A,i,j);
        }
    }
    // Compute B = A^{2^e}mod m
    for (unsigned int i = 0; i < e; i++)
#ifdef MODULAR_FIXED_SIZE
        modMatMat (B, B, B, m);
#else
        modMatMat (N, B, B, B, m);
#endif
}


//! @brief Compute matrix B = A^n % m
//  @details Also works if A = B.
#ifdef MODULAR_FIXED_SIZE
static void modMatPow (MODULAR_NUMBER_TYPE A[N][N], MODULAR_NUMBER_TYPE B[N][N], MODULAR_NUMBER_TYPE m, unsigned int n)
#else
void modMatPow (size_t N, MODULAR_NUMBER_TYPE* A, MODULAR_NUMBER_TYPE* B, MODULAR_NUMBER_TYPE m, unsigned int n)
#endif
{
    MODULAR_NUMBER_TYPE W[N][N];

    // initialize: W = A; B = I
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; ++j) {
            W[i][j] = MATRIX_ELEM(A,i,j);
            MATRIX_ELEM(B,i,j) = 0;
        }
    }

    for (size_t j = 0; j < N; ++j)
        MATRIX_ELEM(B,j,j) = 1;

    // Compute B = A^n % m using the binary decomposition of n
    while (n > 0) {
        if (n & 1) // if n is odd
#ifdef MODULAR_FIXED_SIZE
            modMatMat (W, B, B, m);
        modMatMat (W, W, W, m);
#else
            modMatMat (N, &W[0][0], B, B, m);
        modMatMat (N, &W[0][0], &W[0][0], &W[0][0], m);
#endif
        n >>= 1;
    }
}

#undef MATRIX_ELEM
#undef N

#endif // MODULAR_CH
