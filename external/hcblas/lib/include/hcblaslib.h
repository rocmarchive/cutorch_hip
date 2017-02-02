/*
*
*  FILENAME : hcblaslib.h
*  This file is the top level header file which includes the Hcblaslilbrary class
*  for performing blas operations ( saxpy, sger, sgemv, sgemm, cgemm , dgemm, zgemm )
*
*/

#ifndef HCBLASLIB_H
#define HCBLASLIB_H

#include <iostream>
#include "hc.hpp"
#include "hc_am.hpp"
#include "hc_short_vector.hpp"

using namespace hc;
using namespace hc::short_vector;
using namespace std;


/* enumerator to indicate the status of  blas operation */
enum hcblasStatus {
    HCBLAS_SUCCEEDS = 0,
    HCBLAS_INVALID = -1
};

/* enumerator to define the layout of  input matrix for blas operation */
enum hcblasOrder: unsigned short {
    RowMajor ,
    ColMajor
};

/* enumerator to define the type of operation to be performed on the input matrix
 ( NO_TRANSPOSE, TRANSPOSE, CONJUGATE) */
enum hcblasTranspose {
    NoTrans = 'n',
    Trans = 't'
};

struct hc_Complex
{
     float real;
     float img;
};

/* Class which implements the blas ( SGEMM, CGEMM, SGEMV, SGER, SAXPY )  */
class Hcblaslibrary
{
    public:

    // Constructor to initialize the library with the given accelerator
    Hcblaslibrary(hc::accelerator *acc)
    {
      // When accelerator specified is null, use the default accelerator 
      if (acc == nullptr) 
      {
        hc::accelerator default_acc;
        acc = &default_acc;
      }
      std::vector<accelerator> accs = accelerator::get_all();
      for (int i=0;i<accs.size();i++) {
        if (accs[i] == *acc) {
          this-> initialized = true;
          break;
        }
      }
      assert(this->initialized);
      this->currentAccl = *acc;
      auto accl_view = (*acc).get_default_view();
      this->currentAcclView = accl_view;
      // TODO: Add another constructor to accomodate row major setting
      this->Order = ColMajor;
    }

    ~Hcblaslibrary() 
    {
       // Deinitialize the library
       this->initialized = false;
    }

    // Add current Accerator field
    hc::accelerator currentAccl;

    // Filed to check if library is initialized
    bool initialized = false;

    // Add current Accerator View field set with a default accelerator view of default accelerator
    // TODO - change to pointer
    hc::accelerator_view currentAcclView = this->currentAccl.get_default_view();

    // StreamInfo
    void* currentStream = NULL;

    hcblasOrder Order;
        
/* SAXPY - Y = alpha * X + Y                                    */
/* SAXPY - Overloaded function with arguments of type hc::array */

    hcblasStatus hcblas_saxpy(hc::accelerator_view accl_view,
			      const int N, const float &alpha,
			      const float *X, const int incX,
                              float *Y, const int incY, 
			      const long xOffset, const long yOffset);

/* SAXPY - Overloaded function with arguments related to batch processing */

    hcblasStatus hcblas_saxpy(hc::accelerator_view accl_view,
                              const int N, const float &alpha,
                              const float *X, const int incX, const long X_batchOffset,
                              float *Y, const int incY, const long Y_batchOffset,
                              const long xOffset, const long yOffset, const int batchSize);
							  
/* DAXPY - Y = alpha * X + Y                                    */
/* DAXPY - Overloaded function with arguments of type hc::array */

    hcblasStatus hcblas_daxpy(hc::accelerator_view accl_view,
			      const int N, const double &alpha,
			      const double *X, const int incX,
                              double *Y, const int incY, 
			      const long xOffset, const long yOffset);

/* DAXPY - Overloaded function with arguments related to batch processing */

    hcblasStatus hcblas_daxpy(hc::accelerator_view accl_view,
                              const int N, const double &alpha,
                              const double *X, const int incX, const long X_batchOffset,
                              double *Y, const int incY, const long Y_batchOffset,
                              const long xOffset, const long yOffset, const int batchSize);
							  
/* SGER - A = alpha * X * Y' + A                               */
/* SGER - Overloaded function with arguments of type hc::array */
    hcblasStatus hcblas_sger(hc::accelerator_view accl_view,
			     hcblasOrder order, const int M, const int N, const float &alpha,
                             const float *X, const long xOffset, const int incX,
                             const float *Y, const long yOffset, const int incY,
                             float *A, const long aOffset, const int lda);

/* SGER - Overloaded function with arguments related to batch processing */
    hcblasStatus hcblas_sger(hc::accelerator_view accl_view,
                             hcblasOrder order, const int M, const int N, const float &alpha,
                             const float *X, 
                             const long xOffset, const long X_batchOffset, const int incX,
                             const float *Y, 
                             const long yOffset, const long Y_batchOffset, const int incY,
                             float *A, 
                             const long aOffset, const long A_batchOffset, const int lda, const int batchSize);

/* DGER - A = alpha * X * Y' + A                               */
    hcblasStatus hcblas_dger(hc::accelerator_view accl_view,
			     hcblasOrder order, const int M, const int N, const double &alpha,
                             const double *X, const long xOffset, const int incX,
                             const double *Y, const long yOffset, const int incY,
                             double *A, const long aOffset, const int lda);

/* DGER - Overloaded function with arguments related to batch processing */
    hcblasStatus hcblas_dger(hc::accelerator_view accl_view,
                             hcblasOrder order, const int M, const int N, const double &alpha,
                             const double *X, 
                             const long xOffset, const long X_batchOffset, const int incX,
                             const double *Y, 
                             const long yOffset, const long Y_batchOffset, const int incY,
                             double *A, 
                             const long aOffset, const long A_batchOffset, const int lda, const int batchSize);

/* SGEMV - Y = alpha * op(A) * X + beta * Y                     */
    hcblasStatus hcblas_sgemv(hc::accelerator_view accl_view,
			      hcblasOrder order, hcblasTranspose type, const int M,
                              const int N, const float &alpha, 
                              float *A, const long aOffset, const int lda, 
			      float *X, const long xOffset, const int incX,
                              const float &beta,  
			      float *Y, const long yOffset, const int incY);

/* SGEMV - Overloaded function with arguments related to batch processing */
    hcblasStatus hcblas_sgemv(hc::accelerator_view accl_view,
                              hcblasOrder order, hcblasTranspose type, const int M,
                              const int N, const float &alpha, float *A, 
                              const long aOffset, const long A_batchOffset, const int lda,
                              float *X, 
                              const long xOffset, const long X_batchOffset, const int incX,
                              const float &beta, float *Y, 
                              const long yOffset, const long Y_batchOffset, const int incY, const int batchSize);

/* DGEMV - Y = alpha * op(A) * X + beta * Y                     */
    hcblasStatus hcblas_dgemv(hc::accelerator_view accl_view,
			      hcblasOrder order, hcblasTranspose type, const int M,
                              const int N, const double &alpha, 
                              double *A, const long aOffset, const int lda, 
			      double *X, const long xOffset, const int incX,
                              const double &beta,  
			      double *Y, const long yOffset, const int incY);

/* DGEMV - Overloaded function with arguments related to batch processing */
    hcblasStatus hcblas_dgemv(hc::accelerator_view accl_view,
                              hcblasOrder order, hcblasTranspose type, const int M,
                              const int N, const double &alpha, double *A, 
                              const long aOffset, const long A_batchOffset, const int lda,
                              double *X, 
                              const long xOffset, const long X_batchOffset, const int incX,
                              const double &beta, double *Y, 
                              const long yOffset, const long Y_batchOffset, const int incY, const int batchSize);

/* SGEMM - C = alpha * op(A) * op(B) + beta * C                 */
/* SGEMM - Overloaded function with arguments of type dev pointer */
    hcblasStatus hcblas_sgemm(hc::accelerator_view accl_view,
 			      hcblasOrder order, hcblasTranspose typeA,
                              hcblasTranspose typeB, const int M,
                              const int N, const int K, const float &alpha,
                              float *A, const long lda, 
		              float *B, const long ldb, 
			      const float &beta,  
		       	      float *C, const long ldc, 
			      const long aOffset, const long bOffset, const long cOffset);

    hcblasStatus hcblas_dgemm(hc::accelerator_view accl_view,
 			      hcblasOrder order, hcblasTranspose typeA,
                              hcblasTranspose typeB, const int M,
                              const int N, const int K, const double &alpha,
                              double *A, const long lda, 
		              double *B, const long ldb, 
			      const double  &beta,  
		       	      double *C, const long ldc, 
			      const long aOffset, const long bOffset, const long cOffset);

/* SGEMM - Overloaded function with arguments related to batch processing */
    hcblasStatus hcblas_sgemm(hc::accelerator_view accl_view,
                              hcblasOrder order, hcblasTranspose typeA,
                              hcblasTranspose typeB, const int M,
                              const int N, const int K, const float &alpha,
                              float *A, const long lda, const long A_batchOffset,
                              float *B, const long ldb, const long B_batchOffset,
                              const float &beta,
                              float *C, const long ldc, const long C_batchOffset,
                              const long aOffset, const long bOffset, const long cOffset, const int batchSize);

/*  DGEMM - Overloaded function with arguments related to batch processing */
    hcblasStatus hcblas_dgemm(hc::accelerator_view accl_view,
                              hcblasOrder order, hcblasTranspose typeA,
                              hcblasTranspose typeB, const int M,
                              const int N, const int K, const double &alpha,
                              double *A, const long lda, const long A_batchOffset,
                              double *B, const long ldb, const long B_batchOffset,
                              const double &beta,
                              double *C, const long ldc, const long C_batchOffset,
                              const long aOffset, const long bOffset, const long cOffset, const int batchSize);
/* CGEMM - C = alpha * op(A) * op(B) + beta * C                   */
/* CGEMM - Overloaded function with arguments of type hc::array   */     
   hcblasStatus hcblas_cgemm(hc::accelerator_view accl_view,
			     hcblasOrder order, hcblasTranspose typeA,
                             hcblasTranspose typeB, const int M,
                             const int N, const int K,
                             const float_2 &alpha,
                             float_2 *A, const long aOffset, const long lda,
                             float_2 *B, const long bOffset, const long ldb,
                             const float_2 &beta, 
                             float_2 *C, const long cOffset, const long ldc);

   hcblasStatus hcblas_zgemm(hc::accelerator_view accl_view,
			     hcblasOrder order, hcblasTranspose typeA,
                             hcblasTranspose typeB, const int M,
                             const int N, const int K,
                             const  double_2 &alpha,
                             double_2 *A, const long aOffset, const long lda,
                             double_2 *B, const long bOffset, const long ldb,
                             const  double_2 &beta, 
                             double_2 *C, const long cOffset, const long ldc);

/* CGEMM - Overloaded function with arguments related to batch processing */
   hcblasStatus hcblas_cgemm(hc::accelerator_view accl_view,
                             hcblasOrder order, hcblasTranspose typeA,
                             hcblasTranspose typeB, const int M,
                             const int N, const int K,
                             const float_2 &alpha,
                             float_2 *A, 
                             const long aOffset, const long A_batchOffset, const long lda,
                             float_2 *B, 
			     const long bOffset, const long B_batchOffset, const long ldb,
                             const float_2 &beta,
                             float_2 *C, 
			     const long cOffset, const long C_batchOffset, const long ldc, const int batchSize);
				 
/* ZGEMM - Overloaded function with arguments related to batch processing */
   hcblasStatus hcblas_zgemm(hc::accelerator_view accl_view,
                             hcblasOrder order, hcblasTranspose typeA,
                             hcblasTranspose typeB, const int M,
                             const int N, const int K,
                             const  double_2 &alpha,
                             double_2 *A, 
                             const long aOffset, const long A_batchOffset, const long lda,
                             double_2 *B, 
			     const long bOffset, const long B_batchOffset, const long ldb,
                             const  double_2 &beta,
                             double_2 *C, 
			     const long cOffset, const long C_batchOffset, const long ldc, const int batchSize);

/* SSCAL - X = alpha * X */
/* SSCAL - Overloaded function with arguments of type hc::array */
   hcblasStatus hcblas_sscal(hc::accelerator_view accl_view,
		             const int N, const float &alpha,
                             float *X, const int incX,
                             const long xOffset);

/* SSCAL - Overloaded function with arguments related to batch processing */
   hcblasStatus hcblas_sscal(hc::accelerator_view accl_view, 
		             const int N,const float &alpha,
                             float *X, const int incX,
                             const long xOffset, const long X_batchOffset, const int batchSize);

/* DSCAL - X = alpha * X */
/* DSCAL - Overloaded function with arguments of type hc::array */
   hcblasStatus hcblas_dscal(hc::accelerator_view accl_view,
                             const int N, const double &alpha,
                             double *X, const int incX,
                             const long xOffset);

/* DSCAL - Overloaded function with arguments related to batch processing */
   hcblasStatus hcblas_dscal(hc::accelerator_view accl_view,
                             const int N,const double &alpha,
                             double *X, const int incX,
                             const long xOffset, const long X_batchOffset, const int batchSize);

/* SCOPY - Copies a vector X to a vector Y */
/* SCOPY - Overloaded function with arguments of type hc::array */
   hcblasStatus hcblas_scopy(hc::accelerator_view accl_view, const int N,
                             const float *X, const int incX, const long xOffset,
                             float *Y, const int incY, const long yOffset);

/* SCOPY - Overloaded function with arguments related to batch processing */
   hcblasStatus hcblas_scopy(hc::accelerator_view accl_view, const int N,
                             const float *X, const int incX, const long xOffset,
                             float *Y, const int incY, const long yOffset,
                             const long X_batchOffset, const long Y_batchOffset, const int batchSize);

/* DCOPY - Copies a vector X to a vector Y */
/* DCOPY - Overloaded function with arguments of type hc::array */
   hcblasStatus hcblas_dcopy(hc::accelerator_view accl_view, const int N,
                             const double *X, const int incX, const long xOffset,
                             double *Y, const int incY, const long yOffset);

/* DCOPY - Overloaded function with arguments related to batch processing */
   hcblasStatus hcblas_dcopy(hc::accelerator_view accl_view, const int N,
                             const double *X, const int incX, const long xOffset,
                             double *Y, const int incY, const long yOffset,
                             const long X_batchOffset, const long Y_batchOffset, const int batchSize);

/* SDOT - Single Precision Dot product */
/* SDOT - Overloaded function with arguments of type hc::array */
   hcblasStatus hcblas_sdot(hc::accelerator_view accl_view, const int N,
                            const float *X, const int incX, const long xOffset,
                            const float *Y, const int incY, const long yOffset, float &dot);

/* SDOT - Overloaded function with arguments related to batch processing */
   hcblasStatus hcblas_sdot(hc::accelerator_view accl_view, const int N,
                            const float *X, const int incX, const long xOffset,
                            const float *Y, const int incY, const long yOffset, float &dot,
                            const long X_batchOffset, const long Y_batchOffset, const int batchSize);

/* DDOT - Double Precision Dot product */
/* DDOT - Overloaded function with arguments of type hc::array */
   hcblasStatus hcblas_ddot(hc::accelerator_view accl_view, const int N,
                            const double *X, const int incX, const long xOffset,
                            const double *Y, const int incY, const long yOffset, double &dot);

/* DDOT - Overloaded function with arguments related to batch processing */
   hcblasStatus hcblas_ddot(hc::accelerator_view accl_view, const int N,
                            const double *X, const int incX, const long xOffset,
                            const double *Y, const int incY, const long yOffset, double &dot,
                            const long X_batchOffset, const long Y_batchOffset, const int batchSize);

/* SASUM - Absolute value of a Vector - Single Precision */
/* SASUM - Overloaded function with arguments of type hc::array */
   hcblasStatus hcblas_sasum(hc::accelerator_view accl_view, const int N,
                             float *X, const int incX,
                             const long xOffset, float *Y);

/* SASUM - Overloaded function with arguments related to batch processing */
   hcblasStatus hcblas_sasum(hc::accelerator_view accl_view, const int N,
                             float *X, const int incX,
                             const long xOffset, float *Y, const long X_batchOffset, const int batchSize);

/* DASUM - Absolute value of a Vector - Double Precision */
/* DASUM - Overloaded function with arguments of type hc::array */
   hcblasStatus hcblas_dasum(hc::accelerator_view accl_view, const int N,
                             double *X, const int incX,
                             const long xOffset, double *Y);

/* DASUM - Overloaded function with arguments related to batch processing */
   hcblasStatus hcblas_dasum(hc::accelerator_view accl_view, const int N,
                             double *X, const int incX,
                             const long xOffset, double *Y, const long X_batchOffset, const int batchSize);

};


#endif
