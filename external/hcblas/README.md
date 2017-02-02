## A. Introduction: ##

This repository hosts the HCC based BLAS Library (hcBLAS), that targets GPU acceleration of the traditional set of BLAS routines on AMD devices. . To know what HCC compiler features, refer [here](https://bitbucket.org/multicoreware/hcc/wiki/Home). 


The following list enumerates the current set of BLAS sub-routines that are supported so far. 

* Sgemm  : Single Precision real valued general matrix-matrix multiplication
* Cgemm  : Complex valued general matrix matrix multiplication
* Sgemv  : Single Precision real valued general matrix-vector multiplication
* Sger   : Single Precision General matrix rank 1 operation
* Saxpy  : Scale vector X and add to vector Y
* Sscal  : Single Precision scaling of Vector X 
* Dscal  : Double Precision scaling of Vector X
* Scopy  : Single Precision Copy 
* Dcopy  : Double Precision Copy
* Sasum : Single Precision Absolute sum of values of a vector
* Dasum : Double Precision Absolute sum of values of a vector
* Sdot  : Single Precision Dot product
* Ddot  : Double Precision Dot product

To know more, go through the [Documentation](http://hcblas-documentation.readthedocs.org/en/latest/)


## B. Key Features ##

* Support for 13 commonly used BLAS routines
* Batched GEMM API
* Ability to Choose desired target accelerator
* Single and Double precision


## C. Prerequisites ##

* Refer Prerequisites section [here](http://hcblas-documentation.readthedocs.org/en/latest/Prerequisites.html)

## D. Tested Environment so far 

* Refer Tested environments enumerated [here](http://hcblas-documentation.readthedocs.org/en/latest/Tested_Environments.html)


## E. Installation  

* Follow installation steps as described [here](http://hcblas-documentation.readthedocs.org/en/latest/Installation_steps.html)


## F. Unit testing

* Follow testing procedures as explained [here](http://hcblas-documentation.readthedocs.org/en/latest/Unit_testing.html)

## G. API reference

* The Specification of API's supported along with description  can be found [here](http://hcblas-documentation.readthedocs.org/en/latest/API_reference.html)


## H. Example Code

Sgemm (NoTranspose) example: 

file: sgemmNN_example.cpp

```
#!c++

#include <iostream>
#include "hcblas.h"
#include <cstdlib>
#include "hc_am.hpp"

int main() {
  // Sgemm input variables
  int M = 123;
  int N = 78;
  int K = 23;
  int incx = 1, incy = 1;
  float alpha = 1;
  float beta = 1;
  long lda;
  long ldb;
  long ldc;

  // variable to hold return status of hcblas routines
  hcblasStatus_t status;

  // Create hcBlas handle object. 
  // Sets default target accelerator (id =1) and data layout as column major 
  hcblasHandle_t handle = hcblasCreate();

  // Enumerate the list of accelerators
  std::vector<hc::accelerator>acc = hc::accelerator::get_all();

  // Variables to hold Transpose combinations
  hcblasOperation_t typeA, typeB;

  // Allocate host pointers
  float* h_A = (float*) malloc( M * K * sizeof(float));
  float* h_B = (float*) malloc( K * N * sizeof(float));
  float* h_C = (float*) malloc( M * N * sizeof(float));

  // Initialize host pointers
  for(int i = 0; i < M * K; i++) {
    h_A[i] = rand()%100;
  }
  for(int i = 0; i < K * N;i++) {
    h_B[i] = rand() % 15;
  }
  for(int i = 0; i < M * N;i++) {
    h_C[i] = rand() % 25;
  }

  // Allocate device pointers
  float* d_A = hc::am_alloc(sizeof(float) * M * K, acc[handle->deviceId], 0);
  float* d_B = hc::am_alloc(sizeof(float) * K * N, acc[handle->deviceId], 0);
  float* d_C = hc::am_alloc(sizeof(float) * M * N, acc[handle->deviceId], 0);


  // Initialze device pointers using hcblasSetMatrix utility
  status = hcblasSetMatrix(handle, M, K, sizeof(float), h_A, M, d_A, K);
  if(status != HCBLAS_STATUS_SUCCESS) {
     printf("Data download failure\n");
     exit(1);
  }
  status = hcblasSetMatrix(handle, K, N, sizeof(float), h_B, K, d_A, N);
  if(status != HCBLAS_STATUS_SUCCESS) {
     printf("Data download failure\n");
     exit(1);
  }
  status = hcblasSetMatrix(handle, M, N, sizeof(float), h_C, M, d_C, N);
  if(status != HCBLAS_STATUS_SUCCESS) {
     printf("Data download failure\n");
     exit(1);
  }

  // NoTransA and NoTransB */           
  typeA = HCBLAS_OP_N;
  typeB = HCBLAS_OP_N;

  // Column major Settings */
  lda = M; ldb = K ; ldc = M;

  // Invoke Sgemm Blas routine
  status = hcblasSgemm(handle, typeA, typeB, M, N, K, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);
  if(status != HCBLAS_STATUS_SUCCESS) {
     printf("Function invocation failure\n");
     exit(1);
  }

  // Get the device output d_C back to host
  status = hcblasGetMatrix(handle, M, N, sizeof(float), d_C, M, h_C, N);;
  if(status != HCBLAS_STATUS_SUCCESS) {
     printf("Upload failure\n");
     exit(1);
  }

  // h_C now contains the results. The user can now print or use h_c for further computation

  // Deallocate the resources

  // Destroy the handle
  status = hcblasDestroy(handle);
  if(status != HCBLAS_STATUS_SUCCESS) {
     printf("Handle deallocation failure\n");
     exit(1);
  }

  //Free host resource}s
  free(h_A);
  free(h_B);
  free(h_C);

  // Release device resources 
  hc::am_free(d_A);
  hc::am_free(d_B);
  hc::am_free(d_C);

}


```
* Compiling the example code:
   
     Assuming the library and compiler installation is followed as in [here](http://hcblas-documentation.readthedocs.org/en/latest/#installation-steps)

          /opt/hcc/bin/clang++ `/opt/hcc/bin/hcc-config --cxxflags --ldflags` -lhc_am -lhcblas sgemmNN_example.cpp
