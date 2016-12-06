/* Example Sgemm program (NoTranspose Case) */
/* Compilation: /opt/hcc/bin/clang++ `/opt/hcc/bin/hcc-config --cxxflags --ldflags` -lhc_am -lhcblas sgemm_example.cpp */

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
  hcblasHandle_t handle = NULL;
  status = hcblasCreate(&handle);

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

  //Free host resources
  free(h_A);
  free(h_B);
  free(h_C);
 
  // Release device resources 
  hc::am_free(d_A);
  hc::am_free(d_B);
  hc::am_free(d_C);
}

