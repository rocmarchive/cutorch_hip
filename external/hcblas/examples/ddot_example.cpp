/* Example Ddot program */
/* Compilation: /opt/hcc/bin/clang++ `/opt/hcc/bin/hcc-config --cxxflags --ldflags` -lhc_am -lhcblas ddot_example.cpp */

#include <iostream>
#include "hcblas.h"
#include <cstdlib>
#include "hc_am.hpp"

int main() {
  // variable to hold return status of hcblas routines
  hcblasStatus_t status;

  // Create hcBlas handle object. 
  // Sets default target accelerator (id =1) and data layout as column major  
  hcblasHandle_t handle = NULL;
  status= hcblasCreate(&handle);

  // Ddot input variables
  int n = 123;
  int incx = 1;
  int incy = 1;
  long lenx = 1 + (n-1) * abs(incx);
  long leny = 1 + (n-1) * abs(incy);
  double result;

  // Enumerate the list of accelerators
  std::vector<hc::accelerator>acc = hc::accelerator::get_all();

  // Allocate host pointers
  double *h_X = (double*)calloc(lenx, sizeof(double));//host input
  double *h_Y = (double*)calloc(leny, sizeof(double));

  // Initialize host pointers
  for(int i = 0; i < lenx; i++){
            h_X[i] = rand() % 10;
  }
  for(int i = 0;i < leny;i++){
            h_Y[i] =  rand() % 15;
  }

  // Allocate device pointers
  double* d_X = hc::am_alloc(sizeof(double) * lenx, handle->currentAccl, 0);
  double* d_Y = hc::am_alloc(sizeof(double) * leny, handle->currentAccl, 0);

  // Initialze device pointers using hcblasSetVector utility
  status = hcblasSetVector(handle, lenx, sizeof(double), h_X, 1, d_X, 1);
  if(status != HCBLAS_STATUS_SUCCESS) {
     printf("Data download failure\n");
     exit(1);
  }
  status = hcblasSetVector(handle, leny, sizeof(double), h_Y, 1, d_Y, 1);
  if(status != HCBLAS_STATUS_SUCCESS) {
     printf("Data download failure\n");
     exit(1);
  }

  // Invoke Ddot Blas routine
  status = hcblasDdot(handle, n, d_X, incx, d_Y, incy, &result);
  if(status != HCBLAS_STATUS_SUCCESS) {
     printf("Function invocation failure\n");
     exit(1);
  }

  // Get the device output d_Y back to host
  status = hcblasGetVector(handle, leny, sizeof(double), d_Y, 1, h_Y, 1);
  if(status != HCBLAS_STATUS_SUCCESS) {
     printf("Upload failure\n");
     exit(1);
  }

  // h_Y now contains the results. The user can now print or use h_Y for further computation

  // Deallocate the resources

  // Destroy the handle
  status = hcblasDestroy(handle);
  if(status != HCBLAS_STATUS_SUCCESS) {
     printf("Handle deallocation failure\n");
     exit(1);
  }

  //Free host resources
  free(h_X);
  free(h_Y);

  // Release device resources 
  hc::am_free(d_X);
  hc::am_free(d_Y);
}
