#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Utilities and system includes
#include <assert.h>
// HIP runtime
#include <hip/hip_runtime.h>
#include <hiprng.h>
// HIP and hcrng functions
#include <helper_functions.h>

int main(int argc, char *argv[]) {
  size_t n = 100;
  hiprngGenerator_t gen;
  hiprngStatus_t status = HIPRNG_STATUS_SUCCESS;
  bool ispassed = 1;
  float *hostData1, *devData1,*hostData2, *devData2, *hostData3, *devData3;

  // Allocate n floats on host
  hostData1 = (float *)calloc(n, sizeof(float));

  // Allocate n floats on device
  hipMalloc((void **)&devData1, n * sizeof(float));

  //  Create pseudo-random number generator
  hiprngCreateGenerator(&gen, HIPRNG_RNG_PSEUDO_LFSR113);
  //  Set seed
  hiprngSetPseudoRandomGeneratorSeed(gen,0);
  // Generate random numbers
  status = hiprngGenerateNormal(gen, devData1, n, 0.0, 1.0);
 
  if(status) printf("TEST FAILED\n");

  hipMemcpy(hostData1, devData1, n * sizeof(float), hipMemcpyDeviceToHost);

  // Generate another set of random numbers with same seeed value

  // Allocate n floats on host
  hostData2 = (float *)calloc(n, sizeof(float));

  // Allocate n floats on device
  hipMalloc((void **)&devData2, n * sizeof(float));

  //  Create pseudo-random number generator
  hiprngCreateGenerator(&gen, HIPRNG_RNG_PSEUDO_LFSR113);
  //  Set seed
  hiprngSetPseudoRandomGeneratorSeed(gen,0);
  // Generate random numbers
  status = hiprngGenerateNormal(gen, devData2, n, 0.0, 1.0);
  if(status) printf("TEST FAILED\n");
  hipMemcpy(hostData2, devData2, n * sizeof(float), hipMemcpyDeviceToHost);
  //set different seed value
  hiprngSetPseudoRandomGeneratorSeed(gen,5000);

  // Allocate n floats on host
  hostData3 = (float *)calloc(n, sizeof(float));

  // Allocate n floats on device
  hipMalloc((void **)&devData3, n * sizeof(float));
  // Generate random numbers
  status = hiprngGenerateNormal(gen, devData3, n, 0.0, 1.0);
  if(!ispassed) printf("TEST FAILED\n");
  hipMemcpy(hostData3, devData3, n * sizeof(float), hipMemcpyDeviceToHost);

  // Compare outputs
  for(int i =0; i < n; i++) {
    if (hostData1[i] != hostData2[i]) {
      ispassed = 0;
      printf("Results mismatch with same seed value, %f, %f\n", hostData1[i],hostData2[i]);
      break;
    }
   else
    continue;
   }
  if(!ispassed) printf("TEST FAILED\n");

  // Compare outputs
  for(int i =0; i < n; i++) {
    if (hostData1[i] == hostData3[i]) {
      ispassed = 0;
      printf("Results match with different seed value, %f, %f\n", hostData1[i],hostData3[i]);
      break;
    }
   else
    continue;
   }
  if(!ispassed) printf("TEST FAILED\n");
  //Cleanup
  hiprngDestroyGenerator(gen);
  hipFree(devData1);
  hipFree(devData2);
  hipFree(devData3);

 
  free(hostData1);
  free(hostData2);
  free(hostData3);

  return 0;
}
