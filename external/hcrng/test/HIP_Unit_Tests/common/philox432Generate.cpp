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
  unsigned int *hostData1, *devData1,*hostData2, *devData2, *hostData3, *devData3;

  // Allocate n unsigned ints on host
  hostData1 = (unsigned int *)calloc(n, sizeof(unsigned int));

  // Allocate n unsigned ints on device
  hipMalloc((void **)&devData1, n * sizeof(unsigned int));

  //  Create pseudo-random number generator
  hiprngCreateGenerator(&gen, HIPRNG_RNG_PSEUDO_PHILOX432);
  //  Set seed
  hiprngSetPseudoRandomGeneratorSeed(gen,0);
  // Generate random numbers
  status = hiprngGenerate(gen, devData1, n);
 
  if(status) printf("TEST FAILED\n");

  hipMemcpy(hostData1, devData1, n * sizeof(unsigned int), hipMemcpyDeviceToHost);

  // Generate another set of random numbers with same seeed value

  // Allocate n unsigned ints on host
  hostData2 = (unsigned int *)calloc(n, sizeof(unsigned int));

  // Allocate n unsigned ints on device
  hipMalloc((void **)&devData2, n * sizeof(unsigned int));

  //  Create pseudo-random number generator
  hiprngCreateGenerator(&gen, HIPRNG_RNG_PSEUDO_PHILOX432);
  //  Set seed
  hiprngSetPseudoRandomGeneratorSeed(gen,0);
  // Generate random numbers
  status = hiprngGenerate(gen, devData2, n);
  if(status) printf("TEST FAILED\n");
  hipMemcpy(hostData2, devData2, n * sizeof(unsigned int), hipMemcpyDeviceToHost);
  //set different seed value
  hiprngSetPseudoRandomGeneratorSeed(gen,5000);

  // Allocate n unsigned ints on host
  hostData3 = (unsigned int *)calloc(n, sizeof(unsigned int));

  // Allocate n unsigned ints on device
  hipMalloc((void **)&devData3, n * sizeof(unsigned int));
  // Generate random numbers
  status = hiprngGenerate(gen, devData3, n);
  if(!ispassed) printf("TEST FAILED\n");
  hipMemcpy(hostData3, devData3, n * sizeof(unsigned int), hipMemcpyDeviceToHost);

  // Compare outputs
  for(int i =0; i < n; i++) {
    if (hostData1[i] != hostData2[i]) {
      ispassed = 0;
      printf("Results mismatch with same seed value, %u, %u\n", hostData1[i],hostData2[i]);
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
      printf("Results match with different seed value, %u, %u\n", hostData1[i],hostData3[i]);
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
