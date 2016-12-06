#ifndef TEST_CONSTANTS_H
#define TEST_CONSTANTS_H

#include <stdlib.h>
#include <time.h>

/*
1. VVSMALL + 1 to 10
2. VSMALL - 11 to 100
3. SMALL - 101 to 1024
4  REGULAR - 1025 - 4096
5  large 4096 - 10240
6  vLarge = 10241 - 16384
7  vv large 16385 -131072
*/

// Generates numbers from 1 to 10
inline long gen_vvsmall() {
  srand(1);
  long vvsmall = rand() % 10 + 1;
  return vvsmall;
}

// Generates numbers from 11 to 100
inline long gen_vsmall() {
  srand(1);
  long vsmall = rand() % 89 + 11;
  return vsmall;
}

// Generates numbers from 101 to 1024
inline long gen_small() {
  srand(1);
  long small = rand() % 923 + 101;
  return small;
}

// Generates numbers from 1025 to 4096
inline long gen_regular() {
  srand(1);
  long regular = rand() % 3071 + 1025;
  return regular;
}

// Generates numbers from 4097 to 10240
inline long gen_large() {
  srand(1);
  long large = rand() % 6143 + 4097;
  return large;
}

// Generates numbers from 10241 to 16384
inline long gen_vlarge() {
  srand(1);
  long vlarge = rand() % 6143 + 10241;
  return vlarge;
}


// Generates numbers from 16385 to 131072
inline long gen_vvlarge() {
  srand(1);
  long vvlarge = rand() % 114687 + 16385;
  return vvlarge;
}
#endif
