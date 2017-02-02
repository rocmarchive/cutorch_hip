*****************************
2.3.1. Host-only Code Example
*****************************
-------------------------------------------------------------------------------------------------------------------------------------------

.. code-block:: c++

  //This example shows how to invoke Random generators from Host
  //Example with Mrg31k3p Random number generator
  #include <stdio.h>
  #include <iostream>
  #include <hcRNG/mrg31k3p.h>

  int main() {

  //Create Streams
    hcrngMrg31k3pStream* streams = hcrngMrg31k3pCreateStreams(NULL, 2, NULL, NULL);
    hcrngMrg31k3pStream* single = hcrngMrg31k3pCreateStreams(NULL, 1, NULL, NULL);

  //Initialize the count
    int count = 0;
    for (int i = 0; i < 100; i++) {
  //Calling RandomU01 function from host that generates random numbers on "streams"
        double u = hcrngMrg31k3pRandomU01(&streams[i % 2]);

  //Calling RandomInteger funcion from host that generated random numbers on a single stream
        int    x = hcrngMrg31k3pRandomInteger(single, 1, 6);
        if (x * u < 2) count++;
    }
    std::cout << "Average of indicators = " << (double)count / 100.0 << std::endl;
    return 0;
  }
