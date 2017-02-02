====================
hcBLAS Documentation
====================
************
Introduction
************
--------------------------------------------------------------------------------------------------------------------------------------------

The hcBLAS library is an implementation of BLAS (Basic Linear Algebra Subprograms) targetting the AMD heterogenous hardware via HCC compiler runtime. The computational resources of underlying AMD heterogenous compute gets exposed and exploited through the HCC C++ frontend. Refer `here <https://bitbucket.org/multicoreware/hcc/wiki/Home>`_ for more details on HCC compiler.

To use the hcBLAS API, the application must allocate the required matrices and vectors in the GPU memory space, fill them with data, call the sequence of desired hcBLAS functions, and then upload the results from the GPU memory space back to the host. The hcBLAS API also provides helper functions for writing and retrieving data from the GPU.

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

.. _user-docs:

.. toctree::
   :maxdepth: 2
   
   Getting_Started

.. _api-ref:

.. toctree::
   :maxdepth: 2

   API_reference

.. Index::
   SGEMM
   CGEMM
   SGEMV
   SGER
   SAXPY
   SSCAL
   DSCAL
   SCOPY
   DCOPY
   SASUM
   DASUM
   SDOT
   DDOT
   HCBLAS_TYPES

