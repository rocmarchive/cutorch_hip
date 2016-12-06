#############
2.2.11. DASUM
#############
--------------------------------------------------------------------------------------------------------------------------------------------

| Absolute sum of values of a vector (vector x) containing double elements (Double precision).
|
| Where x is a n-dimensional vector.

Functions
^^^^^^^^^

Implementation type I
---------------------

 .. note:: **Inputs and Outputs are HCC device pointers.**

`hcblasStatus_t <HCBLAS_TYPES.html#hcblas-status-hcblasstatus-t>`_ **hcblasDasum** (hcblasHandle_t handle, int n, double* x, int incx, double* result)

Implementation type II
-----------------------

 .. note:: **Inputs and Outputs are HCC device pointers with batch processing.**

`hcblasStatus_t <HCBLAS_TYPES.html#hcblas-status-hcblasstatus-t>`_ **hcblasDasumBatched** (hcblasHandle_t handle, int n, double* x, int incx, double* result, int batchCount)

Detailed Description
^^^^^^^^^^^^^^^^^^^^

Function Documentation
^^^^^^^^^^^^^^^^^^^^^^

::

             hcblasStatus_t  hcblasDasum(hcblasHandle_t handle, int n,
                                         double                *x, 
                                         int incx, double      *result)

+------------+-----------------+--------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                  |
+============+=================+==============================================================+
|    [in]    |  handle         | handle to the HCBLAS library context.                        |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |  n              | Number of elements in vector x.                              |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |  x              | Buffer object storing vector x.                              |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |  incx           | Increment for the elements of x. Must not be zero.           |
+------------+-----------------+--------------------------------------------------------------+
|    [out]   |  result         | Buffer object that will contain the absolute sum value.      |
+------------+-----------------+--------------------------------------------------------------+

| Implementation type II has other parameters as follows,
+------------+-----------------+--------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                  |
+============+=================+==============================================================+
|    [in]    |  batchCount     | The size of batch for vector x.                              |
+------------+-----------------+--------------------------------------------------------------+

|
| Returns,

==============================    =============================================
STATUS                            DESCRIPTION
==============================    =============================================
HCBLAS_STATUS_SUCCESS             the operation completed successfully
HCBLAS_STATUS_NOT_INITIALIZED     the library was not initialized
HCBLAS_STATUS_EXECUTION_FAILED    the function failed to launch on the GPU
==============================    ============================================= 
