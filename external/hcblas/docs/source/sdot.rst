############
2.2.12. SDOT
############
--------------------------------------------------------------------------------------------------------------------------------------------

| Dot product of two vectors (vectors x and y) containing float elements (Single precision Dot product).
|
| Where x, y are n-dimensional vectors.

Functions
^^^^^^^^^

Implementation type I
---------------------

 .. note:: **Inputs and Outputs are HCC device pointers.**

`hcblasStatus_t <HCBLAS_TYPES.html#hcblas-status-hcblasstatus-t>`_ **hcblasSdot** (hcblasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result)

Implementation type II
-----------------------

 .. note:: **Inputs and Outputs are HCC device pointers with batch processing.**

`hcblasStatus_t <HCBLAS_TYPES.html#hcblas-status-hcblasstatus-t>`_ **hcblasSdotBatched** (hcblasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result, int batchCount)

Detailed Description
^^^^^^^^^^^^^^^^^^^^

Function Documentation
^^^^^^^^^^^^^^^^^^^^^^

::

                hcblasStatus_t hcblasSdot (hcblasHandle_t handle, int n,
                                           const float           *x, int incx,
                                           const float           *y, int incy,
                                           float                 *result)

+------------+-----------------+--------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                  |
+============+=================+==============================================================+
|    [in]    |  handle         | handle to the HCBLAS library context.                        |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |  n	       | Number of elements in vector x.                              |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |	x	       | Buffer object storing vector x.                              |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |  incx           | Increment for the elements of x. Must not be zero.           |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    | 	y              | Buffer object storing the vector y.                          |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |  incy           | Increment for the elements of y. Must not be zero.           |
+------------+-----------------+--------------------------------------------------------------+
|    [out]   |  result         | Buffer object that will contain the dot-product value.       |
+------------+-----------------+--------------------------------------------------------------+

| Implementation type II has other parameters as follows,
+------------+-----------------+--------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                  |
+============+=================+==============================================================+
|    [in]    |  batchCount     | The size of batch for vector x and vector y.                 |
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
