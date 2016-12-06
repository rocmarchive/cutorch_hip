############
2.2.7. DSCAL 
############
--------------------------------------------------------------------------------------------------------------------------------------------

| Scales a double vector by a double constant (Double precision).
|
|    x := alpha*x
|
| Where alpha is a scalar, and x is a n-dimensional vector. 

Functions
^^^^^^^^^

Implementation type I
---------------------

 .. note:: **Inputs and Outputs are HCC device pointers.**

`hcblasStatus_t <HCBLAS_TYPES.html#hcblas-status-hcblasstatus-t>`_ **hcblasDscal** (hcblasHandle_t handle, int n, const double* alpha, double* x, int incx)

Implementation type II
-----------------------

 .. note:: **Inputs and Outputs are HCC device pointers with batch processing.**

`hcblasStatus_t <HCBLAS_TYPES.html#hcblas-status-hcblasstatus-t>`_ **hcblasDscalBatched** (hcblasHandle_t handle, int n, const double* alpha, double* x, int incx, int batchCount)

Detailed Description
^^^^^^^^^^^^^^^^^^^^

Function Documentation
^^^^^^^^^^^^^^^^^^^^^^

::

              hcblasStatus_t  hcblasDscal(hcblasHandle_t handle, int n,
                                          const double          *alpha,
                                          double          *x, int incx)

+------------+-----------------+--------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                  |
+============+=================+==============================================================+
|    [in]    |  handle         | handle to the HCBLAS library context.                        |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |	n              | Number of elements in vector x.                              |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |	alpha          | The constant factor for vector x.                            |
+------------+-----------------+--------------------------------------------------------------+
|    [out]   |	x              | Buffer object storing vector x.                              |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |	incx           | Increment for the elements of x. Must not be zero.           |
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
