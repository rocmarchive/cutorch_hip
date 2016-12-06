###########
2.2.4. SGER 
###########
--------------------------------------------------------------------------------------------------------------------------------------------

| Vector-vector product with float elements and performs the rank 1 operation (Single precision).
|
| Vector-vector products:
|
|    A := alpha*x*y^T + A
|
| Where alpha is a scalar, A is the matrix and x, y are vectors.
| () - the actual matrix and ()^T - transpose of the matrix
 

Functions
^^^^^^^^^

Implementation type I
---------------------

 .. note:: **Inputs and Outputs are HCC device pointers.**

`hcblasStatus_t <HCBLAS_TYPES.html#hcblas-status-hcblasstatus-t>`_ **hcblasSger** (hcblasHandle_t handle, int m, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* A, int lda)

Implementation type II
-----------------------

 .. note:: **Inputs and Outputs are HCC device pointers with batch processing.**

`hcblasStatus_t <HCBLAS_TYPES.html#hcblas-status-hcblasstatus-t>`_ **hcblasSgerBatched** (hcblasHandle_t handle, int m, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* A, int lda, int batchCount)

Detailed Description
^^^^^^^^^^^^^^^^^^^^

Function Documentation
^^^^^^^^^^^^^^^^^^^^^^

 ::

              hcblasStatus_t  hcblasSger(hcblasHandle_t handle, int m, int n,
                                         const float           *alpha,
                                         const float           *x, int incx,
                                         const float           *y, int incy,
                                         float                 *A, int lda)

+------------+-----------------+--------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                  |
+============+=================+==============================================================+
|    [in]    |  handle         | handle to the HCBLAS library context.                        |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |	m              | Number of rows in matrix A.                                  |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |	n	       | Number of columns in matrix A.                               |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |	alpha	       | specifies the scalar alpha.                                  |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |	x              | Buffer object storing vector x.                              |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |	incx	       | Increment for the elements of x. Must not be zero.           |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |	y	       | Buffer object storing vector y.                              |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |	incy	       | Increment for the elements of y. Must not be zero.           |
+------------+-----------------+--------------------------------------------------------------+
|    [out]   | 	A              | Buffer object storing matrix A. On exit, A is overwritten    |
|            |                 | by the updated matrix.                                       |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |	lda	       | Leading dimension of matrix A. It cannot be less than N when |
|            |                 | the order parameter is set to RowMajor, or less than M       |
|            |                 | when the parameter is set to ColMajor.                       |
+------------+-----------------+--------------------------------------------------------------+

| Implementation type II has other parameters as follows,
+------------+-----------------+--------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                  |
+============+=================+==============================================================+
|    [in]    |  batchCount     | The size of batch of threads to be processed in parallel for |
|            |                 | vectors x, y and Output matrix A.                            |
+------------+-----------------+--------------------------------------------------------------+

|
| Returns, 

==============================    =============================================
STATUS                            DESCRIPTION
==============================    =============================================
HCBLAS_STATUS_SUCCESS             the operation completed successfully
HCBLAS_STATUS_NOT_INITIALIZED     the library was not initialized
HCBLAS_STATUS_INVALID_VALUE       the parameters m,n<0 or incx,incy=0
HCBLAS_STATUS_EXECUTION_FAILED    the function failed to launch on the GPU
==============================    ============================================= 
