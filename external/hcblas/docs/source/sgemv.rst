############
2.2.3. SGEMV 
############
--------------------------------------------------------------------------------------------------------------------------------------------

| Single Precision real valued general matrix-vector multiplication.
|
| Matrix-vector products:
|
|    y := alpha*A*x + beta*y 
|    y := alpha*A^T*y + beta*y
|
| Where alpha and beta are scalars, A is the matrix and x, y are vectors.
| () - the actual matrix and ()^T - Transpose of the matrix 


Functions
^^^^^^^^^

Implementation type I
---------------------

 .. note:: **Inputs and Outputs are HCC device pointers.**

`hcblasStatus_t <HCBLAS_TYPES.html#hcblas-status-hcblasstatus-t>`_ **hcblasSgemv** (hcblasHandle_t handle, hcblasOperation_t trans, int m, int n, const float* alpha, float* A, int lda, float* x, int incx, const float* beta, float* y, int incy)

Implementation type II
-----------------------

 .. note:: **Inputs and Outputs are HCC device pointers with batch processing.**
	
`hcblasStatus_t <HCBLAS_TYPES.html#hcblas-status-hcblasstatus-t>`_ **hcblasSgemvBatched** (hcblasHandle_t handle, hcblasOperation_t trans, int m, int n, const float* alpha, float* A, int lda, float* x, int incx, const float* beta, float* y, int incy, int batchCount)

Detailed Description
^^^^^^^^^^^^^^^^^^^^

Function Documentation
^^^^^^^^^^^^^^^^^^^^^^

 ::
              
              hcblasStatus_t hcblasSgemv(hcblasHandle_t handle, hcblasOperation_t trans,
                                         int m, int n,
                                         const float           *alpha,
                                         float                 *A, int lda,
                                         float                 *x, int incx,
                                         const float           *beta,
                                         float                 *y, int incy)

+------------+-----------------+--------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                  |
+============+=================+==============================================================+
|    [in]    |  handle         | handle to the HCBLAS library context.                        |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |	trans          | How matrix A is to be transposed (0 and 1 for NoTrans and    | 
|            |                 | Trans case respectively).                                    |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |	m              | Number of rows in matrix A.                                  |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |	n              | Number of columns in matrix A.                               |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |	alpha          | The factor of matrix A.                                      |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |	A              | Buffer object storing matrix A.                              |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |	lda            | Leading dimension of matrix A. It cannot be less than N when |
|            |                 | the order parameter is set to RowMajor, or less than M when  |
|            |                 | the parameter is set to ColMajor.                            |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |	x	       | Buffer object storing vector x.                              |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |	incx           | Increment for the elements of x. It cannot be zero           |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |	beta           | The factor of the vector y.                                  |
+------------+-----------------+--------------------------------------------------------------+
|    [out]   |	y              | Buffer object storing the vector y.                          |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |	incy           | Increment for the elements of y. It cannot be zero.          |
+------------+-----------------+--------------------------------------------------------------+

| Implementation type II has other parameters as follows,
+------------+-----------------+--------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                  |
+============+=================+==============================================================+
|    [in]    |  batchCount     | The size of batch of threads to be processed in parallel for |
|            |                 | vectors x, y and matrix A.                                   |
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
