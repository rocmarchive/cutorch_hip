############
2.2.2. CGEMM 
############
--------------------------------------------------------------------------------------------------------------------------------------------

| Complex valued general matrix-matrix multiplication.
|
| Matrix-matrix products:
|
|    C := alpha*A*B     + beta*C 
|    C := alpha*A^T*B   + beta*C 
|    C := alpha*A*B^T   + beta*C 
|    C := alpha*A^T*B^T + beta*C 
|
| Where alpha and beta are scalars, and A, B and C are matrices.
| matrix A - m x k matrix
| matrix B - k x n matrix
| matrix C - m x n matrix
| () - the actual matrix and ()^T - transpose of the matrix 

Functions
^^^^^^^^^

Implementation type I
---------------------

 .. note:: **Inputs and Outputs are HCC device pointers.**

`hcblasStatus_t <HCBLAS_TYPES.html#hcblas-status-hcblasstatus-t>`_ **hcblasCgemm** (hcblasHandle_t handle, hcblasOperation_t transa, hcblasOperation_t transb, int m, int n, int k, const `hcComplex* <HCBLAS_TYPES.html#enumerations>`_ alpha, `hcComplex* <HCBLAS_TYPES.html#enumerations>`_ A, int lda, `hcComplex* <HCBLAS_TYPES.html#enumerations>`_ B, int ldb, const `hcComplex* <HCBLAS_TYPES.html#enumerations>`_ beta, `hcComplex* <HCBLAS_TYPES.html#enumerations>`_ C, int ldc)

Implementation type II
-----------------------

 .. note:: **Inputs and Outputs are HCC device pointers with batch processing.**

`hcblasStatus_t <HCBLAS_TYPES.html#hcblas-status-hcblasstatus-t>`_ **hcblasCgemmBatched** (hcblasHandle_t handle, hcblasOperation_t transa, hcblasOperation_t transb, int m, int n, int k, const `hcComplex* <HCBLAS_TYPES.html#enumerations>`_ alpha, `hcComplex* <HCBLAS_TYPES.html#enumerations>`_ A, int lda, `hcComplex* <HCBLAS_TYPES.html#enumerations>`_ B, int ldb, const `hcComplex* <HCBLAS_TYPES.html#enumerations>`_ beta, `hcComplex* <HCBLAS_TYPES.html#enumerations>`_ C, int ldc, int batchCount)

Detailed Description
^^^^^^^^^^^^^^^^^^^^

Function Documentation
^^^^^^^^^^^^^^^^^^^^^^

 ::

             hcblasStatus_t hcblasCgemm(hcblasHandle_t handle,
                                        hcblasOperation_t transa, hcblasOperation_t transb,
                                        int m, int n, int k,
                                        const hcComplex       *alpha,
                                        hcComplex             *A, int lda,
                                        hcComplex             *B, int ldb,
                                        const hcComplex       *beta,
                                        hcComplex             *C, int ldc)

+------------+-----------------+--------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                  |
+============+=================+==============================================================+
|    [in]    |    handle       | handle to the HCBLAS library context.                        |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |    transa       | How matrix A is to be transposed (0 and 1 for NoTrans        |
|            |                 | and Trans case respectively).                                |                            
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |    transb       | How matrix B is to be transposed (0 and 1 for NoTrans        |
|            |                 | and Trans case respectively).                                |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |    m            | Number of rows in matrix A.                                  |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |    n            | Number of columns in matrix B.                               |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |    k            | Number of columns in matrix A and rows in matrix B.          |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |    alpha        | The factor of matrix A.                                      |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |    A            | Buffer object storing matrix A.                              |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |    lda          | Leading dimension of matrix A. It cannot be less than K when |
|            |                 | the order parameter is set to RowMajor, or less than M when  |
|            |                 | the parameter is set to ColMajor.                            |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |    B            | Buffer object storing matrix B.                              |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |    ldb          | Leading dimension of matrix B. It cannot be less than N when |
|            |                 | the order parameter is set to RowMajor, or less than K when  |
|            |                 | it is set to ColMajor.                                       |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |    beta         | The factor of matrix C.                                      |
+------------+-----------------+--------------------------------------------------------------+
|    [out]   |    C            | Buffer object storing matrix C.                              |
+------------+-----------------+--------------------------------------------------------------+
|    [in]    |    ldc          | Leading dimension of matrix C. It cannot be less than N when |
|            |                 | the order parameter is set to RowMajor, or less than M when  |
|            |                 | it is set to ColMajor.                                       |
+------------+-----------------+--------------------------------------------------------------+  

| Implementation type II has other parameters as follows,
+------------+-----------------+--------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                  |
+============+=================+==============================================================+
|    [in]    |  batchCount     | The size of batch of threads to be processed in parallel for |
|            |                 | Matrices A, B and Output matrix C.                           |
+------------+-----------------+--------------------------------------------------------------+

|
| Returns,

==============================    =============================================
STATUS                            DESCRIPTION
==============================    =============================================
HCBLAS_STATUS_SUCCESS             the operation completed successfully
HCBLAS_STATUS_NOT_INITIALIZED     the library was not initialized
HCBLAS_STATUS_INVALID_VALUE       the parameters m,n,k,batchCount<0
HCBLAS_STATUS_EXECUTION_FAILED    the function failed to launch on the GPU
==============================    ============================================= 
