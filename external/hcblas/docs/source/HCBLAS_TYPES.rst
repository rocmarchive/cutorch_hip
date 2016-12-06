#################
2.3. HCBLAS TYPES
#################
--------------------------------------------------------------------------------------------------------------------------------------------

2.3.1. Enumerations
^^^^^^^^^^^^^^^^^^^

| enum hcblasStatus_t {
|  HCBLAS_STATUS_SUCCESS,
|  HCBLAS_STATUS_NOT_INITIALIZED,  
|  HCBLAS_STATUS_ALLOC_FAILED,     
|  HCBLAS_STATUS_INVALID_VALUE,    
|  HCBLAS_STATUS_MAPPING_ERROR,    
|  HCBLAS_STATUS_EXECUTION_FAILED,
|  HCBLAS_STATUS_INTERNAL_ERROR    
| }
| enum hcblasOrder { RowMajor, ColMajor}
| enum hcblasOperation_t {
|  HCBLAS_OP_N, 
|  HCBLAS_OP_T,  
|  HCBLAS_OP_C   
| }

| typedef float2 hcFloatComplex;
| typedef hcFloatComplex hcComplex;
|

2.3.2. Detailed Description
^^^^^^^^^^^^^^^^^^^^^^^^^^^

2.3.2.1. HCBLAS STATUS (hcblasStatus_t)
---------------------------------------

| This enumeration is the set of HCBLAS error codes.
+-------------------------------------+--------------------------------------------------------------------------------+
| Enumerator                                                                                                           |
+=====================================+================================================================================+
| HCBLAS_STATUS_SUCCESS               | the operation completed successfully.                                          |
+-------------------------------------+--------------------------------------------------------------------------------+    
| HCBLAS_STATUS_NOT_INITIALIZED       | HCBLAS library not initialized.                                                |
+-------------------------------------+--------------------------------------------------------------------------------+
| HCBLAS_STATUS_ALLOC_FAILED          | resource allocation failed.                                                    |
+-------------------------------------+--------------------------------------------------------------------------------+
| HCBLAS_STATUS_INVALID_VALUE         | unsupported numerical value was passed to function.                            |
+-------------------------------------+--------------------------------------------------------------------------------+
| HCBLAS_STATUS_MAPPING_ERROR         | access to GPU memory space failed.                                             |
+-------------------------------------+--------------------------------------------------------------------------------+
| HCBLAS_STATUS_EXECUTION_FAILED      | GPU program failed to execute.                                                 |
+-------------------------------------+--------------------------------------------------------------------------------+
| HCBLAS_STATUS_INTERNAL_ERROR        | an internal HCBLAS operation failed.                                           |
+-------------------------------------+--------------------------------------------------------------------------------+

|

2.3.2.2. HCBLAS ORDER (hcblasOrder)
-----------------------------------

| Shows how matrices are placed in memory.
+------------+--------------------------------------------------------------------------------+
| Enumerator                                                                                  |
+============+================================================================================+
| RowMajor   | Every row is placed sequentially.                                              |
+------------+--------------------------------------------------------------------------------+    
| ColMajor   | Every column is placed sequentially.                                           |
+------------+--------------------------------------------------------------------------------+

|

2.3.2.3. HCBLAS TRANSPOSE (hcblasOperation_t)
---------------------------------------------

| Used to specify whether the matrix is to be transposed or not. 
+----------------+--------------------------------------------------------------------------------+
| Enumerator                                                                                      |
+================+================================================================================+
| HCBLAS_OP_N    |  The Non transpose operation is selected.                                      |
+----------------+--------------------------------------------------------------------------------+    
| HCBLAS_OP_T    |  Transpose operation is selected.                                              |
+----------------+--------------------------------------------------------------------------------+
| HCBLAS_OP_C    |  Conjugate transpose operation is selected.                                    |
+----------------+--------------------------------------------------------------------------------+
