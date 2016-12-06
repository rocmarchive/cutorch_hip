*****************
1.5. Unit testing
*****************
--------------------------------------------------------------------------------------------------------------------------------------------

1.5.1 Testing hcBLAS against CBLAS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

a) Automated testing:

       ``cd ~/hcblas/``
 
       ``./build.sh --test=on``

b) Manual testing:

       ``cd ~/hcblas/build/test/src/bin/``

       choose the appropriate named binary


Here are some notes for performing manual testing:

|      TransA (TA) and TransB(TB) takes 0 or 1.             
|      0 - NoTrans (Operate with the given matrix)
|      1 - Trans   (Operate with the transpose of the given matrix)
|
|      Implementation type (Itype) takes 1 or 2.
|      1 - Inputs and Outputs are device pointers.
|      2 - Inputs and Outputs are device pointers with batch processing.
|

      * SGEMM

      ``./sgemm M N K TA TB Itype``

      * CGEMM

      ``./cgemm M N K TA TB Itype``

      * SGEMV

      ``./sgemv M N Trans Itype``

      * SGER

      ``./sger M N Itype``

      * SAXPY

      ``./saxpy N Itype``

      * SSCAL

      ``./sscal N Itype``

      * DSCAL

      ``./dscal N Itype``

      * SCOPY

      ``./scopy N Itype``

      * DCOPY

      ``./dcopy N Itype``

      * SASUM

      ``./sasum N Itype``

      * DASUM

      ``./dasum N Itype``

      * SDOT

      ``./sdot N Itype``

      * DDOT

      ``./ddot N Itype``



