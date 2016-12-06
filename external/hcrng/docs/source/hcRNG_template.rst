==================================
2.1. hcRNG_template File Reference
==================================
--------------------------------------------------------------------------------------------------------------------------------------------

| Template of the specialized interface for specific generators  `More... <hcRNG_template.html#detailed-description>`_
|
| #include <hcRNG.h>
| #include <stdio.h>
|

*******************
2.1.1. Enumerations
*******************
--------------------------------------------------------------------------------------------------------------------------------------------

*  enum  **hcrngStatus_**

   typedef enum **hcrngStatus_** hcrngStatus

  Error codes. `More... <DataStructures.html#hcrng-status-hcrngstatus>`_

**********************
2.1.2. Data Structures
**********************
--------------------------------------------------------------------------------------------------------------------------------------------

*  struct **hcrngStreamState**

  Stream state [host/device]. `More... <DataStructures.html#hcrngstreamstate>`_

*  struct **hcrngStream**

  Stream object [host/device]. `More... <DataStructures.html#hcrngstream>`_

*  struct **hcrngStreamCreator**

  Stream creator object. `More... <DataStructures.html#hcrngstreamcreator>`_
 
****************
2.1.3. Functions
****************
--------------------------------------------------------------------------------------------------------------------------------------------

2.1.3.1. Helper functions
^^^^^^^^^^^^^^^^^^^^^^^^^

*  const char* **hcrngGetErrorString** ()

  Retrieve the last error message. `More... <DataStructures.html#hcrnggeterrorstring>`_

*  const char* **hcrngGetLibraryRoot** ()

  Retrieve the library installation path. `More... <DataStructures.html#hcrnggetlibraryroot>`_

2.1.3.2. Stream Creators
^^^^^^^^^^^^^^^^^^^^^^^^

Functions to create, destroy and modify stream creator objects (factory pattern).

*  hcrngStreamCreator* 	**hcrngCopyStreamCreator** (const hcrngStreamCreator* creator, `hcrngStatus* <DataStructures.html#hcrng-status-hcrngstatus>`_ err)

  Duplicate an existing stream creator object. `More... <hcRNG_template.html#hcrngcopystreamcreator>`_

 
*  `hcrngStatus <DataStructures.html#hcrng-status-hcrngstatus>`_  **hcrngDestroyStreamCreator** (hcrngStreamCreator* creator)

  Destroy a stream creator object. `More... <hcRNG_template.html#hcrngdestroystreamcreator>`_
 
*  `hcrngStatus <DataStructures.html#hcrng-status-hcrngstatus>`_  **hcrngRewindStreamCreator** (hcrngStreamCreator* creator)

  Reset a stream creator to its original initial state. `More... <hcRNG_template.html#hcrngrewindstreamcreator>`_

*  `hcrngStatus <DataStructures.html#hcrng-status-hcrngstatus>`_  **hcrngSetBaseCreatorState** (hcrngStreamCreator* creator, const hcrngStreamState* baseState)

  Change the base stream state of a stream creator. `More... <hcRNG_template.html#hcrngsetbasecreatorstate>`_

*  `hcrngStatus <DataStructures.html#hcrng-status-hcrngstatus>`_  **hcrngChangeStreamsSpacing** (hcrngStreamCreator* creator, int e, int c)

  Change the spacing between successive streams. `More... <hcRNG_template.html#hcrngchangestreamsspacing>`_
 
2.1.3.3. Stream Allocation, Destruction and Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Functions to create or destroy random streams and arrays of random streams.

*  hcrngStream* **hcrngAllocStreams** (size_t count, size_t* bufSize, `hcrngStatus* <DataStructures.html#hcrng-status-hcrngstatus>`_ err)

  Reserve memory for one or more stream objects. `More... <hcRNG_template.html#hcrngallocstreams>`_

*  `hcrngStatus <DataStructures.html#hcrng-status-hcrngstatus>`_ **hcrngDestroyStreams** (hcrngStream* streams)

  Destroy one or many stream objects. `More... <hcRNG_template.html#hcrngdestroystreams>`_

*  hcrngStream* **hcrngCreateStreams** (hcrngStreamCreator* creator, size_t count, size_t* bufSize, `hcrngStatus* <DataStructures.html#hcrng-status-hcrngstatus>`_ err)

  Allocate memory for and create new RNG stream objects. `More... <hcRNG_template.html#hcrngcreatestreams>`_

*  `hcrngStatus <DataStructures.html#hcrng-status-hcrngstatus>`_ **hcrngCreateOverStreams** (hcrngStreamCreator* creator, size_t count, hcrngStream* streams)

  Create new RNG stream objects in already allocated memory. `More... <hcRNG_template.html#hcrngcreateoverstreams>`_

*  hcrngStream* **hcrngCopyStreams** (size_t count, const hcrngStream* streams, `hcrngStatus* <DataStructures.html#hcrng-status-hcrngstatus>`_ err)

  Clone RNG stream objects. `More... <hcRNG_template.html#hcrngcopystreams>`_

*  `hcrngStatus <DataStructures.html#hcrng-status-hcrngstatus>`_ **hcrngCopyOverStreams** (size_t count, hcrngStream* destStreams, const hcrngStream* srcStreams)

  Copy RNG stream objects in already allocated memory [device]. `More... <hcRNG_template.html#hcrngcopyoverstreams>`_
 
2.1.3.4. Stream Output
^^^^^^^^^^^^^^^^^^^^^^

Functions to read successive values from a random stream.

*  double **hcrngRandomU01** (hcrngStream* stream)

  Generate the next random value in (0,1) [device]. `More... <hcRNG_template.html#hcrngrandomu01>`_

*  int 	**hcrngRandomInteger** (hcrngStream* stream, int i, int j)

  Generate the next random integer value [device]. `More... <hcRNG_template.html#hcrngrandominteger>`_

*  `hcrngStatus <DataStructures.html#hcrng-status-hcrngstatus>`_ **hcrngRandomU01Array** (hcrngStream* stream, size_t count, double* buffer)

  Fill an array with successive random values in (0,1) [device]. `More... <hcRNG_template.html#hcrngrandomu01array>`_

*  `hcrngStatus <DataStructures.html#hcrng-status-hcrngstatus>`_ **hcrngRandomIntegerArray** (hcrngStream* stream, int i, int j, size_t count, int* buffer)

  Fill an array with successive random integer values [device]. `More... <hcRNG_template.html#hcrngrandomintegerarray>`_
 
2.1.3.5. Stream Navigation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Functions to roll back or advance streams by many steps.

*  `hcrngStatus <DataStructures.html#hcrng-status-hcrngstatus>`_ **hcrngRewindStreams** (size_t count, hcrngStream* streams)

  Reinitialize streams to their initial states [device]. `More... <hcRNG_template.html#hcrngrewindstreams>`_
 
*  `hcrngStatus <DataStructures.html#hcrng-status-hcrngstatus>`_ **hcrngRewindSubstreams** (size_t count, hcrngStream* streams)

  Reinitialize streams to their initial substream states [device]. `More... <hcRNG_template.html#hcrngrewindsubstreams>`_

*  `hcrngStatus <DataStructures.html#hcrng-status-hcrngstatus>`_ **hcrngForwardToNextSubstreams** (size_t count, hcrngStream* streams)

  Advance streams to the next substreams [device]. `More... <hcRNG_template.html#hcrngforwardtonextsubstreams>`_

*  hcrngStream* **hcrngMakeSubstreams** (hcrngStream* stream, size_t count, size_t* bufSize, `hcrngStatus* <DataStructures.html#hcrng-status-hcrngstatus>`_ err)

  Allocate and make an array of substreams of a stream. `More... <hcRNG_template.html#hcrngmakesubstreams>`_
 
*  `hcrngStatus <DataStructures.html#hcrng-status-hcrngstatus>`_ **hcrngMakeOverSubstreams** (hcrngStream* stream, size_t count, hcrngStream* substreams)

  Make an array of substreams of a stream. `More... <hcRNG_template.html#hcrngmakeoversubstreams>`_
 
*  `hcrngStatus <DataStructures.html#hcrng-status-hcrngstatus>`_ **hcrngAdvanceStreams** (size_t count, hcrngStream* streams, int e, int c)

  Advance the state of streams by many steps. `More... <hcRNG_template.html#hcrngadvancestreams>`_
 
2.1.3.6. Work Functions
^^^^^^^^^^^^^^^^^^^^^^^

Kernel functions to generate Random numbers.

*  `hcrngStatus <DataStructures.html#hcrng-status-hcrngstatus>`_ **hcrngDeviceRandomU01Array_single** (hc::accelerator_view &accl_view, size_t streamCount, hcrngStream* streams, size_t numberCount, float* outBuffer, int streamlength = 0, size_t streams_per_thread = 1)

*  `hcrngStatus <DataStructures.html#hcrng-status-hcrngstatus>`_ **hcrngDeviceRandomU01Array_double** (hc::accelerator_view &accl_view, size_t streamCount, hcrngStream* streams, size_t numberCount, double* outBuffer, int streamlength = 0, size_t streams_per_thread = 1)

The last two arguments are default arguments and can be used in case of multistream usage. `More... <hcRNG_template.html#hcrngdevicerandomu01array>`_

 
2.1.3.7. Miscellaneous Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


*  `hcrngStatus <DataStructures.html#hcrng-status-hcrngstatus>`_ **hcrngWriteStreamInfo** (const hcrngStream* stream, FILE* file)

 Format and output information about a stream object to a file. `More... <hcRNG_template.html#hcrngwritestreaminfo>`_

*************************** 
2.1.4. Detailed Description
***************************
--------------------------------------------------------------------------------------------------------------------------------------------

Template of the specialized interface for specific generators.

The function and type names in this API all start with hcrng. In each specific implementation, this prefix is expanded to a specific prefix; e.g., hcrngMrg31k3p for the MRG31k3p generator.

In the standard case, streams and substreams are defined as in `[10] <bibliography.html>`_, `[2] <bibliography.html>`_, `[5] <bibliography.html>`_ . The sequence of successive states of the base RNG over its entire period of length ρ is divided into streams whose starting points are Z steps apart. The sequence for each stream (of length Z) is further divided into substreams of length W. The integers Z and W have default values that have been carefully selected to avoid detectable dependence between successive streams and substreams, and are large enough to make sure that streams and substreams will not be exhausted in practice. It is strongly recommended to never change these values (even if the software allows it). The initial state of the first stream (the seed of the library) has a default value. It can be changed by invoking hcrngSetBaseCreatorState() before creating a first stream.

A stream object is a structure that contains the current state of the stream, its initial state (at the beginning of the stream), and the initial state of the current substream. Whenever the user creates a new stream, the software automatically jumps ahead by Z steps to find its initial state, and the three states in the stream object are set to it. The form of the state depends on the type of RNG.

Some functions are available on both the host and the devices (they can be used within a kernel) whereas others (such as stream creation) are available only on the host. Many functions are defined only for arrays of streams; for a single stream, it suffices to specify an array of size 1.When a kernel is called, one should pass a copy of the streams from the host to the global memory of the device. Another copy of the stream state uses it in the kernel code to generate random numbers.

To use the hcRNG library from within a user-defined kernel, the user must include the hcRNG header file corresponding to the desired RNG via an include directive. Other specific preprocessor macros can be placed before including the header file to change settings of the library when the default values are not suitable for the user. The following options are currently available:

|    **HCRNG_SINGLE_PRECISION** : With this option, all the random numbers returned by hcrngRandomU01() and hcrngRandomU01Array(), and generated by hcrngDeviceRandomU01Array(), will be of type float instead of double (the default setting). This option can be activated and affects all implemented RNGs.
|
To generate single-precision floating point numbers also on the host, still using the MRG31k3p generator, the host code should contain:

::

#define HCRNG_SINGLE_PRECISION
#include <mrg31k3p.h>

The functions described here are all available on the host, in all implementations, unless specified otherwise. Only some of the functions and types are also available on the device in addition to the host; they are tagged with [device]. Other functions are only available on the device; they are tagged with [device-only]. Some functions return an error code in err.
Implemented RNG's

The following table lists the RNG's that are currently implemented in hcRNG with the name of the corresponding header file.

+--------------------+-----------------------+---------------------------+
|  RNG               |  Prefix               | Host/Device Header File   |
+====================+=======================+===========================+
| MRG31k3p           |	Mrg31k3p 	     | mrg31k3p.h 	         |
+--------------------+-----------------------+---------------------------+
| MRG32k3a 	     |  Mrg32k3a 	     | mrg32k3a.h 	         |
+--------------------+-----------------------+---------------------------+
| LFSR113 	     |  Lfsr113 	     | lfsr113.h 	         |
+--------------------+-----------------------+---------------------------+
| Philox-4×32-10     |	Philox432            | philox432.h 	         |
+--------------------+-----------------------+---------------------------+


2.1.4.1. The MRG31k3p Generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The MRG31k3p generator is defined in `[4] <bibliography.html>`_ . In its specific implementation, the function and type names start with hcrngMrg31k3p. For this RNG, a state is a vector of six 31-bit integers, represented internally as unsigned int. The entire period length of approximately 2^185 is divided into approximately 2^51 non-overlapping streams of length Z=2^134. Each stream is further partitioned into substreams of length W=2^72. The state (and seed) of each stream is a vector of six 31-bit integers. This size of state is appropriate for having streams running in work items on GPU cards, for example, while providing a sufficient period length for most applications.

2.1.4.2. The MRG32k3a Generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MRG32k3a is a combined multiple recursive generator (MRG) proposed by L'Ecuyer `[7] <bibliography.html>`_, implemented here in 64-bit integer arithmetic. This RNG has a period length of approximately 2^191, and is divided into approximately 2^64 non-overlapping streams of length Z=2^127, and each stream is subdivided in 2^51 substreams of length W=2^76. These are the same numbers as in `[5] <bibliography.html>`_ . The state of a stream at any given step is a six-dimensional vector of 32-bit integers, but those integers are stored as unsigned long (64-bit integers) in the present implementation (so they use twice the space). The generator has 32 bits of resolution. Note that in the original version proposed in `[7] <bibliography.html>`_ and `[5] <bibliography.html>`_, the recurrences are implemented in double instead, and the state is stored in six 32-bit integers. The change in implementation is to avoid using double's, which are not available on many GPU devices, and also because the 64-bit implementation is much faster than that in double when 64-bit integer arithmetic is available on the hardware.

2.1.4.3. The LFSR113 Generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The LFSR113 generator is defined in `[8] <bibliography.html>`_. In its implementation, the function and type names start with hcrngLfsr113. For this RNG, a state vector of four 31-bit integers, represented internally as unsigned int. The period length of approximately 2^113 is divided into approximately 2^23 non-overlapping streams of length Z=2^90. Each stream is further partitioned into 2^35 substreams of length W=2^55. Note that the functions hcrngLfsr113ChangeStreamsSpacing() and hcrngLfsr113AdvancedStreams() are not implemented in the current version.

2.1.4.4. The Philox-4×32-10 Generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The counter-based Philox-4×32-10 generator is defined in `[11] <bibliography.html>`_. Unlike the previous three generators, its design is not supported by a theoretical analysis of equidistribution. It has only been subjected to empirical testing with the TestU01 software `[3] <bibliography.html>`_ (the other three generators also have). In its implementation, the function and type names start with hcrngPhilox432. For this RNG, a state is a 128-bit counter with a 64-bit key, and a 2-bit index used to iterate over the four 32-bit outputs generated for each counter value. The counter is represented internally as a vector of four 32-bit unsigned int values and the index, as a single unsigned int value. In the current hcRNG version, the key is the same for all streams, so it is not stored in each stream object but rather hardcoded in the implementation. The period length of 2^130 is divided into 2^28 non-overlapping streams of length Z=2^102. Each stream is further partitioned into 2^36 substreams of length W=2^66. The key (all bits to 0), initial counter and order in which the four outputs per counter value are returned are chosen to generate the same values, in the same order, as Random123's Engine module `[11] <bibliography.html>`_, designed for use with the standard C++11 random library. Note that the function hcrngPhilox432ChangeStreamsSpacing() supports only values of c that are multiples of 4, with either e=0 or e ≥ 2.

*****************************
2.1.5. Function Documentation
*****************************
--------------------------------------------------------------------------------------------------------------------------------------------

2.1.5.1. hcrngCopyStreamCreator()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 

::

 hcrngStreamCreator* hcrngCopyStreamCreator ( const hcrngStreamCreator *  	creator,
                                              hcrngStatus *  	err 
	                                    ) 		

Duplicate an existing stream creator object.

Create an identical copy (a clone) of the stream creator creator. To create a copy of the default creator, put NULL as the creator parameter. All the new stream creators returned by hcrngCopyStreamCreator(NULL, NULL) will create the same sequence of random streams, unless the default stream creator is used to create streams between successive calls to this function.

+------------+-----------------+--------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                  |
+============+=================+==============================================================+
|  [in]      |	creator	       | Stream creator object to be copied, or NULL to copy          |
|            |                 | the default stream creator.                                  |
+------------+-----------------+--------------------------------------------------------------+
|  [out]     |	err	       | Error status variable, or NULL.                              |
+------------+-----------------+--------------------------------------------------------------+

Returns,
    The newly created stream creator object. 

2.1.5.2. hcrngDestroyStreamCreator()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

 hcrngStatus hcrngDestroyStreamCreator ( hcrngStreamCreator *  	creator	) 	

Destroy a stream creator object. Release the resources associated to a stream creator object.

+------------+-----------------+--------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                  |
+============+=================+==============================================================+
|  [out]     |	creator	       | Stream creator object to be destroyed.                       |
+------------+-----------------+--------------------------------------------------------------+

Returns,
    Error status 

2.1.5.3. hcrngRewindStreamCreator()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

 hcrngStatus hcrngRewindStreamCreator ( hcrngStreamCreator *  	creator	) 	

Reset a stream creator to its original initial state, so it can re-create the same streams over again.

+------------+-----------------+--------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                  |
+============+=================+==============================================================+
|  [in]	     | creator	       | Stream creator object to be reset.                           |
+------------+-----------------+--------------------------------------------------------------+

Returns,
    Error status 

2.1.5.4. hcrngSetBaseCreatorState()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::
 
 hcrngStatus hcrngSetBaseCreatorState ( hcrngStreamCreator *  	creator,
                                 	const hcrngStreamState *  	baseState 
	                              ) 		

Change the base stream state of a stream creator.

Set the base state of the stream creator, which can be seen as the seed of the underlying RNG. This will be the initial state (or seed) of the first stream created by this creator. Then, for most conventional RNGs, the initial states of successive streams will be spaced equally, by Z steps in the RNG sequence. The type and size of the baseState parameter depends on the type of RNG. The base state always has a default value, so this function does not need to be invoked.

+------------+-----------------+--------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                  |
+============+=================+==============================================================+
|  [in,out]  |	creator        | Stream creator object.                                       |
+------------+-----------------+--------------------------------------------------------------+
|  [in]	     |  baseState      | New initial base stream state. Can be set to NULL            |
|            |                 | to use the library default.                                  |
+------------+-----------------+--------------------------------------------------------------+

Returns,
    Error status

.. warning:: It is recommended to use the library default base state. 

2.1.5.5. hcrngChangeStreamsSpacing()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

 hcrngStatus hcrngChangeStreamsSpacing ( hcrngStreamCreator *  	creator,
                                 	  int  	e,
                                 	  int  	c 
                                       ) 		

Change the spacing between successive streams.

This function should be used only in exceptional circumstances. It changes the spacing Z between the initial states of the successive streams from the default value to Z=2e+c if e>0, or to Z=c if e=0. One must have e≥0 but c can take negative values. The default spacing values have been carefully selected for each RNG to avoid overlap and dependence between streams, and it is highly recommended not to change them.

+------------+-----------------+--------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                  |
+============+=================+==============================================================+
|  [in,out]  |	creator	       | Stream creator object.                                       |
+------------+-----------------+--------------------------------------------------------------+
|  [in]	     |  e	       | Value of e.                                                  |
+------------+-----------------+--------------------------------------------------------------+
|  [in]	     |  c	       | Value of c.                                                  |
+------------+-----------------+--------------------------------------------------------------+

Returns,
    Error status

.. warning:: It is recommended to use the library default spacing and not to invoke this function. 

2.1.5.6. hcrngAllocStreams()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

 hcrngStream* hcrngAllocStreams ( size_t  	count,
                  		  size_t *  	bufSize,
		                  hcrngStatus *  	err 
	                        ) 		

Reserve memory space for count stream objects, without creating the stream objects. Returns a pointer to the allocated buffer and returns in bufSize the size of the allocated buffer, in bytes.

+------------+-----------------+--------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                  |
+============+=================+==============================================================+
|  [in]	     |  count	       | Number of stream objects to allocate.                        |
+------------+-----------------+--------------------------------------------------------------+
|  [out]     |	bufSize	       | Size in bytes of the allocated buffer, or NULL if not needed.|
+------------+-----------------+--------------------------------------------------------------+
|  [out]     |	err            | Error status variable, or NULL.                              |
+------------+-----------------+--------------------------------------------------------------+

Returns,
    Pointer to the newly allocated buffer. 

2.1.5.7. hcrngDestroyStreams()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

 hcrngStatus hcrngDestroyStreams ( hcrngStream*  streams )

Destroy one or many stream objects. Release the memory space taken by those stream objects.

+------------+-----------------+--------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                  |
+============+=================+==============================================================+
|  [in,out]  |	streams	       | Stream object buffer to be released.                         |
+------------+-----------------+--------------------------------------------------------------+

Returns,
    Error status 

Examples:
    `Multistream.cpp <Multistream.cpp.html>`_, and `RandomArray.cpp <Randomarray.cpp.html>`_.

2.1.5.8. hcrngCreateStreams()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

 hcrngStream* hcrngCreateStreams ( hcrngStreamCreator *  	creator,
                                   size_t  	count,
	                           size_t *  	bufSize,
                                   hcrngStatus *  	err 
                                 ) 		

Allocate memory for and create new RNG stream objects.

Create and return an array of count new streams using the specified creator. This function also reserves the memory space required for the structures and initializes the stream states. It returns in bufSize the size of the allocated buffer, in bytes. To use the default creator, put NULL as the creator parameter. To create a single stream, just put set count to 1.

+------------+-----------------+------------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                      |
+============+=================+==================================================================+
|  [in,out]  |	creator        | Stream creator object, or NULL to use the default stream creator.|
+------------+-----------------+------------------------------------------------------------------+
|  [in]	     |  count          | Size of the array (use 1 for a single stream object).            |
+------------+-----------------+------------------------------------------------------------------+
|  [out]     |	bufSize	       | Size in bytes of the allocated buffer, or NULL if not needed.    |
+------------+-----------------+------------------------------------------------------------------+
|  [out]     |	err            | Error status variable, or NULL.                                  |
+------------+-----------------+------------------------------------------------------------------+

Returns,
    The newly created array of stream object. 

Examples:
    `Multistream.cpp <Multistream.cpp.html>`_, and `RandomArray.cpp <Randomarray.cpp.html>`_.

2.1.5.9. hcrngCreateOverStreams()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

 hcrngStatus hcrngCreateOverStreams ( hcrngStreamCreator *  	creator,
                        	      size_t  	count,
		                      hcrngStream *  	streams 
	                            ) 		

Create new RNG stream objects in already allocated memory.

This function is similar to hcrngCreateStreams(), except that it does not reserve memory for the structure. It creates the array of new streams in the preallocated streams buffer, which could have been reserved earlier via either hcrngAllocStreams() or hcrngCreateStreams(). It permits the client to reuse memory that was previously allocated for other streams.

+------------+-----------------+------------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                      |
+============+=================+==================================================================+
|  [in,out]  |	creator        | Stream creator object, or NULL to use the default stream creator.|
+------------+-----------------+------------------------------------------------------------------+
|  [in]	     |   count         | Size of the array (use 1 for a single stream object).            |
+------------+-----------------+------------------------------------------------------------------+
|  [out]     |	streams	       | Buffer in which the new stream(s) will be stored.                |
+------------+-----------------+------------------------------------------------------------------+

Returns,
    Error status 

2.1.5.10. hcrngCopyStreams()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

 hcrngStream* hcrngCopyStreams ( size_t  	count,
                        	 const hcrngStream *  	streams,
		                 hcrngStatus *  	err 
	                       ) 		

Clone RNG stream objects. Create an identical copy (a clone) of each of the count stream objects in the array streams. This function allocates memory for all the new structures before cloning, and returns a pointer to the new structure.

+------------+-----------------+-------------------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                             |
+============+=================+=========================================================================+
|  [in]	     |  count	       | Number of random number in the array (use 1 for a single stream object).|
+------------+-----------------+-------------------------------------------------------------------------+
|  [in]	     |  streams	       | Stream object or array of stream objects to be cloned.                  |
+------------+-----------------+-------------------------------------------------------------------------+
|  [out]     |	err            | Error status variable, or NULL.                                         |
+------------+-----------------+-------------------------------------------------------------------------+

Returns,
    The newly created stream object or array of stream objects. 

2.1.5.11. hcrngCopyOverStreams()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

 hcrngStatus hcrngCopyOverStreams ( size_t  	count,
		                    hcrngStream *  	destStreams,
		                    const hcrngStream *  	srcStreams 
	                          ) 		

Copy RNG stream objects in already allocated memory [device]. Copy (or restore) the stream objects srcStreams into the buffer destStreams, and each of the count stream objects from the array srcStreams into the buffer destStreams. This function does not allocate memory for the structures in destStreams; it assumes that this has already been done. 

+------------+-----------------+-------------------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                             |
+============+=================+=========================================================================+
|  [in]	     |  count	       | Number of stream objects to copy (use 1 for a single stream object).    |
+------------+-----------------+-------------------------------------------------------------------------+
|  [out]     |	destStreams    | Destination buffer into which to copy (its content will be overwritten).|
+------------+-----------------+-------------------------------------------------------------------------+
|  [in]	     |  srcStreams     | Stream object or array of stream objects to be copied.                  |
+------------+-----------------+-------------------------------------------------------------------------+

Returns,
    Error status

2.1.5.12. hcrngRandomU01()
^^^^^^^^^^^^^^^^^^^^^^^^^^

::

 double hcrngRandomU01 ( hcrngStream *  stream	) 	

Generate the next random value in (0,1) [device]. Generate and return a (pseudo)random number from the uniform distribution over the interval (0,1), using stream. If this stream is from an RNG, the stream state is advanced by one step before producing the (pseudo)random number. By default, the returned value is of type double. But if the option HCRNG_SINGLE_PRECISION is defined, the returned value will be of type float. Setting this option changes the type of the returned value for all RNGs and all functions that use hcrngRandomU01().

+------------+-----------------+--------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                  |
+============+=================+==============================================================+
|  [in,out]  |	stream	       | Stream used to generate the random value.                    |
+------------+-----------------+--------------------------------------------------------------+

Returns,
    A random floating-point value uniformly distributed in (0,1) 

Examples:
    `Multistream.cpp <Multistream.cpp.html>`_, and `RandomArray.cpp <Randomarray.cpp.html>`_.

2.1.5.13. hcrngRandomInteger()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

 int hcrngRandomInteger ( hcrngStream *  	stream,
		          int  	i,
		          int  	j 
	                ) 		

Generate the next random integer value [device]. Generate and return a (pseudo)random integer from the discrete uniform distribution over the integers {i,…,j}, using stream, by calling hcrngRandomU01() once and transforming the output by inversion. That is, it returns i + (int)((j-i+1) * hcrngRandomU01(stream)).

+------------+-----------------+--------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                  |
+============+=================+==============================================================+
|  [in,out]  |	stream	       | Stream used to generate the random value.                    |
+------------+-----------------+--------------------------------------------------------------+
|  [in]	     |  i	       | Smallest integer value (inhcusive).                          |
+------------+-----------------+--------------------------------------------------------------+
|  [in]	     |  j	       | Largest integer value (inhcusive).                           |
+------------+-----------------+--------------------------------------------------------------+


Returns,
    A random integer value uniformly distributed in {i,…,j}.

2.1.5.14. hcrngRandomU01Array()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

 hcrngStatus hcrngRandomU01Array ( hcrngStream *  	stream,
		                   size_t  	count,
		                   double *  	buffer 
	                         ) 		

Fill an array with successive random values in (0,1) [device].Fill preallocated buffer with count successive (pseudo)random numbers. Equivalent to calling hcrngRandomU01() count times to fill the buffer. If HCRNG_SINGLE_PRECISION is defined, the buffer argument is of type float and will be filled by count values of type float instead.

+------------+-----------------+--------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                  |
+============+=================+==============================================================+
|  [in,out]  |	stream	       | Stream used to generate the random values.                   |
+------------+-----------------+--------------------------------------------------------------+
|  [in]	     |  count	       | Number of values in the array.                               |
+------------+-----------------+--------------------------------------------------------------+
|  [out]     |	buffer	       | Destination buffer (must be pre-allocated).                  |
+------------+-----------------+--------------------------------------------------------------+

Returns,
    Error status 

2.1.5.15. hcrngRandomIntegerArray()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

 hcrngStatus hcrngRandomIntegerArray ( hcrngStream *  	stream,
                                       int  	i,
		                       int  	j,
		                       size_t  	count,
		                       int *  	buffer 
	                             ) 		

Fill an array with successive random integer values [device].Same as hcrngRandomU01Array(), but for integer values in {i,…,j}. Equivalent to calling hcrngRandomInteger() count times to fill the buffer.

+------------+-----------------+--------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                  |
+============+=================+==============================================================+
|  [in,out]  |	stream	       | Stream used to generate the random values.                   |
+------------+-----------------+--------------------------------------------------------------+
|  [in]      |	i	       | Smallest integer value (inhcusive).                          |
+------------+-----------------+--------------------------------------------------------------+
|  [in]      |	j	       | Largest integer value (inhcusive).                           |
+------------+-----------------+--------------------------------------------------------------+
|  [in]	     |  count	       | Number of values in the array.                               |
+------------+-----------------+--------------------------------------------------------------+
|  [out]     |	buffer         | Destination buffer (must be pre-allocated).                  |
+------------+-----------------+--------------------------------------------------------------+

Returns,
    Error status 

2.1.5.16. hcrngRewindStreams()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

 hcrngStatus hcrngRewindStreams ( size_t  	count,
                           	  hcrngStream *  	streams 
	                        ) 		

Reinitialize streams to their initial states [device]. Reinitialize all the streams in streams to their initial states. The current substream also becomes the initial one.

+------------+-----------------+---------------------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                               |
+============+=================+===========================================================================+
|  [in]      |	count          | Number of stream objects in the array (use 1 for a single stream object). |
+------------+-----------------+---------------------------------------------------------------------------+
|  [in,out]  |	streams	       | Stream object or array of stream objects to be reset to the               |
|            |                 | start of the stream(s).                                                   |
+------------+-----------------+---------------------------------------------------------------------------+

Returns,
    Error status

.. warning:: This function can be slow on the device, because it reads the initial state from global memory. 

2.1.5.17. hcrngRewindSubstreams()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

 hcrngStatus hcrngRewindSubstreams ( size_t  	count,
                         	     hcrngStream *  	streams 
	                            ) 		

Reinitialize streams to their initial substream states [device]. Reinitialize all the streams in streams to the initial states of their current substream.

+------------+-----------------+---------------------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                               |
+============+=================+===========================================================================+
|  [in]	     |  count	       | Number of stream objects in the array (use 1 for a single stream object). |
+------------+-----------------+---------------------------------------------------------------------------+
|  [in,out]  |	streams	       | Stream object or array of stream objects to be reset to the beginning     |
|            |                 | of the current substream(s).                                              |
+------------+-----------------+---------------------------------------------------------------------------+

Returns,
    Error status

Examples:
    `Multistream.cpp <Multistream.cpp.html>`_

2.1.5.18. hcrngForwardToNextSubstreams()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

 hcrngStatus hcrngForwardToNextSubstreams ( size_t  	count,
                               		    hcrngStream *  	streams 
	                                  ) 		

Advance streams to the next substreams [device]. Reinitialize all the streams in streams to the initial states of their next substream. The current states and the initial states of the current substreams are changed.

+------------+-----------------+---------------------------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                                     |
+============+=================+=================================================================================+
|  [in]	     |  count	       | Number of stream objects in the array (use 1 for a single stream object).       |
+------------+-----------------+---------------------------------------------------------------------------------+
|  [in,out]  |	streams        | Stream object or array of stream objects to be advanced to the next substream(s)|
+------------+-----------------+---------------------------------------------------------------------------------+

Returns,
    Error status

Examples:
    `Multistream.cpp <Multistream.cpp.html>`_

2.1.5.19. hcrngMakeSubstreams()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

 hcrngStream* hcrngMakeSubstreams ( hcrngStream *  	stream,
                         	    size_t  	count,
		                    size_t *  	bufSize,
		                    hcrngStatus *  	err 
	                          ) 		

Allocate and make an array of substreams of a stream. 

Make and return an array of count copies of stream, whose current (and initial substream) states are the initial states of count successive substreams of stream. The first substream in the returned array is simply a copy of stream. This function also reserves the memory space required for the structures and initializes the stream states. It returns in bufSize the size of the allocated buffer, in bytes. To create a single stream, just set count to 1. When this function is invoked, the substream state and initial state of stream are advanced by count substreams.

2.1.5.20. hcrngMakeOverSubstreams()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

 hcrngStatus hcrngMakeOverSubstreams ( hcrngStream *  	stream,
                           	       size_t  	count,
		                       hcrngStream *  	substreams 
	                             ) 		

Make an array of substreams of a stream.

This function is similar to hcrngMakeStreams(), except that it does not reserve memory for the structure. It creates the array of new streams in the preallocated substreams buffer, which could have been reserved earlier via either hcrngAllocStreams(), hcrngMakeSubstreams() or hcrngCreateStreams(). It permits the client to reuse memory that was previously allocated for other streams.

2.1.5.21. hcrngAdvanceStreams()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

 hcrngStatus hcrngAdvanceStreams ( size_t  	count,
                               	   hcrngStream *  	streams,
		                   int  	e,
		                   int  	c 
	                         ) 		

Advance the state of streams by many steps.

This function should be used only in very exceptional circumstances. It advances the state of the streams in array streams by k steps, without modifying the states of other streams, nor the initial stream and substream states for those streams. If e>0, then k=2e+c; if e<0, then k=−2|e|+c; and if e=0, then k=c. Note that c can take negative values. We discourage the use of this procedure to customize the length of streams and substreams. It is better to use the default spacing, which has been carefully selected for each RNG type.

+------------+-----------------+---------------------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                               |
+============+=================+===========================================================================+
|  [in]	     |  count	       | Number of stream objects in the array (use 1 for a single stream object). |
+------------+-----------------+---------------------------------------------------------------------------+
|  [in,out]  |  streams	       | Stream object or array of stream objects to be advanced.                  |
+------------+-----------------+---------------------------------------------------------------------------+
|  [in]	     |  e	       | Value of e.                                                               |
+------------+-----------------+---------------------------------------------------------------------------+
|  [in]	     |  c	       | Value of c.                                                               |
+------------+-----------------+---------------------------------------------------------------------------+

Returns,
    Error status

.. warning:: Check the implementation for all cases e>0, e=0 and e<0. 

2.1.5.22. hcrngDeviceRandomU01Array()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

 hcrngStatus hcrngDeviceRandomU01Array_single ( hc::accelerator_view &accl_view, 
                                                size_t 		    streamCount, 
                                                hcrngStream*         streams,
                                                size_t               numberCount, 
                                                float*               outBuffer, 
                                                int                  streamlength = 0, 
                                                size_t               streams_per_thread = 1 )
 hcrngStatus hcrngDeviceRandomU01Array_double ( hc::accelerator_view &accl_view, 
                                                size_t              streamCount, 
                                                hcrngStream*         streams,
                                                size_t               numberCount, 
                                                double*              outBuffer, 
                                                int                  streamlength = 0, 
                                                size_t               streams_per_thread = 1 )

Fill a buffer of random numbers.

Fill the buffer pointed to by outBuffer with numberCount uniform random numbers of type double (or of type float if HCRNG_SINGLE_PRECISION is defined), using streamCount work items. In the current implementation, numberCount must be a multiple of streamCount. It is adviced to call the kernel depending on the type of output buffer. Kernels of type float and double has suffixes "_single" and "_double" respectively. 

+------------+---------------------+-----------------------------------------------------------------------------------+
|  In/out    |  Parameters         | Description                                                                       |
+============+=====================+===================================================================================+
|  [in]      |  accl_view          | `Using accelerator and accelerator_view Objects                                   |  
|            |                     | <https://msdn.microsoft.com/en-us/library/hh873132.aspx>`_                        |
+------------+---------------------+-----------------------------------------------------------------------------------+
|  [in]	     |  streamCount        | Number of streams in stream_array.                                                |
+------------+---------------------+-----------------------------------------------------------------------------------+
|  [in]	     |  streams	           | HCC device pointer that contains an array of stream objects.                      |
+------------+---------------------+-----------------------------------------------------------------------------------+
|  [in]	     |  numberCount        | Number of random number to store in the device pointer.                           |
+------------+---------------------+-----------------------------------------------------------------------------------+
|  [out]     |	outBuffer          | HCC device pointer in which the generated numbers will be stored.                 |
+------------+---------------------+-----------------------------------------------------------------------------------+
|  [in]      |  stream_length      | [Default argument] The length of the subtsream.                                   |
|            |                     | stream_length       = 0   ( do not use substreams )                               |
|            |                     | stream_length       = > 0 ( go to next substreams after stream_length values)     |
|            |                     | stream_length       = < 0 ( restart substream after stream_length values )        |
+------------+---------------------+-----------------------------------------------------------------------------------+
|  [in]      |  streams_per_thread | [Default argument] Number of streams a thread should handle. Must be a multiple   |
|            |                     | of streamCount.                                                                   |
+------------+---------------------+-----------------------------------------------------------------------------------+
  
Returns,
    Error status

Examples:
    `Multistream.cpp <Multistream.cpp.html>`_, and `RandomArray.cpp <Randomarray.cpp.html>`_.

.. warning:: In the current implementation, numberCount must be a multiple of streamCount and streams_per_thread must be a multiple of streamCount. The array streams is left unchanged, as there is no write-back from the device code. stream_length and streams_per_thread are default arguments and can be used for multistream random number generation.

2.1.5.23. hcrngWriteStreamInfo()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

 hcrngStatus hcrngWriteStreamInfo ( const hcrngStream *  	stream,
		                    FILE *  	file 
	                          ) 		

Format and output information about a stream object to a file.

+------------+-----------------+---------------------------------------------------------------------------+
|  In/out    |  Parameters     | Description                                                               |
+============+=================+===========================================================================+
|  [in]	     |  stream	       | Stream object about which to write information.                           |
+------------+-----------------+---------------------------------------------------------------------------+
|  [in]	     |  file	       | File to which to output. Can be set to stdout or stderr                   |
|            |                 | for standard output and error.                                            |
+------------+-----------------+---------------------------------------------------------------------------+

Returns,
    Error status 


