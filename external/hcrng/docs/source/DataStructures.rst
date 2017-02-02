****************
2.2. hcRNG TYPES
****************

2.2.1. Enumerations
^^^^^^^^^^^^^^^^^^^
--------------------------------------------------------------------------------------------------------------------------------------------

2.2.1.1. HCRNG STATUS (hcrngStatus)
-----------------------------------

::

  typedef enum hcrngStatus_ {
    HCRNG_SUCCESS                  = 0,
    HCRNG_OUT_OF_RESOURCES         = -1,
    HCRNG_INVALID_VALUE            = -2,
    HCRNG_INVALID_RNG_TYPE         = -3,
    HCRNG_INVALID_STREAM_CREATOR   = -4,
    HCRNG_INVALID_SEED             = -5,
    HCRNG_FUNCTION_NOT_IMPLEMENTED = -6
  } hcrngStatus;

| This enumeration is the set of hcRNG error codes.
+-------------------------------------+--------------------------------------------------------------------------------+
| Enumerator                                                                                                           |
+=====================================+================================================================================+
| HCRNG_SUCCESS                       | the operation completed successfully.                                          |
+-------------------------------------+--------------------------------------------------------------------------------+    
| HCRNG_OUT_OF_RESOURCES              | resource allocation failed.                                                    |
+-------------------------------------+--------------------------------------------------------------------------------+
| HCRNG_INVALID_VALUE                 | unsupported numerical value was passed to function.                            |
+-------------------------------------+--------------------------------------------------------------------------------+
| HCRNG_INVALID_RNG_TYPE              | unsupported rng type specified.                                                |
+-------------------------------------+--------------------------------------------------------------------------------+
| HCRNG_INVALID_STREAM_CREATOR        | Stream creator is invalid.                                                     |
+-------------------------------------+--------------------------------------------------------------------------------+
| HCRNG_INVALID_SEED                  | Seed value is greater than particular generators' predefined values.           |
+-------------------------------------+--------------------------------------------------------------------------------+
| HCRNG_FUNCTION_NOT_IMPLEMENTED      | an internal hcRNG function not implemented.                                    |
+-------------------------------------+--------------------------------------------------------------------------------+

2.2.2. Data Structures
^^^^^^^^^^^^^^^^^^^^^^
-------------------------------------------------------------------------------------------------------------------------------------------

2.2.2.1. hcrngStreamState
-------------------------

::

 struct hcrngStreamState

Stream state [host/device]. Contains the state of a random stream. The definition of a state depends on the type of generator.

Examples:
    `Multistream.cpp <Multistream.cpp.html>`_.

2.2.2.2. hcrngStream
--------------------

::

 struct hcrngStream

Stream object [host/device]. A structure that contains the current information on a stream object. It generally depends on the type of generator. It typically stores the current state, the initial state of the stream, and the initial state of the current substream.

Examples:
    `Multistream.cpp <Multistream.cpp.html>`_, and `RandomArray.cpp <Randomarray.cpp.html>`_.

2.2.2.3. hcrngStreamCreator
---------------------------

::

 struct hcrngStreamCreator

Stream creator object. For each type of RNG, there is a single default creator of streams, and this should be sufficient for most applications. Multiple creators could be useful for example to create the same successive stream objects multiple times in the same order, instead of storing them in an array and reusing them, or to create copies of the same streams in the same order at different locations in a distributed system, e.g., when simulating similar systems with common random numbers. Stream creators are created according to an abstract factory pattern.

2.2.3. Helper Functions
^^^^^^^^^^^^^^^^^^^^^^^
-------------------------------------------------------------------------------------------------------------------------------------------

2.2.3.1. hcrngGetLibraryRoot()
------------------------------

::

 const char* hcrngGetLibraryRoot ()

Retrieve the library installation path.

Returns,
    Value of the HCRNG_ROOT environment variable, if defined, else the current directory (.) of execution of the program.


2.2.3.2. hcrngGetErrorString()  
------------------------------

::

 const char* hcrngGetErrorString () 	

Retrieve the last error message. The buffer containing the error message is internally allocated and must not be freed by the client.

Returns,
    Error message or NULL.  
