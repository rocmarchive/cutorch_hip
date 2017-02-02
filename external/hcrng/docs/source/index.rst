===================
hcRNG Documentation
===================

************
Introduction
************
-------------------------------------------------------------------------------------------------------------------------------------------

The hcRNG library is an implementation of uniform random number generators targetting the AMD heterogenous hardware via HCC compiler runtime. The computational resources of underlying AMD heterogenous compute gets exposed and exploited through the HCC C++ frontend. Refer `here <https://bitbucket.org/multicoreware/hcc/wiki/Home>`_ for more details on HCC compiler.

The following list enumerates the current set of RNG generators that are supported so far.

 1. MRG31k3p `[4] <bibliography.html>`_
 2. MRG32k3a `[7] <bibliography.html>`_
 3. LFSR113  `[8] <bibliography.html>`_
 4. Philox-4x32-10 `[11] <bibliography.html>`_

Library provides multiple streams that are created on the host computer and used to generate random numbers either on the host or on computing devices by work items. Such multiple streams are essential for parallel simulation `[6] <bibliography.html>`_ and are often useful as well for simulation on a single processing element (or within a single work item), for example when comparing similar systems via simulation with common random numbers `[1] <bibliography.html>`_, `[9] <bibliography.html>`_, `[10] <bibliography.html>`_, `[5] <bibliography.html>`_ . Streams can also be divided into segments of equal length called substreams, as in `[2] <bibliography.html>`_, `[5] <bibliography.html>`_, `[10] <bibliography.html>`_ .


.. _user-docs:

.. toctree::
   :maxdepth: 2
   
   Getting_Started

.. _api-ref:

.. toctree::
   :maxdepth: 2

   API_reference

.. Index::
   index
   hcRNG_template
   examples
   bibliography
