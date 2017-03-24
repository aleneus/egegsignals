.. egegsignals documentation master file, created by
   sphinx-quickstart on Thu Mar 23 15:28:29 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to egegsignals's documentation!
=======================================

Electrogastrography (EGG) and electrogastroenterography (EGEG) are electrophysiological methods for diagnostics of stomach and gastrointestinal tract motility.

This package provides a processing of EGG and EGEG signals.

Requirements
------------

* numpy
* scipy

What's new in 1.0.0
-------------------

* Interfaces of base parameters functions changed. Now that functions need spectrum and don't need signal x more.
* Short-time Fourier transform implemented.
* DFIC parameter added.
* Help functions spectrum and expand_to added.
* Additional tests added.

.. toctree::
   :maxdepth: 1

   What's new in 0.2.0 <0.2.0>

Contents:

.. toctree::
   :maxdepth: 2

   parameters
   hfart

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

