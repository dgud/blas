BLAS for erlang
===============

blas is an erlang nif binding to BLAS, Basic Linear Algebra
Subprograms.

Blas subroutines are a de facto standard API for linear
algebra libraries and routines.

Currently only supports the erlang native floats i.e.
the 64b floats.

Requirements
------------

Needs a blas library to link against.


Status
------

First rough implementation, interruptable/dirty scheduler nifs, range tests in the
nif, Makefiles/configure, tests and documentation needs more work.

Would also be nice to have support for sparse data and LAPACK.

Currently only limited tested with ATLAS on Ubuntu-14.04


