% Performs a matrix-matrix multiplication between a complex matrix and a triangular complex matrix.

-module(ctrmm_test).
-include_lib("eunit/include/eunit.hrl").

ctrmm_test()->
    M = 2,
    N = 2,
    A = blas:new(chain:ltb(s, [1,2, 3,4, 0,0, 5,6])),
    B = blas:new(chain:ltb(s, [1,2, 3,4, 0,0, 1,1])),
    blas:run({ctrmm, blasRowMajor, blasLeft, blasUpper, blasNoTrans, blasNonUnit, M, N, chain:ltb(c,[1.0, 2.0]), A, M, B, M}),
    [-11.0,  -2.0, -40.0,   5.0,   0.0,   0.0, -23.0,   9.0] = chain:btl(s, blas:to_bin(B)).