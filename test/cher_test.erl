% Performs a rank-1 update of a complex Hermitian matrix.

-module(cher_test).
-include_lib("eunit/include/eunit.hrl").

cher_test()->
    N = 2,
    X = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0])),
    A = blas:new(chain:ltb(s, [2,0, 1,-2, 0,0, 4,5])),
    blas:run({cher, blasRowMajor, blasUpper, N, chain:ltb(s, [2.0, 3.0]), X, 1, A, N}),
    [-16.0,  -1.0, -39.0,   3.0, -39.0,   7.0, -82.0,  32.0] =:= chain:btl(s, blas:to_bin(A)).