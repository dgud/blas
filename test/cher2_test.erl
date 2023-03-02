% Performs a rank-2 update of a complex Hermitian matrix.

-module(cher2_test).
-include_lib("eunit/include/eunit.hrl").

cher2_test()->
    N = 2,
    X = blas:new(chain:ltb(s, [1.0,2.0,3.0,4.0])),
    Y = blas:new(chain:ltb(s, [5.0,6.0,7.0,8.0])),
    A = blas:new(chain:ltb(s, [2,0, 1,-2, 1,2, 4,5])),
    blas:run({cher2, blasRowMajor, blasUpper, N, chain:ltb(s,[2.0, 0.0]), X, 1, Y, 1, A, N}),
    [70.0,   0.0, 125.0,   6.0, 125.0,  -6.0, 216.0,   5.0] == chain:btl(s, blas:to_bin(A)).

