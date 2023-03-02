% Performs a matrix-vector multiplication between a complex Hermitian matrix and a complex vector.

-module(chemv_test).
-include_lib("eunit/include/eunit.hrl").

chemv_test()->
    N = 3,
    A = blas:new(chain:ltb(s, [1, 2, 3, 2, 4, 5, 3, 5, 6])),
    X = blas:new(chain:ltb(s, [1, 2, 3])),
    Y = blas:new(chain:ltb(s, [0, 0, 0])),
    blas:run({chemv, blasRowMajor, blasUpper, N, 1.0, A, N, X, 1, 0.0, Y, 1}),
    [14, 25, 31] =:= chain:btl(s, blas:to_bin(Y)).