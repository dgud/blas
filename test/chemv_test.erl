% Performs a matrix-vector multiplication between a complex Hermitian matrix and a complex vector.

-module(chemv_test).
-include_lib("eunit/include/eunit.hrl").

chemv_test()->
    N = 2,
    A = blas:new(chain:ltb(s, [2,0, 1,-2, 1,2, 4,5])),
    X = blas:new(chain:ltb(s, [1.0,2.0,3.0,4.0])),
    Y = blas:new(chain:ltb(s, [5.0,6.0,7.0,8.0])),
    blas:run({chemv, blasRowMajor, blasUpper, N, chain:ltb(s,[1.0, 0.0]), A, N, X, 1, chain:ltb(s,[0.0, 2.0]), Y, 1}),
    [ 1.0,  12.0, -27.0,  49.0] == chain:btl(s, blas:to_bin(Y)).