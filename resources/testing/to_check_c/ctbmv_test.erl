% Performs a matrix-vector multiplication between a triangular band matrix and a complex vector.

-module(ctbmv_test).
-include_lib("eunit/include/eunit.hrl").

ctbmv_test()->
    N = 4,
    K = 1,
    A = blas:new(chain:ltb(s, [0,1, 0,2, 0,3, 0,0])),
    Lda = 2,
    X = blas:new(chain:ltb(s, [1,1,1,1])),
    Incx = 1,
    blas:run({ctbmv, blasRowMajor, blasUpper, blasNoTrans, blasUnit, N, K, A, Lda, X, Incx}),
    [2, 3, 4, 1] =:= chain:btl(s, blas:to_bin(X)).