% Performs a matrix-vector multiplication between a triangular band matrix and a complex vector.

-module(ctbmv_test).
-include_lib("eunit/include/eunit.hrl").

ctbmv_test()->
    N = 2,
    K = 1,
    A = blas:new(chain:ltb(s, [1,1, 2,2, 0,0, 3,3])),
    X = blas:new(chain:ltb(s, [1.0,0.0,1.0,0.0])),
    blas:run({ctbmv, blasRowMajor, blasUpper, blasNoTrans, blasUnit, N, K, A, 2, X, 1}),
    [3.0,2.0,1.0,0.0] = chain:btl(s, blas:to_bin(X)).