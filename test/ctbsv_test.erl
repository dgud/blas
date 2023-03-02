% Solves a system of linear equations with a triangular band matrix and a complex vector using back substitution.

-module(ctbsv_test).
-include_lib("eunit/include/eunit.hrl").

ctbsv_test()->
    N = 2,
    K = 1,
    A = blas:new(chain:ltb(c, [1,1, 2,2, 0,0, 3,3])),
    X = blas:new(chain:ltb(c, [3,2,1,0])),
    blas:run({ctbsv, blasRowMajor, blasUpper, blasNoTrans, blasUnit, N, K, A, 2, X, 1}),
    [1.0,0.0,1.0,0.0] = chain:btl(c, blas:to_bin(X)).