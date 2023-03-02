% Solves a system of linear equations with a packed triangular matrix and a complex vector using back substitution.

-module(ctpsv_test).
-include_lib("eunit/include/eunit.hrl").

ctpsv_test()->
    N = 3,
    A = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
    X = blas:new(chain:ltb(s, [1.0, 2.0, 3.0])),
    blas:run({ctpsv, blasRowMajor, blasUpper, blasNoTrans, blasNonUnit, N, A, X, 1}),
    [-0.25,-0.125,0.5] = chain:btl(s, blas:to_bin(X)).