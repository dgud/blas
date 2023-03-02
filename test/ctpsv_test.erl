% Solves a system of linear equations with a packed triangular matrix and a complex vector using back substitution.

-module(ctpsv_test).
-include_lib("eunit/include/eunit.hrl").

ctpsv_test()->
    N = 2,
    A = blas:new(chain:ltb(c, [1,1, 2,2, 3,3])),
    X = blas:new(chain:ltb(c, [-3, 17, -3, 21])),
    blas:run({ctpsv, blasRowMajor, blasUpper, blasNoTrans, blasNonUnit, N, A, X, 1}),
    [1.0,2.0, 3.0,4.0] = chain:btl(c, blas:to_bin(X)).