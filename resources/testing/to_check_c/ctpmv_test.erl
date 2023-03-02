% Performs a matrix-vector multiplication between a packed triangular matrix and a complex vector.

-module(ctpmv_test).
-include_lib("eunit/include/eunit.hrl").

ctpmv_test()->
    N = 3,
    Ap = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
    X = blas:new(chain:ltb(s, [-0.25,-0.125,0.5])),
    blas:run({ctpmv, blasRowMajor, blasUpper, blasNoTrans, blasNonUnit, N, Ap, X, 1}),
    [1.0,2.0,3.0] = chain:btl(s, blas:to_bin(X)).