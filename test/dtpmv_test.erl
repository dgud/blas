-module(dtpmv_test).
-include_lib("eunit/include/eunit.hrl").

dtpmv_test()->
    N = 3,
    Ap = blas:new(chain:ltb(d, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
    X = blas:new(chain:ltb(d, [-0.25,-0.125,0.5])),
    blas:run({dtpmv, blasRowMajor, blasUpper, blasNoTrans, blasNonUnit, N, Ap, X, 1}),
    [1.0,2.0,3.0] = chain:btl(d, blas:to_bin(X)).