-module(dtpsv_test).
-include_lib("eunit/include/eunit.hrl").

dtpsv_test()->
    N = 3,
    A = blas:new(chain:ltb(d, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
    X = blas:new(chain:ltb(d, [1.0, 2.0, 3.0])),
    blas:run({dtpsv, blasRowMajor, blasUpper, blasNoTrans, blasNonUnit, N, A, X, 1}),
    [-0.25,-0.125,0.5] = chain:btl(d, blas:to_bin(X)).