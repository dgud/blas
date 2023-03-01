-module(stpsv_test).
-include_lib("eunit/include/eunit.hrl").

stpsv_test()->
    N = 3,
    A = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
    X = blas:new(chain:ltb(s, [1.0, 2.0, 3.0])),
    blas:run({stpsv, blasRowMajor, blasUpper, blasNoTrans, blasNonUnit, N, A, X, 1}),
    [-1.0, 0.5, 1.0] = chain:btl(s, X).