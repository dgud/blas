-module(strsv_test).
-include_lib("eunit/include/eunit.hrl").

strsv_test()->
    N = 3,
    A = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0])),
    X = blas:new(chain:ltb(s, [1.0, 2.0, 3.0])),
    blas:run({strsv, blasRowMajor, blasUpper, blasNoTrans, blasNonUnit, N, A, X, 1}),
    [-0.166667, 0.833333, 0.5] = chain:btl(s, X).