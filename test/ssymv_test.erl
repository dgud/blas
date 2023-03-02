-module(ssymv_test).
-include_lib("eunit/include/eunit.hrl").

ssymv_test()->
    N = 3,
    A = blas:new(chain:ltb(s, [1, 2, 3, 2, 4, 5, 3, 5, 6])),
    X = blas:new(chain:ltb(s, [1, 2, 3])),
    Y = blas:new(chain:ltb(s, [0, 0, 0])),
    blas:run({ssymv, blasRowMajor, blasUpper, N, 1.0, A, N, X, 1, 0.0, Y, 1}),
    [14.0, 25.0, 31.0] = chain:btl(s, blas:to_bin(Y)).