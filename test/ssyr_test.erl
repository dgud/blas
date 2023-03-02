-module(ssyr_test).
-include_lib("eunit/include/eunit.hrl").

ssyr_test()->
    N = 3,
    X = blas:new(chain:ltb(s, [1.0, 2.0, 3.0])),
    A = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0])),
    blas:run({ssyr, blasRowMajor, blasUpper, N, 2.0, X, 1, A, N}),
    [3.0,  6.0,  9.0,  0.0, 12.0, 17.0,  0.0, 0.0, 24.0] = chain:btl(s, blas:to_bin(A)).