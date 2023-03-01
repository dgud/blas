-module(ssyrk_test).
-include_lib("eunit/include/eunit.hrl").

ssyrk_test()->
    N = 3,
    A = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])),
    C = blas:new(chain:ltb(s, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])),
    blas:run({ssyrk, blasRowMajor, blasUpper, blasNoTrans, N, N, 1.0, A, N, 0.0, C, N}),
    [90.0, 66.0, 42.0, 0.0, 114.0, 78.0, 0.0, 0.0, 138.0] = chain:btl(s, C).