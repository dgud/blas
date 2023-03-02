-module(dsyrk_test).
-include_lib("eunit/include/eunit.hrl").

dsyrk_test()->
    N = 3,
    A = blas:new(chain:ltb(d, [1.0, 2.0, 3.0,  2.0, 4.0, 5.0,  3.0, 5.0, 6.0])),
    C = blas:new(chain:ltb(d, [0,0,0,0,0,0,0,0,0])),
    blas:run({dsyrk, blasRowMajor, blasUpper, blasNoTrans, N, N, 1.0, A, N, 0.0, C, N}),
    [14.0, 25.0, 31.0, 0.0, 45.0, 56.0, 0.0, 0.0, 70.0] = chain:btl(d, blas:to_bin(C)).