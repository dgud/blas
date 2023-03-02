% Performs a rank-k update of a complex symmetric matrix.

-module(csyrk_test).
-include_lib("eunit/include/eunit.hrl").

csyrk_test()->
    N = 3,
    A = blas:new(chain:ltb(s, [1.0, 2.0, 3.0,  2.0, 4.0, 5.0,  3.0, 5.0, 6.0])),
    C = blas:new(chain:ltb(s, [0,0,0,0,0,0,0,0,0])),
    blas:run({csyrk, blasRowMajor, blasUpper, blasNoTrans, N, N, 1.0, A, N, 0.0, C, N}),
    [14, 25, 31, 0, 45, 56, 0, 0, 70] =:= chain:btl(s, blas:to_bin(C)).