% Performs a rank-k update of a complex symmetric matrix.

-module(csyrk_test).
-include_lib("eunit/include/eunit.hrl").

csyrk_test()->
    N = 2,
    A = blas:new(chain:ltb(s, [1,2, 3,4, 5,6, 7,8])),
    C = blas:new(chain:ltb(s, [0,0, 0,0, 0,0, 0,0])),
    blas:run({csyrk, blasRowMajor, blasUpper, blasNoTrans, N, N, chain:ltb(s,[1.0, 2.0]), A, N,  chain:ltb(s,[0.0, 1.0]), C, N}),
    [ -66.0,8.0,  -154.0,32.0,  0.0,0.0, -370.0,120.0] = chain:btl(s, blas:to_bin(C)).