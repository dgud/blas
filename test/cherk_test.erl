% Performs a rank-k update of a complex Hermitian matrix.

-module(cherk_test).
-include_lib("eunit/include/eunit.hrl").

cherk_test()->
    N = 3,
    A = blas:new(chain:ltb(s, [1.0, 2.0, 3.0,  2.0, 4.0, 5.0,  3.0, 5.0, 6.0])),
    C = blas:new(chain:ltb(s, [0,0,0,0,0,0,0,0,0])),
    blas:run({cherk, blasRowMajor, blasUpper, blasNoTrans, N, N, chain:ltb(s,[1.0, 2.0]), A, N, chain:ltb(s,[0.0, 0.0]), C, N}),
    [14, 25, 31, 0, 45, 56, 0, 0, 70] =:= chain:btl(s, blas:to_bin(C)).