% Performs a rank-2k update of a complex Hermitian matrix.

-module(cher2k_test).
-include_lib("eunit/include/eunit.hrl").

cher2k_test()->
    N = 2,
    A = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 3.0, 5.0, 6.0, 2.0, 1.0])),
    B = blas:new(chain:ltb(s, [1.0, 0.0, 3.0, 5.0, 4.0, 1.0, 2.0, 0.0])),
    C = blas:new(chain:ltb(s, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
    
    blas:run({cher2k, blasRowMajor, blasUpper, blasNoTrans, N, N, chain:ltb(s, [1.0, 2.0]), A, N, B, N, chain:ltb(s,[0.0, 0.5]), C, N}),
    [66.0,   0.0,   4.0,   6.0,   0.0,  0.0, -24.0,   0.0] = chain:btl(s, blas:to_bin(C)).