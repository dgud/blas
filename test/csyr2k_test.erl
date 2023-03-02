% Performs a rank-2k update of a complex symmetric matrix.

-module(csyr2k_test).
-include_lib("eunit/include/eunit.hrl").

csyr2k_test()->
    N = 2,
    A = blas:new(chain:ltb(s, [1,1, 2,2, 0,0, 3,3])),
    B = blas:new(chain:ltb(s, [0,1, 2,0, 0,0, 0,0])),
    C = blas:new(chain:ltb(s, [0,0, 0,0, 0,0, 1,0])),
    
    blas:run({csyr2k, blasRowMajor, blasUpper, blasNoTrans, N, N, chain:ltb(s,[1.5, 1.0]), A, N, B, N, chain:ltb(s,[0.5, 0.0]), C, N}),
    [-1.0, 21.0,  3.0, 15.0,  0.0, 0.0,  0.5,  0.0] = chain:btl(s, blas:to_bin(C)).