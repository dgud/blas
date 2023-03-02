% Performs a matrix-matrix multiplication between two complex matrices.

-module(chemm_test).
-include_lib("eunit/include/eunit.hrl").

chemm_test()->
    M = 2,
    N = 2,
    A = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 2.0, 1.0])),
    B = blas:new(chain:ltb(s, [1.0, 0.0, 3.0, 5.0, 4.0, 1.0, 2.0, 0.0])),
    C = blas:new(chain:ltb(s, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
    blas:run({chemm, blasRowMajor, blasLeft, blasUpper, M, N, chain:ltb(s,[1.0, 0.0]), A, M, B, N, chain:ltb(s,[0.0, 0.0]), C, N}),
    [9.0, 19.0,  9.0, 13.0, 11.0, -2.0, 33.0,  3.0] = chain:btl(s, blas:to_bin(C)).