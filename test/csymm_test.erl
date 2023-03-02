% Performs a matrix-matrix multiplication between a complex symmetric matrix and a complex matrix.

-module(csymm_test).
-include_lib("eunit/include/eunit.hrl").

csymm_test()->
    M = 2,
    N = 2,
    A = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])),
    B = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])),
    C = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])),
    blas:run({csymm, blasRowMajor, blasLeft, blasUpper, M, N, chain:ltb(c, [2.0, 3.0]), A, M, B, N, chain:ltb(c, [3.0, 0.0]), C, N}),
    [79,  94, 109, 124, 177, 206, 235, 264, 255, 294, 333, 372] =:= chain:btl(s, blas:to_bin(C)).