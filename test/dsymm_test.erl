-module(dsymm_test).
-include_lib("eunit/include/eunit.hrl").

dsymm_test()->
    M = 3,
    N = 4,
    A = blas:new(chain:ltb(d, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])),
    Lda = M,
    B = blas:new(chain:ltb(d, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])),
    C = blas:new(chain:ltb(d, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])),
    blas:run({dsymm, blasRowMajor, blasLeft, blasUpper, M, N, 2.0, A, Lda, B, N, 3.0, C, N}),
    [79,  94, 109, 124, 177, 206, 235, 264, 255, 294, 333, 372] =:= chain:btl(d, blas:to_bin(C)).