-module(ssymm_test).
-include_lib("eunit/include/eunit.hrl").

ssymm_test()->
    M = 3,
    N = 4,
    A = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])),
    Lda = m,
    B = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])),
    Ldb = m,
    C = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])),
    Ldc = m,
    blas:run({ssymm, blasRowMajor, blasLeft, blasUpper, M, N, 2.0, A, Lda, B, Ldb, 3.0, C, Ldc}),
    [37.0, 44.0, 51.0, 58.0, 74.0, 88.0, 102.0, 116.0, 111.0, 132.0, 153.0, 174.0] = chain:btl(s, C).