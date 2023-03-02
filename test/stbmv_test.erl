-module(stbmv_test).
-include_lib("eunit/include/eunit.hrl").

stbmv_test()->
    N = 4,
    K = 1,
    A = blas:new(chain:ltb(s, [0,1, 0,2, 0,3, 0,0])),
    Lda = 2,
    X = blas:new(chain:ltb(s, [1,1,1,1])),
    Incx = 1,
    blas:run({stbmv, blasRowMajor, blasUpper, blasNoTrans, blasUnit, N, K, A, Lda, X, Incx}),
    [2.0, 3.0, 4.0, 1.0] = chain:btl(s, blas:to_bin(X)).