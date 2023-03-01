-module(dtbmv_test).
-include_lib("eunit/include/eunit.hrl").

dtbmv_test()->
    N = 4,
    K = 1,
    A = blas:new(chain:ltb(d, [0,1, 0,2, 0,3, 0,0])),
    Lda = 2,
    X = blas:new(chain:ltb(d, [1,1,1,1])),
    Incx = 1,
    blas:run({dtbmv, blasRowMajor, blasUpper, blasNoTrans, blasUnit, N, K, A, Lda, X, Incx}),
    [2, 3, 4, 1] =:= chain:btl(d, blas:to_bin(X)).