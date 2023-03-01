-module(stbmv_test).
-include_lib("eunit/include/eunit.hrl").

stbmv_test()->
    N = 5,
    K = 2,
    A = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])),
    Lda = 3,
    X = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0, 5.0])),
    Incx = 1,
    Y = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0, 5.0])),
    Incy = 1,
    blas:run({stbmv, blasRowMajor, blasUpper, blasNoTrans, blasNonUnit, N, K, A, Lda, X, Incx, Y, Incy}),
    [7.0, 18.0, 29.0, 40.0, 51.0] = chain:btl(s, Y).