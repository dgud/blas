-module(sgbmv_test).
-include_lib("eunit/include/eunit.hrl").

sgbmv_test()->
    M = 5,
    N = 4,
    Kl = 2,
    Ku = 1,
    A = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])),
    X = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0])),
    Y = blas:new(chain:ltb(s, [5.0, 6.0, 7.0, 8.0, 9.0])),
    blas:run({sgbmv, blasRowMajor, blasNoTrans, M, N, Kl, Ku, 2.0, A, kl+ku+1, X, 1, 0.5, Y, 1}),
    [37.0, 54.0, 71.0, 88.0, 105.0] = chain:btl(s, Y).