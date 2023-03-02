% Performs a matrix-vector multiplication between a general band matrix and a complex vector.

-module(cgbmv_test).
-include_lib("eunit/include/eunit.hrl").

cgbmv_test()->
    M = 2,
    N = 2,
    Kl = 1,
    Ku = 1,
    A = blas:new(chain:ltb(s, [0,0, 2,3, 1,-2,  1,1, 4,0, 0,0])),
    X = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0])),
    Y = blas:new(chain:ltb(s, [5.0, 6.0, 7.0, 8.0])),
    blas:run({cgbmv, blasRowMajor, blasNoTrans, M, N, Kl, Ku, chain:ltb(s,[1.0, 0.0]), A, Kl+Ku+1, X, 1, chain:ltb(s,[1.0, 0.0]), Y, 1}),
    %io:format("~w\n", chain:btl(s, blas:to_bin(Y))),
    [12.0, 11.0, 18.0, 27.0] = chain:btl(s, blas:to_bin(Y)).