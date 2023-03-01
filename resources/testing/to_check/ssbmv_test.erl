-module(ssbmv_test).
-include_lib("eunit/include/eunit.hrl").

ssbmv_test()->
    N = 5,
    K = 2,
    A = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])),
    X = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0, 5.0])),
    Y = blas:new(chain:ltb(s, [0.0, 0.0, 0.0, 0.0, 0.0])),
    blas:run({ssbmv, blasRowMajor, blasUpper, N, K, 1.0, A, X, 0.0, Y}),
    [22.0, 28.0, 34.0, 40.0, 46.0] = chain:btl(s, Y).