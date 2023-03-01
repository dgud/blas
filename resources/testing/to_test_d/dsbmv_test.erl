-module(dsbmv_test).
-include_lib("eunit/include/eunit.hrl").

dsbmv_test()->
    N = 5,
    K = 2,
    A = blas:new(chain:ltb(d, [1.0, 2.0, 3.0,  4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])),
    X = blas:new(chain:ltb(d, [1.0, 2.0, 3.0, 4.0, 5.0])),
    Y = blas:new(chain:ltb(d, [0.0, 0.0, 0.0, 0.0, 0.0])),
    blas:run({dsbmv, blasRowMajor, blasUpper, N, K, 1.0, A, 3, X, 1, 0.0, Y, 1}),
    [ 14.0,49.0,111.0,131.0,136.0] = chain:btl(d, blas:to_bin(Y)).