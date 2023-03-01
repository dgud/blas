-module(dsymv_test).
-include_lib("eunit/include/eunit.hrl").

dsymv_test()->
    N = 3,
    A = blas:new(chain:ltb(d, [1, 2, 3, 2, 4, 5, 3, 5, 6])),
    X = blas:new(chain:ltb(d, [1, 2, 3])),
    Y = blas:new(chain:ltb(d, [0, 0, 0])),
    blas:run({dsymv, blasRowMajor, blasUpper, N, 1.0, A, N, X, 1, 0.0, Y, 1}),
    [14, 25, 31] =:= chain:btl(d, blas:to_bin(Y)).