-module(ssyr2_test).
-include_lib("eunit/include/eunit.hrl").

ssyr2_test()->
    N = 3,
    X = blas:new(chain:ltb(s, [1.0, 2.0, 3.0])),
    Y = blas:new(chain:ltb(s, [4.0, 5.0, 6.0])),
    A = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0])),
    blas:run({ssyr2, blasRowMajor, blasUpper, N, 2.0, X, 1, Y, 1, A, N}),
    [129, 130, 131, 0, 132, 133, 0, 0, 134] =:= chain:btl(s, blas:to_bin(A)).

