-module(dtrmm_test).
-include_lib("eunit/include/eunit.hrl").

dtrmm_test()->
    M = 3,
    N = 3,
    A = blas:new(chain:ltb(d, [1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0])),
    B = blas:new(chain:ltb(d, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])),
    blas:run({dtrmm, blasColMajor, blasLeft, blasUpper, blasNoTrans, blasNonUnit, M, N, 1.0, A, M, B, M}),
    [17.0,21.0,18.0,38.0,45.0,36.0,59.0,69.0,54.0] = chain:btl(d, blas:to_bin(B)).