-module(stbsv_test).
-include_lib("eunit/include/eunit.hrl").

stbsv_test()->
    Order = 3,
    N = 3,
    K = 1,
    A = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0])),
    B = blas:new(chain:ltb(s, [7.0, 8.0, 9.0])),
    blas:run({stbsv, Order, blasUpper, blasNoTrans, blasNonUnit, N, K, A, N, B, 1}),
    [1.0, -2.0, 3.0] = chain:btl(s, B).