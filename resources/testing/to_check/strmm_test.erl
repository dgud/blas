-module(strmm_test).
-include_lib("eunit/include/eunit.hrl").

strmm_test()->
    M = 3,
    N = 4,
    A = blas:new(chain:ltb(s, [1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0])),
    B = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])),
    C = blas:new(chain:ltb(s, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
    blas:run({strmm, blasColMajor, blasLeft, blasUpper, blasNoTrans, blasNonUnit, M, N, 1.0, A, M, B, M, 0.0, C, M}),
    [1.0, 2.0, 3.0, 4.0, 10.0, 16.0, 22.0, 28.0, 37.0, 52.0, 67.0, 82.0] = chain:btl(s, C).