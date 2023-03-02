-module(stbsv_test).
-include_lib("eunit/include/eunit.hrl").

stbsv_test()->
    N = 4,
    K = 1,
    A = blas:new(chain:ltb(s, [0,1, 0,2, 0,3, 0,0])),
    X = blas:new(chain:ltb(s, [2, 3, 4, 1])),
    ok = blas:run({stbsv, blasRowMajor, blasUpper, blasNoTrans, blasUnit, N, K, A, 2, X, 1}),
    [1.0,1.0,1.0,1.0] = chain:btl(s, blas:to_bin(X)).