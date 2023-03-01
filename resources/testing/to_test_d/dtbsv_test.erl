-module(dtbsv_test).
-include_lib("eunit/include/eunit.hrl").

dtbsv_test()->
    N = 4,
    K = 1,
    A = blas:new(chain:ltb(d, [0,1, 0,2, 0,3, 0,0])),
    X = blas:new(chain:ltb(d, [2, 3, 4, 1])),
    ok = blas:run({dtbsv, blasRowMajor, blasUpper, blasNoTrans, blasUnit, N, K, A, 2, X, 1}),
    [1,1,1,1] =:= chain:btl(d, blas:to_bin(X)).