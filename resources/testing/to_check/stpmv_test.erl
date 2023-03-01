-module(stpmv_test).
-include_lib("eunit/include/eunit.hrl").

stpmv_test()->
    N = 3,
    Ap = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
    X = blas:new(chain:ltb(s, [1.0, 2.0, 3.0])),
    blas:run({stpmv, blasRowMajor, blasUpper, blasNoTrans, blasNonUnit, N, Ap, X, 1}),
    [1.0, 7.0, 22.0] = chain:btl(s, X).