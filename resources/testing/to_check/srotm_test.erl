-module(srotm_test).
-include_lib("eunit/include/eunit.hrl").

srotm_test()->
    N = 4,
    X = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0])),
    Y = blas:new(chain:ltb(s, [5.0, 6.0, 7.0, 8.0])),
    Param = blas:new(chain:ltb(s, [1.0, 2.0, 3.0, 4.0, 5.0])),
    blas:run({srotm, N, X, 1, Y, 1, Param}),
    X_out = blas:new(chain:ltb(s, [-2.0, -3.0, -4.0, -5.0])),
    [1.0, 2.0, 3.0, 4.0] = chain:btl(s, Y_out).