-module(srotmg_test).
-include_lib("eunit/include/eunit.hrl").

srotmg_test()->
    Param = blas:new(chain:ltb(s, [0.0])),
    blas:run({srotmg, 1.0, 2.0, 3.0, 4.0, Param}),
    [0.5f, 0.0, 0.75f, 0.0, 0.0] = chain:btl(s, Param_out).