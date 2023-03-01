-module(strsm_test).
-include_lib("eunit/include/eunit.hrl").

strsm_test()->
    blas:run({strsm,blasColMajor, Side, Uplo, blasNoTrans, Diag, m, n, alpha, A, m, B, n}).