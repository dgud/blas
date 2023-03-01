-module(dsyr2k_test).
-include_lib("eunit/include/eunit.hrl").

dsyr2k_test()->
    N = 3,
    A = blas:new(chain:ltb(d, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])),
    B = blas:new(chain:ltb(d, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])),
    C = blas:new(chain:ltb(d, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])),
    
    blas:run({dsyr2k, blasRowMajor, blasUpper, blasNoTrans, N, N, 1.0, A, N, B, N, 0.5, C, N}),
    [2.0, 2.0, 3.0, 2.0, 6.0, 8.0, 3.0, 8.0, 18.0] =:= chain:btl(d, blas:to_bin(C)).