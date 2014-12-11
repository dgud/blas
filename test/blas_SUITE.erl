-module(blas_SUITE).

-compile(export_all).

test() ->
    I0 = [1.0,2.0,3.0,4.0,5.0],
    D0 = blasd:from_list(I0),
    5  = blasd:size(D0),
    I0 = blasd:to_list(D0),

    I1 = [{-1.0,-1.0}, {0.0,0.0}, {1.0,0.0}, {2.0,0.0}, {3.0,0.0}],
    D1 = blasd:from_list(I1),
    10  = blasd:size(D1),
    I1 = blasd:to_tuple_list(2, D1),

    {R0,_Z0,C0,S0} = blasd:rotg(1.0, 1.0),
    D2 = blasd:copy(D1),
    ok = blasd:rot(D2,C0,S0),

    ok = blasd:rot(3, D2,2,2, D2,3,2, C0, S0),
    io:format("~p~n",[blasd:to_tuple_list(2, D2)]),

    ok = blasd:scal(2.0, D3 = blasd:copy(D2)),
    io:format("~p~n",[blasd:to_tuple_list(2, D3)]),

    blasd:axpy(2.0, D0, R=blasd:from_list([1.0,1.0,1.0,1.0,1.0])),
    io:format("daxpy ~p => ~p~n", [blasd:to_list(D0),blasd:to_list(R)]),
    ok.
