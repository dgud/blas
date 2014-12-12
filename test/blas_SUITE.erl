-module(blas_SUITE).

-compile(export_all).

test() ->
    level1(),
    level2(),
    ok.

level1() ->
    I0 = [1.0,2.0,3.0,4.0,5.0],
    AllOnes = blasd:from_list(lists:duplicate(5, 1.0)),
    AllTwos = blasd:from_list(lists:duplicate(5, 2.0)),
    D0 = blasd:from_list(I0),
    5  = blasd:vec_size(D0),
    I0 = blasd:to_list(D0),

    I1 = [{-1.0,-1.0}, {0.0,0.0}, {1.0,0.0}, {2.0,0.0}, {3.0,0.0}],
    D1 = blasd:from_list(I1),
    10  = blasd:vec_size(D1),
    I1 = blasd:to_tuple_list(2, D1),

    {_R0,_Z0,C0,S0} = blasd:rotg(1.0, 1.0),
    D2 = blasd:copy(D1),
    ok = blasd:rot(D2,C0,S0),

    ok = blasd:rot(3, D2,2,2, D2,3,2, C0, S0),
    io:format("Rot ~p~n",[blasd:to_tuple_list(2, D2)]),

    ok = blasd:scal(2.0, D3 = blasd:copy(D0)),
    [2.0,4.0,6.0,8.0,10.0] = blasd:to_list(D3),

    blasd:axpy(2.0, D0, D4 = blasd:copy(AllOnes)),
    [3.0,5.0,7.0,9.0,11.0] = blasd:to_list(D4),
    D5=blasd:copy(D0),
    ok = blasd:swap(2, D5, 0, 2, D5, 1, 2),
    [2.0,1.0,4.0,3.0,5.0] = blasd:to_list(D5),
    15.0 = blasd:asum(D0),
    30.0 = blasd:dot(D0, AllTwos),
    io:format("NRM: ~p~n",[blasd:nrm2(D0)]),
    {4,5.0} = blasd:iamax(D5),
    ok.

level2() ->
    Mat0 = [1.0,2.0,3.0,
	    2.0,2.0,4.0,
	    3.0,2.0,2.0,
	    4.0,2.0,1.0,
	    9.9,9.9,9.9
	   ],
    Mat = blasd:from_list(Mat0),
    Zeros = blasd:from_list(lists:duplicate(5, 0.0)),
    X0 = blasd:from_list([1.0,2.0,3.0]),
    ok = blasd:gemv(4,3,1.0,Mat,X0,0.0,Y0=blasd:copy(Zeros)),
    [14.0, 18.0, 13.0, 11.0, 0.0] = blasd:to_list(Y0),
    ok = blasd:symv(upper,3,1.0,Mat,X0,0.0,Y1=blasd:copy(Zeros)),
    [14.0, 18.0, 17.0, 0.0, 0.0] = blasd:to_list(Y1),

    SymPacked0 = [1.0,2.0,3.0,
	  	      2.0,4.0,
		          2.0],
    SymP = blasd:from_list(SymPacked0),
    ok = blasd:spmv(upper,3,1.0,SymP,X0,0.0,Y1=blasd:copy(Zeros)),
    [14.0, 18.0, 17.0, 0.0, 0.0] = blasd:to_list(Y1),

    %% Note the diagonal are expected to be ones
    ok = blasd:trmv(lower, unit, 3, Mat, X1=blasd:copy(X0)),
    [1.0, 4.0, 10.0] = blasd:to_list(X1),

    ok = blasd:tpmv(upper, non_unit, 3, SymP, X2=blasd:copy(X0)),
    [14.0, 16.0, 6.0] = blasd:to_list(X2),

    %%  Solve
    ok = blasd:trsv(lower, unit, 3, Mat, X3=blasd:copy(X0)),
    [1.0, 0.0, 0.0] = blasd:to_list(X3),
    ok = blasd:tpsv(upper, non_unit, 3, SymP, X4=blasd:copy(X0)),
    [0.5, -2.0, 1.5] = blasd:to_list(X4),

    %% Rank
    ok = blasd:ger(4, 3, 1.0,
		   blasd:from_list(lists:duplicate(5, 1.0)++[9.0]),
		   blasd:from_list(lists:duplicate(3, 2.0)++[9.0]),
		   M1 = blasd:copy(Mat)),
    [3.0,4.0,5.0,4.0,4.0,6.0,5.0,4.0,4.0,6.0,4.0,3.0,9.9,9.9,9.9] =
	blasd:to_list(M1),

    ok = blasd:syr(upper, 3, 2.0,
		   blasd:from_list(lists:duplicate(5, 1.0)++[9.0]),
		   M2 = blasd:copy(Mat)),
    [3.0,4.0,5.0,  _,4.0,6.0,  _,  _,4.0,4.0,2.0,1.0,9.9,9.9,9.9] =
	blasd:to_list(M2),

    ok = blasd:spr(upper, 3, 2.0,
		   blasd:from_list(lists:duplicate(5, 1.0)++[9.0]),
		   M3 = blasd:copy(SymP)),
    [3.0,4.0,5.0,4.0,6.0,4.0] = blasd:to_list(M3),

    ok = blasd:syr2(upper, 3, 1.0,
		    blasd:from_list(lists:duplicate(5, 1.0)++[9.0]),
		    blasd:from_list(lists:duplicate(3, 2.0)++[9.0]),
		    M4 = blasd:copy(Mat)),
    [5.0,6.0,7.0,  _,6.0,8.0,  _,  _,6.0,4.0,2.0,1.0,9.9,9.9,9.9] =
	blasd:to_list(M4),
    ok = blasd:spr2(upper, 3, 1.0,
		    blasd:from_list(lists:duplicate(5, 1.0)++[9.0]),
		    blasd:from_list(lists:duplicate(3, 2.0)++[9.0]),
		    M5 = blasd:copy(SymP)),
    [5.0,6.0,7.0,6.0,8.0,6.0] = blasd:to_list(M5),

    io:format("~p~n",[blasd:to_list(M3)]),
    ok.

