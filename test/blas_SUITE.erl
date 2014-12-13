-module(blas_SUITE).

-compile(export_all).

test() ->
    data_types(),
    level1(),
    level2(),
    level3(),
    ok.

vec(A) -> blasd_raw:from_list(A).

data_types() ->
    I0 = [0.0,1.0,2.0,3.0,4.0,5.0],
    C0 = blasd_raw:from_list(I0),
    I0 = blasd_raw:to_list(C0),
    6  = blasd_raw:cont_size(C0),
    [{0.0,1.0},{2.0,3.0},{4.0,5.0}] = blasd_raw:to_tuple_list(2, C0),
    {'EXIT', {badarg,_}} = (catch blasd_raw:to_tuple_list(4, C0)),
    0.0 = blasd_raw:value(0,C0),
    3.0 = blasd_raw:value(3,C0),
    5.0 = blasd_raw:value(5,C0),
    {'EXIT', {badarg,_}} = (catch blasd_raw:value(-1,C0)),
    {'EXIT', {badarg,_}} = (catch blasd_raw:value(6,C0)),
    {2.0,3.0} = blasd_raw:values(2, 2, C0),
    {'EXIT', {badarg,_}} = (catch blasd_raw:values(5, 2, C0)),

    [{2,2.0},{4,4.0}] = blasd_raw:values([2,4], C0),
    {'EXIT', {badarg,_}} = (catch blasd_raw:values([2.0], C0)),
    {'EXIT', {badarg,_}} = (catch blasd_raw:values([6], C0)),

    {'EXIT', {badarg,_}} = (catch blasd_raw:update(-1, 0.5, C0)),
    {'EXIT', {badarg,_}} = (catch blasd_raw:update(3, foo, C0)),
    {'EXIT', {badarg,_}} = (catch blasd_raw:update(6, 0.5, C0)),
    {'EXIT', {badarg,_}} = (catch blasd_raw:update(foo, 0.5, C0)),
    {'EXIT', {badarg,_}} = (catch blasd_raw:update(foo, 0.5, foo)),
    ok = blasd_raw:update(0, -0.5, C0),
    ok = blasd_raw:update(3,  0.3, C0),
    ok = blasd_raw:update(5,  0.5, C0),
    ok = blasd_raw:update(1,  [0.1,0.2], C0),
    [-0.5,0.1,0.2,0.3,4.0,0.5] = blasd_raw:to_list(C0),
    ok = blasd_raw:update([{0,0.0}, {4,0.4}], C0),
    [0.0,0.1,0.2,0.3,0.4,0.5] = blasd_raw:to_list(C0),
    {'EXIT', {badarg,_}} = (catch blasd_raw:update([{0,0.0}, {4,0.4,234}], C0)),
    ok.

level1() ->
    I0 = [1.0,2.0,3.0,4.0,5.0],
    AllOnes = vec(lists:duplicate(5, 1.0)),
    AllTwos = vec(lists:duplicate(5, 2.0)),
    D0 = vec(I0),
    5  = blasd_raw:cont_size(D0),
    I0 = blasd_raw:to_list(D0),

    I1 = [{-1.0,-1.0}, {0.0,0.0}, {1.0,0.0}, {2.0,0.0}, {3.0,0.0}],
    D1 = vec(I1),
    10  = blasd_raw:cont_size(D1),
    I1 = blasd_raw:to_tuple_list(2, D1),

    {_R0,_Z0,C0,S0} = blasd_raw:rotg(1.0, 1.0),
    D2 = blasd_raw:copy(D1),
    ok = blasd_raw:rot(D2,C0,S0),

    ok = blasd_raw:rot(3, D2,2,2, D2,3,2, C0, S0),
    io:format("Rot ~p~n",[blasd_raw:to_tuple_list(2, D2)]),

    ok = blasd_raw:scal(2.0, D3 = blasd_raw:copy(D0)),
    [2.0,4.0,6.0,8.0,10.0] = blasd_raw:to_list(D3),

    blasd_raw:axpy(2.0, D0, D4 = blasd_raw:copy(AllOnes)),
    [3.0,5.0,7.0,9.0,11.0] = blasd_raw:to_list(D4),
    D5=blasd_raw:copy(D0),
    ok = blasd_raw:swap(2, D5, 0, 2, D5, 1, 2),
    [2.0,1.0,4.0,3.0,5.0] = blasd_raw:to_list(D5),
    15.0 = blasd_raw:asum(D0),
    30.0 = blasd_raw:dot(D0, AllTwos),
    io:format("NRM: ~p~n",[blasd_raw:nrm2(D0)]),
    {4,5.0} = blasd_raw:iamax(D5),
    ok.

level2() ->
    Mat0 = [1.0,2.0,3.0,
	    2.0,2.0,4.0,
	    3.0,2.0,2.0,
	    4.0,2.0,1.0,
	    9.9,9.9,9.9
	   ],
    Mat = vec(Mat0),
    Zeros = vec(lists:duplicate(5, 0.0)),
    X0 = vec([1.0,2.0,3.0]),
    ok = blasd_raw:gemv(4,3,1.0,Mat,X0,0.0,Y0=blasd_raw:copy(Zeros)),
    [14.0, 18.0, 13.0, 11.0, 0.0] = blasd_raw:to_list(Y0),
    ok = blasd_raw:symv(upper,3,1.0,Mat,X0,0.0,Y1=blasd_raw:copy(Zeros)),
    [14.0, 18.0, 17.0, 0.0, 0.0] = blasd_raw:to_list(Y1),

    SymPacked0 = [1.0,2.0,3.0,
	  	      2.0,4.0,
		          2.0],
    SymP = vec(SymPacked0),
    ok = blasd_raw:spmv(upper,3,1.0,SymP,X0,0.0,Y1=blasd_raw:copy(Zeros)),
    [14.0, 18.0, 17.0, 0.0, 0.0] = blasd_raw:to_list(Y1),

    %% Note the diagonal are expected to be ones
    ok = blasd_raw:trmv(lower, unit, 3, Mat, X1=blasd_raw:copy(X0)),
    [1.0, 4.0, 10.0] = blasd_raw:to_list(X1),

    ok = blasd_raw:tpmv(upper, non_unit, 3, SymP, X2=blasd_raw:copy(X0)),
    [14.0, 16.0, 6.0] = blasd_raw:to_list(X2),

    %%  Solve
    ok = blasd_raw:trsv(lower, unit, 3, Mat, X3=blasd_raw:copy(X0)),
    [1.0, 0.0, 0.0] = blasd_raw:to_list(X3),
    ok = blasd_raw:tpsv(upper, non_unit, 3, SymP, X4=blasd_raw:copy(X0)),
    [0.5, -2.0, 1.5] = blasd_raw:to_list(X4),

    %% Rank
    ok = blasd_raw:ger(4, 3, 1.0,
		   vec(lists:duplicate(5, 1.0)++[9.0]),
		   vec(lists:duplicate(3, 2.0)++[9.0]),
		   M1 = blasd_raw:copy(Mat)),
    [3.0,4.0,5.0,4.0,4.0,6.0,5.0,4.0,4.0,6.0,4.0,3.0,9.9,9.9,9.9] =
	blasd_raw:to_list(M1),

    ok = blasd_raw:syr(upper, 3, 2.0,
		   vec(lists:duplicate(5, 1.0)++[9.0]),
		   M2 = blasd_raw:copy(Mat)),
    [3.0,4.0,5.0,  _,4.0,6.0,  _,  _,4.0,4.0,2.0,1.0,9.9,9.9,9.9] =
	blasd_raw:to_list(M2),

    ok = blasd_raw:spr(upper, 3, 2.0,
		   vec(lists:duplicate(5, 1.0)++[9.0]),
		   M3 = blasd_raw:copy(SymP)),
    [3.0,4.0,5.0,4.0,6.0,4.0] = blasd_raw:to_list(M3),

    ok = blasd_raw:syr2(upper, 3, 1.0,
		    vec(lists:duplicate(5, 1.0)++[9.0]),
		    vec(lists:duplicate(3, 2.0)++[9.0]),
		    M4 = blasd_raw:copy(Mat)),
    [5.0,6.0,7.0,  _,6.0,8.0,  _,  _,6.0,4.0,2.0,1.0,9.9,9.9,9.9] =
	blasd_raw:to_list(M4),
    ok = blasd_raw:spr2(upper, 3, 1.0,
		    vec(lists:duplicate(5, 1.0)++[9.0]),
		    vec(lists:duplicate(3, 2.0)++[9.0]),
		    M5 = blasd_raw:copy(SymP)),
    [5.0,6.0,7.0,6.0,8.0,6.0] = blasd_raw:to_list(M5),

    ok.

level3() ->
    A = [1.0, -3.0,
	 2.0,  4.0,
	 1.0, -1.0,
	 9.9,  9.9,
	 9.9,  9.9],
    B = [ 1.0, 2.0, 1.0,
	 -3.0, 4.0,-1.0,
	  9.9, 9.9, 9.9],
    C = lists:duplicate(9, 0.5) ++ [9.9,9.9,9.9],
    ok = blasd_raw:gemm(3, 3, 2, 1.0, vec(A), vec(B), 2.0, R0 = vec(C)),
    [{11.0,-9.0, 5.0},
     {-9.0,21.0,-1.0},
     {5.0, -1.0, 3.0},
     {9.9, 9.9, 9.9}] = blasd_raw:to_tuple_list(3, R0),

    Sym = [1.0,-3.0,
	  -3.0, 4.0,
	   9.9, 9.9],
    ok = blasd_raw:symm(2, 3, 1.0, vec(Sym), vec(B), 0.0, R1 = vec(C)),
    [{10.0,-10.0,4.0}, {-15.0,10.0,-7.0}, {0.5,0.5,0.5}|_]
	= blasd_raw:to_tuple_list(3,R1),

    ok = blasd_raw:trmm(non_unit, 2, 3, 1.0, vec(Sym), R2 = vec(B)),
    [{10.0,-10.0,4.0}, {-12.0,16.0,-4.0}, {9.9,9.9,9.9}]
	= blasd_raw:to_tuple_list(3,R2),

    B1 = [ 1.0, 1.0, 1.0,
	  -1.0,-1.0,-1.0,
	   9.9, 9.9, 9.9],

    ok = blasd_raw:trsm(upper, unit, 2, 3, 1.0, vec(Sym), R3=vec(B1)),
    [{-2.0,-2.0,-2.0},
     {-1.0,-1.0,-1.0},
     {9.9,9.9,9.9}]  = blasd_raw:to_tuple_list(3,R3),

    ok = blasd_raw:syrk(3,2, 1.0, vec(A), 0.0, R4 = vec(C)),
    [{10.0, -10.0, 4.0},  %% C is symmetric (lower half is skipped)
     {_, 20.0,    -2.0},
     {_,    _,     2.0},
     {9.9,9.9,9.9}] = blasd_raw:to_tuple_list(3,R4),

    Id = [1.0, 0.0,
	  0.0, 1.0],
    ok = blasd_raw:syr2k(2,2, 1.0, vec(A), vec(Id), 0.0, R5 = vec(C)),
    [2.0, -1.0,  %% Symmetric (lower half is skipped)
     _,    8.0,
     0.5|_] = blasd_raw:to_list(R5),

    ok.
