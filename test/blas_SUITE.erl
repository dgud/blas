-module(blas_SUITE).

-compile(export_all).

test() ->
    raw(),
    data_types(),
    level1(),
    level2(),
    level3(),
    ok.

data_types() ->
    I0 = [0.0,1.0,2.0,3.0,4.0,5.0],
    C0 = blas:vec_from_list(I0),
    I0 = blas:to_list(C0),
    6  = blas:vec_size(C0),
    [{0.0,1.0},{2.0,3.0},{4.0,5.0}] = blas:to_tuple_list(2, C0),
    {'EXIT', {badarg,_}} = (catch blas:to_tuple_list(4, C0)),
    0.0 = blas:value(0,C0),
    3.0 = blas:value(3,C0),
    5.0 = blas:value(5,C0),
    {2.0,3.0} = blas:values(2, 2, C0),
    {'EXIT', {badarg,_}} = (catch blas:values(5, 2, C0)),
    [{2,2.0},{4,4.0}] = blas:values([2,4], C0),
    {'EXIT', {badarg,_}} = (catch blas:values([2.0], C0)),

    {'EXIT', {badarg,_}} = (catch blas:update(-1, 0.5, C0)),
    C1 = blas:update(0, -0.5, C0),
    C2 = blas:update(3,  0.3, C1),
    C3 = blas:update(5,  0.5, C2),
    C4 = blas:update(1,  [0.1,0.2], C3),
    [-0.5,0.1,0.2,0.3,4.0,0.5] = blas:to_list(C4),
    I0 = blas:to_list(C0),
    C5 = blas:update([{0,0.0}, {4,0.4}], C4#{destr:=true}),
    [0.0,0.1,0.2,0.3,0.4,0.5] = blas:to_list(C4),
    #{type:=vector, n:=3,inc:=2,start:=1} =
	blas:set_dim(3, blas:set_inc(2, blas:set_start(1, C5))),

    M0 = blas:mat_from_vec(2,3,C0),
    M0 = blas:mat_from_list(2,3,I0),
    I0T = [{0.0,1.0,2.0},{3.0,4.0,5.0}],
    M0 = blas:mat_from_list(I0T),
    {2,3} = blas:mat_size(M0),
    #{type:=matrix, uplo:=upper, diag:=non_unit} = blas:set_triangle(upper, M0),
    #{type:=matrix, uplo:=undefined, diag:=unit} = blas:set_unit(unit, M0),
    #{op:=transp, side:=left} = blas:set_op(transp, M0),
    #{op:=no_transp, side:=right} = blas:set_side(right, M0),

    I0 = blas:to_list(M0),
    I0T = blas:to_tuple_list(M0),
    {3.0,4.0,5.0} = blas:values(3, 3, M0),
    ok.

level1() ->
    D0 = blas:vec_from_list([1.0,2.0,3.0,4.0,5.0]),
    AllOnes = blas:vec_from_list(lists:duplicate(5, 1.0)),
    AllTwos = blas:vec_from_list(lists:duplicate(5, 2.0)),

    I1 = [{-1.0,-1.0}, {0.0,0.0}, {1.0,0.0}, {2.0,0.0}, {3.0,0.0}],
    D1 = (blas:vec_from_list(I1))#{inc:=2},

    {_R0,_Z0,C0,S0} = blas:rotg(1.0, 1.0),
    D2 = blas:rot(D1,C0,S0),

    io:format("Rot ~p~n",[blas:to_tuple_list(2, D2)]),

    D3 = blas:scal(2.0, D0),
    [2.0,4.0,6.0,8.0,10.0] = blas:to_list(D3),

    D4 = blas:axpy(2.0, D0, AllOnes),
    [3.0,5.0,7.0,9.0,11.0] = blas:to_list(D4),
    D5 = blas:set_inc(2, blas:set_dim(2,(blas:copy(D0))#{destr:=true})),
    {_,_} = blas:swap(D5, blas:set_start(1, D5)),
    [2.0,1.0,4.0,3.0,5.0] = blas:to_list(D5),
    15.0 = blas:asum(D0),
    30.0 = blas:dot(D0, AllTwos),
    io:format("NRM: ~p~n",[blas:nrm2(D0)]),
    {4,5.0} = blas:iamax(blas:set_inc(1, blas:set_dim(5, D5))),
    ok.

level2() ->
    Mat0 = [1.0,2.0,3.0,
	    2.0,2.0,4.0,
	    3.0,2.0,2.0,
	    4.0,2.0,1.0,
	    9.9,9.9,9.9
	   ],
    Mat = blas:mat_from_list(4,3, Mat0),
    Zeros = blas:vec_from_list(lists:duplicate(5, 0.0)),
    X0 = blas:vec_from_list([1.0,2.0,3.0]),
    Y0 = blas:gemv(1.0,Mat,X0,0.0,Zeros),
    [14.0, 18.0, 13.0, 11.0, 0.0] = blas:to_list(Y0),
    Y1 = blas:symv(1.0,blas:set_triangle(upper, Mat),X0,0.0,Zeros),
    [14.0, 18.0, 17.0, 0.0, 0.0] = blas:to_list(Y1),

    SymPacked0 = [1.0,2.0,3.0,
	  	      2.0,4.0,
		          2.0],
    SymP0 = blas:mat_from_vec(3, 3, blas:vec_from_list(SymPacked0)),
    SymP = blas:set_triangle(upper, SymP0),
    Y2 = blas:spmv(1.0,SymP,X0,0.0,Zeros),
    [14.0, 18.0, 17.0, 0.0, 0.0] = blas:to_list(Y2),

    %% Note the diagonal are expected to be ones
    X1 = blas:trmv(blas:set_triangle(lower, blas:set_unit(unit, Mat)), X0),
    [1.0, 4.0, 10.0] = blas:to_list(X1),

    X2 = blas:tpmv(SymP, X0),
    [14.0, 16.0, 6.0] = blas:to_list(X2),

    %%  Solve
    X3 = blas:trsv(blas:set_triangle(lower, blas:set_unit(unit, Mat)), X0),
    [1.0, 0.0, 0.0] = blas:to_list(X3),
    X4 = blas:tpsv(SymP, X0),
    [0.5, -2.0, 1.5] = blas:to_list(X4),

    %% Rank
    M1 = blas:ger(1.0,
		   blas:vec_from_list(lists:duplicate(5, 1.0)++[9.0]),
		   blas:vec_from_list(lists:duplicate(3, 2.0)++[9.0]),
		   Mat),
    [3.0,4.0,5.0,4.0,4.0,6.0,5.0,4.0,4.0,6.0,4.0,3.0,9.9,9.9,9.9] =
	blas:to_list(M1),

    M2 = blas:syr(2.0,
		   blas:vec_from_list(lists:duplicate(5, 1.0)++[9.0]),
		   blas:set_triangle(upper,Mat)),
    [3.0,4.0,5.0,  _,4.0,6.0,  _,  _,4.0,4.0,2.0,1.0,9.9,9.9,9.9] =
	blas:to_list(M2),

    M3 = blas:spr(2.0, blas:vec_from_list(lists:duplicate(5, 1.0)++[9.0]),
		  SymP),
    [3.0,4.0,5.0,4.0,6.0,4.0] = blas:to_list(M3),

    M4 = blas:syr2(1.0,
		    blas:vec_from_list(lists:duplicate(5, 1.0)++[9.0]),
		    blas:vec_from_list(lists:duplicate(3, 2.0)++[9.0]),
		    blas:set_triangle(upper,Mat)),
    [5.0,6.0,7.0,  _,6.0,8.0,  _,  _,6.0,4.0,2.0,1.0,9.9,9.9,9.9] =
	blas:to_list(M4),
    M5 = blas:spr2(1.0,
		    blas:vec_from_list(lists:duplicate(5, 1.0)++[9.0]),
		    blas:vec_from_list(lists:duplicate(3, 2.0)++[9.0]),
		    SymP),
    [5.0,6.0,7.0,6.0,8.0,6.0] = blas:to_list(M5),

    ok.

level3() ->
    A0 = [1.0, -3.0,
	 2.0,  4.0,
	 1.0, -1.0,
	 9.9,  9.9,
	 9.9,  9.9],
    B0 = [ 1.0, 2.0, 1.0,
	 -3.0, 4.0,-1.0,
	  9.9, 9.9, 9.9],
    C0 = lists:duplicate(9, 0.5) ++ [9.9,9.9,9.9],
    A = blas:mat_from_list(3,2,A0),
    B = blas:mat_from_list(2,3,B0),
    C = blas:mat_from_list(3,3,C0),
    R0 = blas:gemm(1.0, A, B, 2.0, C),
    [{11.0,-9.0, 5.0},
     {-9.0,21.0,-1.0},
     {5.0, -1.0, 3.0},
     {9.9, 9.9, 9.9}] = blas:to_tuple_list(3, R0),

    Sym0 = [1.0,-3.0,
	   -3.0, 4.0,
	    9.9, 9.9],
    Sym = blas:set_triangle(upper, blas:mat_from_list(2,2,Sym0)),
    R1 = blas:symm(1.0, Sym, B, 0.0, blas:set_dim({2,3},C)),
    [{10.0,-10.0,4.0}, {-15.0,10.0,-7.0}, {0.5,0.5,0.5}|_]
	= blas:to_tuple_list(R1),

    R2 = blas:trmm(1.0, Sym, B),
    [{10.0,-10.0,4.0}, {-12.0,16.0,-4.0}, {9.9,9.9,9.9}]
	= blas:to_tuple_list(R2),

    B10 = [ 1.0, 1.0, 1.0,
	   -1.0,-1.0,-1.0,
	    9.9, 9.9, 9.9],

    B1 = blas:mat_from_list(2,3,B10),
    UnitSym = blas:set_unit(unit,Sym),
    R3 = blas:trsm(1.0, UnitSym, B1),
    [{-2.0,-2.0,-2.0},
     {-1.0,-1.0,-1.0},
     {9.9,9.9,9.9}]  = blas:to_tuple_list(R3),

    R4 = blas:syrk(1.0, A, 0.0, blas:set_triangle(upper, C)),
    [{10.0, -10.0, 4.0},  %% C is symmetric (lower half is skipped)
     {_, 20.0,    -2.0},
     {_,    _,     2.0},
     {9.9,9.9,9.9}] = blas:to_tuple_list(R4),

    R5 = blas:syrk(1.0, blas:set_op(transp,A), 0.0,
		   blas:set_inc(2, blas:set_dim({2,2},blas:set_triangle(upper, C)))),
    [{6.0, 4.0},  %% C is symmetric (lower half is skipped)
     {_,  26.0},
     {0.5, 0.5}|_] = blas:to_tuple_list(R5),

    Id = blas:mat_from_list([{1.0, 0.0},
			      {0.0, 1.0}]),
    R6 = blas:syr2k(1.0, blas:set_dim({2,2},A), Id, 0.0,
		    blas:set_inc(2, blas:set_dim({2,2},blas:set_triangle(upper, C)))),
    [2.0, -1.0,  %% Symmetric (lower half is skipped)
     _,    8.0,
     0.5|_] = blas:to_list(R6),

    ok.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

raw() ->
    raw_data_types(),
    raw_level1(),
    raw_level2(),
    raw_level3().


vec(A) -> blasd_raw:from_list(A).

raw_data_types() ->
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

raw_level1() ->
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

raw_level2() ->
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

raw_level3() ->
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

    ok = blasd_raw:syrk(row_maj, upper, transp, 2,3, 1.0,
			vec(A), 2, 0.0, R5 = vec(C), 2),
    [{6.0, 4.0},  %% C is symmetric (lower half is skipped)
     {_,  26.0},
     {0.5, 0.5}|_] = blasd_raw:to_tuple_list(2,R5),


    Id = [1.0, 0.0,
	  0.0, 1.0],
    ok = blasd_raw:syr2k(2,2, 1.0, vec(A), vec(Id), 0.0, R6 = vec(C)),
    [2.0, -1.0,  %% Symmetric (lower half is skipped)
     _,    8.0,
     0.5|_] = blasd_raw:to_list(R6),

    ok.
