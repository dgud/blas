%%%-------------------------------------------------------------------
%%% @author Dan Gudmundsson <dgud@erlang.org>
%%% @copyright (C) 2014, Dan Gudmundsson
%%% @doc  A functional API to the standard BLAS Basic Linnear Algebra
%%%
%%%   Vectors and Matrices are created, read and updated by the provided api in
%%%   this module. Indecies are zero based. And should work well
%%%   together with the 'array' api.
%%%
%%%   The data containers have properties which steer how the data
%%%   will be used in the functions below. As an example interleaved
%%%   vectors can be used.  And some the matrix properties must be set
%%%   for some of functions.
%%%
%%%   Data can be used and shared between processes as long as 'destr'
%%%   is not set.
%%%
%%% @end Created : 14 Dec 2014 by Dan Gudmundsson
%%%   <dgud@erlang.org>
%%%   -------------------------------------------------------------------

-module(blas).

%% Data handling
-export([vec/1, vec_from_list/1, vec_from_mat/1,
	 mat/2, mat/3, mat_from_list/1, mat_from_list/3, mat_from_vec/3,
	 to_list/1, to_tuple_list/1, to_tuple_list/2,
	 vec_size/1,mat_size/1,
	 value/2, values/2, values/3,
	 update/2, update/3,
	 set_start/2, set_inc/2, set_dim/2,
	 set_triangle/2, set_unit/2, set_op/2, set_side/2
	]).

%% Level 1
-export([ %% Level 1
	  rotg/2,
	  rot/3,rot/4,
	  scal/2,
	  copy/1,copy/2,
	  axpy/3,
	  swap/2,
	  dot/2,
	  nrm2/1,
	  asum/1,
	  iamax/1,
	  %% Level 2
	  gemv/5,
	  symv/5,spmv/5,
	  trmv/2, tpmv/2,
	  trsv/2, tpsv/2,
	  ger/4,
	  syr/3, spr/3,
	  syr2/4, spr2/4,
	  %% Level 3
	  gemm/5,
	  symm/5,
	  trmm/3,
	  trsm/3,
	  syrk/4,
	  syr2k/5
	]).

-type vec() :: #{type  => vector,
		 start => integer(), %% Start idx, default: 0
		 inc   => integer(), %% number of elements until next idx, default: 1
		 n     => integer() | all, %%  number of elements to be used, default: all
		 destr => boolean(), %% Allow destructive updates of data, default: false
		 v => blasd_raw:cont()}.

-type mat() :: #{type => matrix,
		 m => integer(),   %% Number of rows
		 n => integer(),   %% Number of columns
		 inc => integer(), %% number of elements until next row (incl this row) (if row_maj)
		 order => row_maj,
		 uplo => matrix_uplo()|undefined, %% Set when matrix is triangular or symmetrical
		 diag => matrix_diag(), %% Set to unit when matrix is unit triangular, default non_unit
		 op   => matrix_op(),   %% default: no_transp
		 side => matrix_side(), %% default: left
		 destr => boolean(),    %% Allow destructive updates of data, default: false
		 v=> blasd_raw:cont()}.

-type matrix_op() :: no_transp|transp|conj_transp.
%%-type matrix_order() :: row_maj|col_maj.
-type matrix_uplo()  :: upper|lower.
-type matrix_side()  :: left|right.
-type matrix_diag() :: unit|non_unit.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-define(IMPL, blasd_raw).

%% API Data conversion
%%
%% @doc Create a vector of N zeros
%%      or create a vector of the binary containing 64b native floats.
-spec vec(N::non_neg_integer()|binary()) -> vec().
vec(N) when is_integer(N), N > 0 ->
    def_vec(?IMPL:make_cont(N, true));
vec(Bin) when is_binary(Bin) ->
    N = byte_size(Bin) div 8,
    def_vec(?IMPL:make_cont(N, Bin)).

%% @doc Create a vector from a list of values
-spec vec_from_list(List::[tuple()]|list(float())) -> vec().
vec_from_list(List) ->
    def_vec(?IMPL:from_list(List)).

%% @doc Create a vector from a matrix
-spec vec_from_mat(Mat::mat()) -> vec().
vec_from_mat(Mat) ->
    V = do_copy(Mat),
    def_vec(V).

%% @doc Set start index on a vector
-spec set_start(Start::integer(), Vec::vec()) -> vec().
set_start(Start, Vec) ->
    Vec#{start:=Start}.

%% @doc Set interleave size on a vector
-spec set_inc(Inc::integer(), Vec::vec()) -> vec();
	     (Inc::integer(), Vec::mat()) -> mat().
set_inc(Inc, Vec) ->
    Vec#{inc:=Inc}.

%% @doc Set the dimension(s) on a vector or matrix
%% the following operations will only use 'dim' elements
-spec set_dim(N::integer(), Vec::vec()) -> vec();
	     ({M::integer(),N::integer()}, Mat::mat()) -> mat().
set_dim({M,N}, Mat=#{type:=matrix}) ->  Mat#{m:=M, n:=N};
set_dim(N, Vec=#{type:=vector}) ->  Vec#{n:=N}.

%% @doc Returns the vector size
-spec vec_size(Vec::vec()) -> N::integer().
vec_size(#{type:=vector, v:=Vec}) ->
    ?IMPL:cont_size(Vec).

%% @doc Get the value at Idx (zero based)
-spec value(Idx::integer(), Vec::vec()|mat()) -> Value::float().
value(Idx, #{v:=Vec}) ->
    ?IMPL:value(Idx,Vec).

%% @doc Get N values starting at Idx
-spec values(Idx::integer(), N::integer(), Vec::vec()|mat()) -> Vs::tuple().
values(Idx, N, #{v:=Vec}) ->
    ?IMPL:values(Idx, N, Vec).

%% @doc Get values at Indecies
-spec values(list(integer()), Vec::vec() | mat()) ->
		    list({Idx::integer(), Value::float()}).
values(IdxList, #{v:=Vec}) ->
    ?IMPL:values(IdxList, Vec).

%% @doc Update value(s) starting at Idx
-spec update(Idx::integer(),
	     Vs::float() | list(float()),
	     Vec::vec() | mat()) -> vec()|mat().
update(Idx, Vs, Vec) ->
    ?IMPL:update(Idx, Vs, Res = do_copy(Vec)),
    Vec#{v:=Res}.

%% @doc Update values at indexes
-spec update(list({Idx::integer(), Value::float()}), Vec::vec()|mat()) -> vec()|mat().
update(Vs, Vec) ->
    ?IMPL:update(Vs, Res = do_copy(Vec)),
    Vec#{v:=Res}.

%% @doc Create a matrix of size M*N and zero the values
-spec mat(M::non_neg_integer(), N::non_neg_integer()) -> vec().
mat(M, N) when M > 0, N > 0 ->
    def_mat(M,N,?IMPL:make_cont(M*N, true)).

%% @doc Create a matrix of size M*N and fill with value from fun/2
%% I ranges from 0..M-1 and J 0..N-1
-spec mat(M::non_neg_integer(), N::non_neg_integer(), Fun) -> mat()
    when Fun::fun((I::integer(),J::integer()) -> float()).
mat(M, N, Fun) when M > 0, N > 0 ->
    Mat = def_mat(M,N,?IMPL:make_cont(M*N, false)),
    fill_matrix(M,N,Fun,Mat#{destr:=true}),
    Mat.

%% @doc Convert a vec to a matrix
-spec mat_from_vec(M::integer(), N::integer(), Vec::vec()) -> mat().
mat_from_vec(M, N, V0=#{}) ->
    V = do_copy(V0),
    def_mat(M,N,V).

%% @doc Convert a tuple list to a matrix
-spec mat_from_list(list(tuple())) -> mat().
mat_from_list([T|_]=List) when is_tuple(T) ->
    def_mat(length(List),tuple_size(T),?IMPL:from_list(List)).

%% @doc Convert a tuple list to a matrix
-spec mat_from_list(M::integer(), N::integer(), list(float())) -> mat().
mat_from_list(M,N,List) ->
    def_mat(M,N,?IMPL:from_list(List)).

%% @doc Returns the matrix dimensions
%% Takes op in to consideration such that an MxN matrix
%% with an op set to transpose returns NxM
mat_size(#{op:=no_transp, m:=M, n:=N}) -> {M,N};
mat_size(#{m:=M, n:=N}) -> {N,M}.

%% @doc Set which part of the matrix is triangular
%% or which part of a symmetrical matrix should be used
-spec set_triangle(matrix_uplo(), mat()) -> mat().
set_triangle(UpLo, Mat) ->
    Mat#{uplo:=UpLo}.

%% @doc Set unit or non_unit matrix
-spec set_unit(matrix_diag(), mat()) -> mat().
set_unit(UpLo, Mat) ->
    Mat#{diag:=UpLo}.

%% @doc Set operation to be done on matrix
-spec set_op(matrix_op(), mat()) -> mat().
set_op(Op, Mat) ->
    Mat#{op:=Op}.

%% @doc Set the side to the matrix is in the op
%% i.e. Matrix*Vec or Vec*Matrix
-spec set_side(matrix_side(), mat()) -> mat().
set_side(Side, Mat) ->
    Mat#{side:=Side}.

%% @doc Convert a vector to a list of values
-spec to_list(VecOrMat::vec()|mat()) -> list(float()).
to_list(#{v:=Vec}) ->
    ?IMPL:to_list(Vec).

%% @doc Convert a matrix to a list of tuples
-spec to_tuple_list(Mat::mat()) -> list(tuple()).
to_tuple_list(#{type:=matrix, n:=N, v:=Vec}) ->
    ?IMPL:to_tuple_list(N, Vec).

%% @doc Convert a vector or matrix to a list of tuples
-spec to_tuple_list(Ts::integer(),Vec::vec()) -> list(tuple()).
to_tuple_list(TS, #{v:=Vec}) ->
    ?IMPL:to_tuple_list(TS, Vec).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% @doc Copy values from a vector or matrix
%% Start at 'start' pos and step 'inc' and copy n (or m*n) values
-spec copy(X::vec()) -> vec().
copy(X=#{type:=vector, start:=X0, inc:=Inc, v:=Vec0}) ->
    N    = cont_size(X),
    Vec1 = ?IMPL:copy(N, Vec0, X0, Inc),
    def_vec(Vec1);

copy(#{type:=matrix, m:=M, n:=N, v:=Vec0}) ->
    Vec1 = ?IMPL:copy(Vec0),
    def_mat(M,N,Vec1).

%% @doc Copy values from the vector to another
%% Start at 'start' pos and step 'inc' and copy n values
-spec copy(X::vec(), Y::vec()) -> Y1::vec().
copy(X=#{type:=vector, start:=X0, inc:=XInc, v:=XV},
     Y=#{type:=vector, start:=Y0, inc:=YInc}) ->
    N = cont_size(X),
    YV = do_copy(Y),
    ?IMPL:copy(N, XV, X0, XInc, YV, Y0, YInc),
    Y#{v:=YV}.

%% @doc Swap vectors
%% Given two vectors x and y, the swap routines return vectors y
%% and x swapped, each replacing the other.
-spec swap(X::vec(), Y::vec()) -> {X1::vec(),Y1::vec()}.
swap(X=#{type:=vector, start:=X0, inc:=XInc},
     Y=#{type:=vector, start:=Y0, inc:=YInc}) ->
    N = cont_size(X),
    N =:= cont_size(Y) orelse error(badarg),
    XV=do_copy(X),
    YV=do_copy(Y),
    ?IMPL:swap(N, XV, X0, XInc, YV, Y0, YInc),
    {X#{v:=XV}, Y#{v:=YV}}.

%% @doc
%%  Setup Givens rotation from A,B
-spec rotg(A::float(),B::float()) ->
		  {R::float(),Z::float(),Cos::float(),Sin::float()}.
rotg(A,B) -> ?IMPL:rotg(A,B).

%% @doc Apply plane rotation on 2D interleaved data
%%   where Cos and Sin are the angle of rotation
-spec rot(A2D::vec(), Cos::float(), Sin::float()) -> vec().
rot(XY=#{type:=vector, start:=X0, inc:=Inc}, C, S) when Inc > 1 ->
    N   = cont_size(XY),
    Vec = do_copy(XY),
    ?IMPL:rot(N div 2, Vec, X0, Inc, Vec, X0+1, Inc, C, S),
    XY#{v:=Vec}.

%% @doc Apply plane rotation on vectors
-spec rot(X::vec(), Y::vec(), Cos::float(), Sin::float()) -> {vec(),vec()}.
rot(X=#{type:=vector, start:=X0, inc:=XInc},
    Y=#{type:=vector, start:=Y0, inc:=YInc}, C,S) ->
    N = cont_size(X),
    VX = do_copy(X),
    VY = do_copy(Y),
    ?IMPL:rot(N, VX, X0, XInc, VY, Y0, YInc, C, S),
    {X#{v:=VX}, Y#{v:=VY}}.

%% @doc Scale values in the vector with Alpha
%% Alpha*X=>X'
-spec scal(Alpha::float(), Cont::vec()) -> vec().
scal(Alpha, X=#{type:=vector, start:=X0, inc:=XInc}) ->
    N = cont_size(X),
    VX = do_copy(X),
    ?IMPL:scal(N, Alpha, VX, X0, XInc),
    X#{v:=VX}.

%% @doc Computes the sum of the absolute values of X
%%  i.e. computes sum |X|
%%
-spec asum(X::vec()) -> float().
asum(X=#{type:=vector, start:=X0, inc:=XInc, v:=Vec}) ->
    N = cont_size(X),
    ?IMPL:asum(N, Vec, X0, XInc).

%% @doc Compute alpha*X+Y=>Y'
-spec axpy(Alpha::float(), X::vec(), Y::vec()) -> vec().
axpy(Alpha,
     X=#{type:=vector, start:=X0, inc:=XInc},
     Y=#{type:=vector, start:=Y0, inc:=YInc}) ->
    N = cont_size(X),
    N =:= cont_size(Y) orelse error(badarg),
    XV=do_copy(X),
    YV=do_copy(Y),
    ?IMPL:axpy(N, Alpha, XV, X0, XInc, YV, Y0, YInc),
    Y#{v:=YV}.

%% @doc vector vector dot product
%%   sum([Xi*Yi'])
-spec dot(X::vec(), Y::vec()) -> float().
dot(X=#{type:=vector, start:=X0, inc:=XInc, v:=VX},
    Y=#{type:=vector, start:=Y0, inc:=YInc, v:=VY}) ->
    N = cont_size(X),
    N =:= cont_size(Y) orelse error(badarg),
    ?IMPL:dot(N, VX, X0, XInc, VY, Y0, YInc).

%% @doc Computes the Euclidian norm of the elements of X
%%  i.e. computes ||X||
-spec nrm2(X::vec()) -> float().
nrm2(X=#{type:=vector, start:=X0, inc:=XInc, v:=VX}) ->
    N = cont_size(X),
    ?IMPL:nrm2(N, VX, X0, XInc).

%% @doc Locates the max value
%%  (index counted from 'start')
-spec iamax(X::vec()) -> {Index::integer(), Max::float()}.
iamax(X=#{type:=vector, start:=X0, inc:=XInc, v:=VX}) ->
    N = cont_size(X),
    ?IMPL:iamax(N, VX, X0, XInc).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% @doc Perform a generic M*N matrix vector operation
%%  no_transp:    Alpha*A*x+Beta*Y => Y'
%%  transp:       Alpha*A'*x+Beta*Y => Y'
%%  conj_transp:  Alpha*conjug(A')*x+Beta*Y => Y'
%% where:
%%  A is an M*N matrix
-spec gemv(Alpha::float(), A::mat(), X::vec(), Beta::float(), Y::vec()) -> Y1::vec().
gemv(Alpha, #{order:=Ord, op:=Op, m:=M, n:=N, inc:=AInc, v:=AV},
     #{start:=X0, inc:=XInc, v:=XV}, Beta,
     Y=#{start:=Y0, inc:=YInc}) ->
    YV = do_copy(Y),
    ?IMPL:gemv(Ord, Op, M, N, Alpha, AV, AInc,
	       XV, X0, XInc, Beta, YV, Y0, YInc),
    Y#{v:=YV}.

%% @doc Perform a symmetric M*N matrix vector operation
%%      same as gemv but with a symmetric matrix.
%% Alpha*A*X+Beta*Y => Y'
-spec symv(Alpha::float(), A::mat(), X::vec(), Beta::float(), Y::vec()) -> Y1::vec().
symv(Alpha, A=#{order:=Ord, uplo:=Uplo, n:=N, inc:=AInc, v:=AV},
     #{start:=X0, inc:=XInc, v:=XV}, Beta,
     Y=#{start:=Y0, inc:=YInc}) ->
    check_triangular(A),
    YV = do_copy(Y),
    ?IMPL:symv(Ord, Uplo, N, Alpha, AV, AInc,
	       XV, X0, XInc, Beta, YV, Y0, YInc),
    Y#{v:=YV}.

%% @doc Perform a packed symmetric M*N matrix vector operation
%% same as symv but with a packed symmetric matrix,
%% i.e. only half matrix is expected in A.
%% Alpha*A*X+Beta*Y => Y'
-spec spmv(Alpha::float(), A::mat(),
	   X::vec(), Beta::float(), Y::vec()) -> Y1::vec().
spmv(Alpha, A=#{order:=Ord, uplo:=Uplo, n:=N, v:=AV},
     #{start:=X0, inc:=XInc, v:=XV}, Beta,
     Y=#{start:=Y0, inc:=YInc}) ->
    check_triangular(A),
    YV = do_copy(Y),
    ?IMPL:spmv(Ord, Uplo, N, Alpha, AV,
	       XV, X0, XInc, Beta, YV, Y0, YInc),
    Y#{v:=YV}.

%% @doc Perform a triangular matrix * vector
%%  no_transp:    A*X => X'
%%  transp:       A'*X => X'
%%  conj_transp:  conjug(A')*X => X'
%% where:
%%  A is an N*N matrix upper or lower matrix
-spec trmv(A::mat(), X::vec()) -> Y1::vec().
trmv(A=#{order:=Ord, uplo:=Uplo, op:=Op, diag:=Diag, n:=N, inc:=AInc, v:=AV},
     X=#{start:=X0, inc:=XInc}) ->
    check_triangular(A),
    XV = do_copy(X),
    ?IMPL:trmv(Ord, Uplo, Op, Diag, N, AV, AInc, XV, X0, XInc),
    X#{v:=XV}.

%% @doc Perform a packed triangular matrix * vector
%% Same as trmv/2,
-spec tpmv(A::mat(), X::vec()) -> Y1::vec().
tpmv(A=#{order:=Ord, uplo:=Uplo, op:=Op, diag:=Diag, n:=N, v:=AV},
     X=#{start:=X0, inc:=XInc}) ->
    check_triangular(A),
    XV = do_copy(X),
    ?IMPL:tpmv(Ord, Uplo, Op, Diag, N, AV, XV, X0, XInc),
    X#{v:=XV}.

%% @doc Solve a system of linear equations whose coefficients are in a triangular matrix.
%%  A*B => X
-spec trsv(A::mat(), X::vec()) -> B::vec().
trsv(A=#{order:=Ord, uplo:=Uplo, op:=Op, diag:=Diag,
	 n:=N, inc:=AInc, v:=AV},
     X=#{start:=X0, inc:=XInc}) ->
    check_triangular(A),
    XV = do_copy(X),
    ?IMPL:trsv(Ord, Uplo, Op, Diag, N, AV, AInc, XV, X0, XInc),
    X#{v:=XV}.

%% @doc Solve a system of linear equations whose coefficients are in a packed triangular matrix.
%%  A*B => X
-spec tpsv(A::mat(), X::vec()) -> B::vec().
tpsv(A=#{order:=Ord, uplo:=Uplo, op:=Op, diag:=Diag, n:=N, v:=AV},
     X=#{start:=X0, inc:=XInc}) ->
    check_triangular(A),
    XV = do_copy(X),
    ?IMPL:tpsv(Ord, Uplo, Op, Diag, N, AV, XV, X0, XInc),
    X#{v:=XV}.

%% @doc Performs a rank-1 update of a general matrix.
%%   alpha*X*Y'+A => A'
-spec ger(Alpha::float(), X::vec(), Y::vec(), A::mat()) -> A1::mat().
ger(Alpha, #{start:=X0, inc:=XInc, v:=XV},
    #{start:=Y0, inc:=YInc, v:=YV},
    A=#{order:=Ord, m:=M, n:=N, inc:=AInc}) ->
    AV = do_copy(A),
    ?IMPL:ger(Ord, M, N, Alpha, XV, X0, XInc, YV, Y0, YInc, AV, AInc),
    A#{v:=AV}.

%% @doc Performs a rank-1 update of a symmetric matrix.
%%   alpha*X*X'+A => A'
-spec syr(Alpha::float(), X::vec(), A::mat()) -> A1::mat().
syr(Alpha, #{start:=X0, inc:=XInc, v:=XV},
    A=#{order:=Ord, uplo:=UpLo, n:=N, inc:=AInc}) ->
    check_triangular(A),
    AV = do_copy(A),
    ?IMPL:syr(Ord, UpLo, N, Alpha, XV, X0, XInc, AV, AInc),
    A#{v:=AV}.

%% @doc Performs a rank-1 update of a symmetric packed matrix.
%%   alpha*X*X'+A => A'
%% same syr
-spec spr(Alpha::float(), X::vec(), A::mat()) -> A1::mat().
spr(Alpha, #{start:=X0, inc:=XInc, v:=XV},
    A=#{order:=Ord, uplo:=UpLo, n:=N}) ->
    check_triangular(A),
    AV = do_copy(A),
    ?IMPL:spr(Ord, UpLo, N, Alpha, XV, X0, XInc, AV),
    A#{v:=AV}.

%% @doc Performs a rank-2 update of a symmetric matrix.
%%   alpha*X*Y'+ alpha*Y*X' + A => A'
-spec syr2(Alpha::float(), X::vec(), Y::vec(), A::mat()) -> A1::mat().
syr2(Alpha, #{start:=X0, inc:=XInc, v:=XV},
     #{start:=Y0, inc:=YInc, v:=YV},
     A=#{order:=Ord, uplo:=UpLo, n:=N, inc:=AInc}) ->
    check_triangular(A),
    AV = do_copy(A),
    ?IMPL:syr2(Ord, UpLo, N, Alpha, XV, X0, XInc, YV, Y0, YInc, AV, AInc),
    A#{v:=AV}.

%% @doc Performs a rank-2 update of a packed symmetric matrix.
%%   alpha*x*y'+ alpha*y*x' + A => A
-spec spr2(Alpha::float(), X::vec(), Y::vec(), A::mat()) -> A1::mat().
spr2(Alpha, #{start:=X0, inc:=XInc, v:=XV},
     #{start:=Y0, inc:=YInc, v:=YV},
     A=#{order:=Ord, uplo:=UpLo, n:=N}) ->
    check_triangular(A),
    AV = do_copy(A),
    ?IMPL:spr2(Ord, UpLo, N, Alpha, XV, X0, XInc, YV, Y0, YInc, AV),
    A#{v:=AV}.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Level 3

%% @doc Perform a matrix-matrix operation with general matrices
%%
%%  alpha*op(A)*op(B) + beta*C => C'
%%  where Op(A) and Op(B) is described by OpA and OpB.
%%  and
%%    opA(A) MxK matrix
%%    opB(B) KxN matrix
%%    C MxN matrix
-spec gemm(Alpha::float(), A::mat(), B::mat(), Beta::float(), C::mat()) -> C1::mat().
gemm(Alpha,
     A=#{order:=Ord, op:=OpA, inc:=AInc, v:=AV},
     B=#{order:=Ord, op:=OpB, inc:=BInc, v:=BV},
     Beta, C=#{order:=Ord, m:=M, n:=N, inc:=CInc}) ->
    CV = do_copy(C),
    {M,K} = mat_size(A),
    {K,N} = mat_size(B),
    ?IMPL:gemm(Ord, OpA, OpB, M, N, K,
	       Alpha, AV, AInc, BV, BInc, Beta, CV, CInc),
    C#{v:=CV}.

%% @doc Perform a matrix-matrix operation with a symmetric matrix
%%
%%  Symmetric matrix A and B and C are  MxN matrices
%%  Depending on Side:
%%    left: alpha*A*B + beta*C => C'
%%   right: alpha*B*A + beta*C => C'
-spec symm(Alpha::float(),
	   A::mat(),
	   B::mat(),
	   Beta::float(), C::mat()) -> C1::mat().
symm(Alpha,
     A=#{order:=Ord, side:=Side, uplo:=UpLo, inc:=AInc, v:=AV},
     #{order:=Ord, m:=M, n:=N, inc:=BInc, v:=BV},
     Beta, C=#{order:=Ord, m:=M, n:=N, inc:=CInc}) ->
    check_triangular(A),
    CV = do_copy(C),
    ?IMPL:symm(Ord, Side, UpLo, M, N,
	       Alpha, AV, AInc, BV, BInc, Beta, CV, CInc),
    C#{v:=CV}.

%% @doc Perform a triangular matrix-matrix operation
%% alpha*op( A )*B => B
%%   or
%% alpha*B*op( A ) => B
%% where alpha is a scalar, B is an M by N matrix, A is a unit, or
%% non-unit, upper or lower triangular matrix and op( A ) is one  of
%% op( A ) = A   or   op( A ) = A'.
-spec trmm(Alpha::float(), A::mat(), B::mat()) -> B1::mat().
trmm(Alpha,
     A=#{order:=Ord, side:=Side, uplo:=UpLo, op:=Op, diag:=Diag, inc:=AInc, v:=AV},
     B=#{order:=Ord, m:=M, n:=N, inc:=BInc}) ->
    check_triangular(A),
    BV = do_copy(B),
    ?IMPL:trmm(Ord, Side, UpLo, Op, Diag, M, N, Alpha, AV, AInc, BV, BInc),
    B#{v:=BV}.

%% @doc solves one of the matrix equations
%%
%%   op( A )*X = alpha*B => X,   or   X*op( A ) = alpha*B => X
%%
%% where alpha is a scalar, X and B are m by n matrices, A is a unit, or
%% non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
%% op( A ) = A   or   op( A ) = A'.
%%
-spec trsm(Alpha::float(), A::mat(), B::mat()) -> B1::mat().
trsm(Alpha,
     A=#{order:=Ord, side:=Side, uplo:=UpLo, op:=Op, diag:=Diag, inc:=AInc, v:=AV},
     B=#{order:=Ord, m:=M, n:=N, inc:=BInc}) ->
    check_triangular(A),
    BV = do_copy(B),
    ?IMPL:trsm(Ord, Side, UpLo, Op, Diag, M, N, Alpha, AV, AInc, BV, BInc),
    B#{v:=BV}.

%% @doc syrk performs one of the symmetric rank k operations
%%    alpha*A*A' + beta*C => C'
%% or
%%    alpha*A'*A + beta*C => C'
%%
%% where alpha and beta are scalars, C is an  n by n  symmetric matrix
%% and A is an n by k matrix in the first case and a  k by n  matrix
%% in the second case.
-spec syrk(Alpha::float(), A::mat(), Beta::float(), C::mat()) -> C1::mat().
syrk(Alpha,
     A=#{order:=Ord, op:=Op, inc:=AInc, v:=AV},
     Beta, C=#{order:=Ord, uplo:=UpLo, m:=N, n:=N, inc:=CInc}) ->
    check_triangular(C),
    CV = do_copy(C),
    {N, K} = mat_size(A),
    ?IMPL:syrk(Ord, UpLo, Op, N, K, Alpha, AV, AInc, Beta, CV, CInc),
    C#{v:=CV}.

%% @doc Perform a symmetrik rank-2k operation
%%   alpha*A*B' + alpha*B*A' + beta*C => C'
%% or
%%   alpha*A'*B + alpha*B'*A + beta*C => C'
%%
%% where  alpha and beta  are scalars, C is an  n by n  symmetric matrix
%% and  A and B  are  n by k  matrices  in the  first  case  and  k by n
%% matrices in the second case.
-spec syr2k(Alpha::float(), A::mat(), B::mat(),
	    Beta::float(), C::mat()) -> C1::mat().
syr2k(Alpha,
      A=#{order:=Ord, op:=Op, inc:=AInc, v:=AV},
      B=#{order:=Ord, v:=BV, inc:=BInc},
      Beta, C=#{order:=Ord, uplo:=UpLo, m:=N, n:=N, inc:=CInc}) ->
    check_triangular(C),
    CV = do_copy(C),
    {N, K} = mat_size(A),
    {N, K} = mat_size(B),
    ?IMPL:syr2k(Ord, UpLo, Op, N, K, Alpha, AV, AInc, BV, BInc, Beta, CV, CInc),
    C#{v:=CV}.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Size requirements should be checked in nif code
cont_size(#{n:=all, start:= X0, v:=Vec, inc:=Inc}) ->
    (?IMPL:cont_size(Vec) - X0) div Inc;
cont_size(#{n:=N}) ->
    N.

check_triangular(#{uplo:=undefined}) -> error({badarg, triangle_not_set});
check_triangular(#{type:=matrix}) -> ok.

do_copy(#{destr:=true, v:=V}) -> V;
do_copy(#{destr:=false, v:=V}) -> ?IMPL:copy(V).

def_vec(Vec) ->
    #{type  => vector,
      start => 0,
      inc   => 1,
      n     => all,
      destr => false,
      v => Vec}.

def_mat(M,N,Vec) ->
  #{type => matrix,
    m => M,
    n => N,
    inc => N,
    order => row_maj,
    uplo  => undefined,
    diag  => non_unit,
    op    => no_transp,
    side  => left,
    destr => false,
    v => Vec
   }.

fill_matrix(M, N, F, Mat) when is_function(F)->
    Ref = erlang:make_ref(),
    NoWorkers = erlang:system_info(schedulers_online),
    RowsPerWorker = max(1, M div NoWorkers),
    _ = fill_matrix_func_1pp(M-1, RowsPerWorker, N, Ref, F, Mat),
    wait(min(NoWorkers,M), Ref),
    Mat.

fill_matrix_func_1pp(I, Rows, N, Ref, F, Mat) when I >= 0 ->
    Parent = self(),
    spawn(fun() ->
		  fill_matrix_func_2pp(N-1, I, N, max(0, I - Rows), F, Mat, []),
		  Parent ! {Ref, I}
	  end),
    fill_matrix_func_1pp(I-Rows, Rows, N, Ref, F, Mat);
fill_matrix_func_1pp(_M, _Rows, _N, _Ref, _F, _Mat) ->
    ok.

fill_matrix_func_2pp(J, I, N, Stop, F, Mat, Acc) when J >= 0 ->
    fill_matrix_func_2pp(J-1, I, N, Stop, F, Mat, [F(I,J)|Acc]);
fill_matrix_func_2pp(_, I, N, Stop, F, Mat0, Acc) ->
    Mat = blas:update(I*N, Acc, Mat0),
    if I > Stop -> fill_matrix_func_2pp(N-1, I-1, N, Stop, F, Mat, []);
       true -> Mat
    end.

wait(M, Ref) when M > 0 ->
    receive
	{Ref, _I} -> wait(M-1,Ref)
    end;
wait(_, _) -> ok.
