%%%-------------------------------------------------------------------
%%% @author Dan Gudmundsson <dgud@erlang.org>
%%% @copyright (C) 2014, Dan Gudmundsson
%%% @doc  A wrapper for the standard BLAS Basic Linnear Algebra
%%%
%%%       Most of these functions use desctructive operations on the supplied vectors.
%%%
%%% @end
%%% Created :  9 Dec 2014 by Dan Gudmundsson <dgud@erlang.org>
%%%-------------------------------------------------------------------
-module(blasd).

-export([from_list/1, to_list/1, to_tuple_list/2, vec_size/1]).
%% Level 1
-export([ %% Level 1
	  rotg/2,
	  rot/3, rot/9,
	  scal/2, scal/5,
	  copy/1, copy/4,
	  axpy/3, axpy/8,
	  swap/7,
	  dot/2, dot/7,
	  nrm2/1, nrm2/4,
	  asum/1, asum/4,
	  iamax/1, iamax/4,
	  %% Level 2
	  gemv/7, gemv/14,
	  symv/7, symv/13, spmv/7, spmv/12,
	  trmv/5, trmv/10, tpmv/5, tpmv/9,
	  trsv/5, trsv/10, tpsv/5, tpsv/9,
	  ger/6, ger/12,
	  syr/5, syr/9, spr/5, spr/8,
	  syr2/6, syr2/12, spr2/6, spr2/11
	  %% Level 3

	]).

-opaque(vec).
-type vec() :: binary().
-type matrix_op() :: no_transp|transp|conj_transp.
-type matrix_order() :: row_maj|col_maj.
-type matrix_uplo()  :: upper|lower.
-type matrix_diag() :: unit|non_unit.

-on_load(on_load/0).

-define(nif_stub,nif_stub_error(?LINE)).
nif_stub_error(Line) ->
    erlang:nif_error({nif_not_loaded,module,?MODULE,line,Line}).

%% API Data conversion
-spec from_list(List::[tuple()]|list(float())) -> vec().
from_list(_List) -> ?nif_stub.

-spec to_list(Vec::vec()) -> list(float()).
to_list(Vec) -> to_tuple_list(1, Vec).

-spec to_tuple_list(Ts::integer(),Vec::vec()) -> list(tuple()).
to_tuple_list(_TS, _Vec) -> ?nif_stub.

-spec vec_size(Vec::vec()) -> N::integer().
vec_size(_Vec) -> ?nif_stub.

%% API LEVEL 1

%% @doc
%%  Setup Givens rotation from A,B
-spec rotg(A::float(),B::float()) ->
		   {R::float(),Z::float(),Cos::float(),Sin::float()}.

rotg(_A,_B) -> ?nif_stub.

%% @doc Apply plane rotation on 2D interleaved data
%%   where Cos and Sin are the angle of rotation
-spec rot(A2D::vec(), Cos::float(), Sin::float()) -> ok.
rot(XY, C, S) ->
    N = vec_size(XY),
    rot(N div 2, XY, 0, 2, XY, 1, 2, C, S).

%% @doc Apply plane rotation in vectors
%%   N Number of elements to apply data to
%%   X,Y raw data
%%   XStart,YStart  Start positions
%%   XInc, YInc     Increments to next element (1 if not interleaved)
-spec rot(N::integer(),
	   X::vec(), XStart::integer(), Xinc::integer(),
	   Y::vec(), YStart::integer(), Yinc::integer(),
	   Cos::float(), Sin::float()) -> ok.
rot(_N, _X, _XStart, _Xinc, _Y, _YStart, _Yinc, _C, _S) -> ?nif_stub.

%% @doc Scale all values in the vector with Alpha
-spec scal(Alpha::float(), Vec::vec()) -> ok.
scal(Alpha, X) ->
    N = vec_size(X),
    scal(N, Alpha, X, 0, 1).

%% @doc Scale values in the vector with Alpha
%% Start at StartX pos and step Xinc
-spec scal(N::integer(), Alpha::float(),
	   X::vec(), XStart::integer(), Xinc::integer()) -> ok.
scal(N, Alpha, X, XStart, Xinc) ->
    one_vec(N, Alpha, X, XStart, Xinc, 0).

%% @doc Copy all values in the vector
-spec copy(Vec::vec()) -> vec().
copy(X) ->
    N = vec_size(X),
    copy(N, X, 0, 1).

%% @doc Copy values in the vector
%% Start at StartX pos and step Xinc
-spec copy(N::integer(), X::vec(), XStart::integer(), Xinc::integer()) -> vec().
copy(_N, _X, _XStart, _Xinc) -> ?nif_stub.

%% @doc Compute AX+Y=Y
-spec axpy(A::float(), X::vec(), Y::vec()) -> ok.
axpy(A, X, Y) ->
    N = vec_size(X),
    N =:= vec_size(Y) orelse error(badarg),
    axpy(N, A, X, 0, 1, Y, 0, 1).

%% @doc Compute AX+Y=Y
-spec axpy(N::integer(), Alpha::float(),
	    X::vec(), XStart::integer(), Xinc::integer(),
	    Y::vec(), YStart::integer(), Yinc::integer()) -> ok.
axpy(N, A, X, XStart, Xinc, Y, YStart, Yinc) ->
    two_vec(N, A, X, XStart, Xinc, Y, YStart, Yinc, 0).

%% @doc Swap vectors
%% Given two vectors x and y, the swap routines return vectors y
%% and x swapped, each replacing the other.
-spec swap(N::integer(),
	    X::vec(), XStart::integer(), Xinc::integer(),
	    Y::vec(), YStart::integer(), Yinc::integer()) -> ok.
swap(N, X, XStart, Xinc, Y, YStart, Yinc) ->
    two_vec(N, 0.0, X, XStart, Xinc, Y, YStart, Yinc, 1).

%% @doc vector vector dot product
%%   sum([Xi*YiT])
-spec dot(X::vec(), Y::vec()) -> float().
dot(X,Y) ->
    N = vec_size(X),
    N =:= vec_size(Y) orelse error(badarg),
    dot(N, X, 0, 1, Y, 0, 1).

-spec dot(N::integer(),
	   X::vec(), XStart::integer(), Xinc::integer(),
	   Y::vec(), YStart::integer(), Yinc::integer()) -> float().
dot(N, X, XStart, Xinc, Y, YStart, Yinc) ->
    two_vec(N, 0.0, X, XStart, Xinc, Y, YStart, Yinc, 2).

two_vec(_N, _A, _X, _XStart, _Xinc, _Y, _YStart, _Yinc, _Op) -> ?nif_stub.


%% @doc Computes the Euclidian norm of the elements of X
%%  i.e. computes ||X||
%%
-spec nrm2(X::vec()) -> float().
nrm2(X) ->
    N = vec_size(X),
    nrm2(N, X, 0, 1).

-spec nrm2(N::integer(),
	   X::vec(), XStart::integer(), Xinc::integer()) -> float().
nrm2(N, X, XStart, Xinc) ->
    one_vec(N, 0.0, X, XStart, Xinc, 1).

%% @doc Computes the sum of the absolute values of X
%%  i.e. computes sum |X|
%%
-spec asum(X::vec()) -> float().
asum(X) ->
    N = vec_size(X),
    asum(N, X, 0, 1).

-spec asum(N::integer(),
	    X::vec(), XStart::integer(), Xinc::integer()) -> float().
asum(N, X, XStart, Xinc) ->
    one_vec(N, 0.0, X, XStart, Xinc, 2).

%% @doc Locates the max value
%%  (index from Xstart)
-spec iamax(X::vec()) -> {Index::integer(), Max::float()}.
iamax(X) ->
    N = vec_size(X),
    iamax(N, X, 0, 1).

-spec iamax(N::integer(),
	     X::vec(), XStart::integer(), Xinc::integer()) ->
		    {Index::integer(), Max::float()}.
iamax(N, X, XStart, Xinc) ->
    one_vec(N, 0.0, X, XStart, Xinc, 3).

one_vec(_N, _Alpha, _X, _XStart, _Xinc, _Op) -> ?nif_stub.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Level 2


%% @doc Perform a generic M*N matrix vector operation
%%  no_transp:    Alpha*A*x+Beta*Y = Y
%%  transp:       Alpha*A'*x+Beta*Y = Y
%%  conj_transp:  Alpha*conjug(A')*x+Beta*Y = Y
%% where:
%%  Trans is one of normal, transpose, conjugate
%%  M is the number of rows
%%  N is the number of columns
%%  A is an M*N matrix
%%  Stride specifies the size of the first dimension of A
%%         i.e. >= N on row_major and >= M on col_major
%%  X and Y are vectors fo length ( 1 + ( N - 1 )*abs( Incx ) ) when Trans is normal
%%          and otherwise ( 1 + ( M - 1 )*abs( Incx ) )
-spec gemv(M::integer(), N::integer(),
	   Alpha::float(), A::vec(),
	   X::vec(), Beta::float(), Y::vec()) -> ok.
gemv(M, N, Alpha, A, X, Beta, Y) ->
    gemv(row_maj, no_transp, M, N, Alpha, A, N, X, 0, 1, Beta, Y, 0, 1).
-spec gemv(Order::matrix_order(),Trans::matrix_op(),
	    M::integer(), N::integer(),
	    Alpha::float(), A::vec(), Stride::integer(),
	    X::vec(), StartX::integer(), IncX::integer(),
	    Beta::float(),
	    Y::vec(), StartY::integer(), IncY::integer()) -> ok.
gemv(Ord, Trans, M, N, Alpha, A, Stride,
     X, StartX, IncX, Beta, Y, StartY, IncY) ->
    gemv_impl(Ord, Trans, M, N, Alpha, A, Stride,
	      X, StartX, IncX, Beta, Y, StartY, IncY, 0).

gemv_impl(_Ord, _Trans, _M, _N, _Alpha, _A, _Stride,
	  _X, _StartX, _IncX, _Beta, _Y, _StartY, _IncY, _Op) ->
  ?nif_stub.

%% @doc Perform a symmetric M*N matrix vector operation
%%        same as gemv but with a symmetric matrix.
-spec symv(UpLo::matrix_uplo(), N::integer(),
	   Alpha::float(), A::vec(),
	   X::vec(), Beta::float(), Y::vec()) -> ok.
symv(UpLo, N, Alpha, A, X, Beta, Y) ->
    symv(row_maj, UpLo, N, Alpha, A, N, X, 0, 1, Beta, Y, 0, 1).
-spec symv(Order::matrix_order(), UpLo::matrix_uplo(),
	   N::integer(),
	   Alpha::float(), A::vec(), Stride::integer(),
	   X::vec(), StartX::integer(), IncX::integer(),
	   Beta::float(),
	   Y::vec(), StartY::integer(), IncY::integer()) -> ok.
symv(Ord, UpLo, N, Alpha, A, Stride, X, StartX, IncX, Beta, Y, StartY, IncY) ->
    gemv_impl(Ord, UpLo, 0, N, Alpha, A, Stride, X, StartX, IncX, Beta, Y, StartY, IncY, 1).

%% @doc Perform a packed symmetric M*N matrix vector operation
%%        same as gemv but with a packed symmetric matrix,
%%        i.e. only half matrix is expected in A.
-spec spmv(UpLo::matrix_uplo(), N::integer(),
	   Alpha::float(), A::vec(),
	   X::vec(), Beta::float(), Y::vec()) -> ok.
spmv(UpLo, N, Alpha, A, X, Beta, Y) ->
    spmv(row_maj, UpLo, N, Alpha, A, X, 0, 1, Beta, Y, 0, 1).
-spec spmv(Order::matrix_order(), UpLo::matrix_uplo(),
	   N::integer(),
	   Alpha::float(), A::vec(),
	   X::vec(), StartX::integer(), IncX::integer(),
	   Beta::float(),
	   Y::vec(), StartY::integer(), IncY::integer()) -> ok.
spmv(Ord, UpLo, N, Alpha, A, X, StartX, IncX, Beta, Y, StartY, IncY) ->
    gemv_impl(Ord, UpLo, 1, N, Alpha, A, 0, X, StartX, IncX, Beta, Y, StartY, IncY, 2).

%% @doc Perform a triangular matrix * vector
%%  no_transp:    A*X = X
%%  transp:       A'*X = X
%%  conj_transp:  conjug(A')*X = Y
%% where:
%%  Trans is one of normal, transpose, conjugate
%%  A is an N*N matrix upper or lower matrix
%%  Stride specifies the size of the first dimension of A
%%         i.e. >= N on row_major and >= M on col_major
%%  X and Y are vectors of length ( 1 + ( N - 1 )*abs( Inc ) ) when Trans is normal
%%          and otherwise ( 1 + ( M - 1 )*abs( Inc ) )
-spec trmv(UpLo::matrix_uplo(), Diag::matrix_diag(), N::integer(), A::vec(), X::vec()) -> ok.
trmv(UpLo, Diag, N, A, X) ->
    trmv(row_maj, UpLo, no_transp, Diag, N, A, N, X, 0, 1).
-spec trmv(Order::matrix_order(), UpLo::matrix_uplo(),
	   Trans::matrix_op(), Diag::matrix_diag(),
	   N::integer(), A::vec(), Stride::integer(),
	   X::vec(), StartX::integer(), IncX::integer()) -> ok.
trmv(Ord, Uplo, Trans, Diag, N, A, Stride, X, StartX, IncX) ->
    trmv_impl(Ord, Uplo, Trans, Diag, N, A, Stride, X, StartX, IncX, 0).

trmv_impl(_Ord, _Uplo, _Trans, _Diag, _N, _A, _Stride,
	  _X, _StartX, _IncX, _Op) -> ?nif_stub.
trmv_impl(_Ord, _Uplo, _Trans, _Diag, _N, _A, _Stride,
	  _X, _StartX, _IncX, _Op, _Alpha) -> ?nif_stub.


%% @doc Perform a packed triangular matrix * vector
%%  see above
-spec tpmv(UpLo::matrix_uplo(), Diag::matrix_diag(), N::integer(), A::vec(), X::vec()) -> ok.
tpmv(UpLo, Diag, N, A, X) ->
    tpmv(row_maj, UpLo, no_transp, Diag, N, A, X, 0, 1).
-spec tpmv(Order::matrix_order(), UpLo::matrix_uplo(),
	   Trans::matrix_op(), Diag::matrix_diag(),
	   N::integer(), A::vec(),
	   X::vec(), StartX::integer(), IncX::integer()) -> ok.
tpmv(Ord, Uplo, Trans, Diag, N, A, X, StartX, IncX) ->
    trmv_impl(Ord, Uplo, Trans, Diag, N, A, 0, X, StartX, IncX, 1).

%% @doc Solve a system of linear equations whose coefficients are in a triangular matrix.
%%  A*b = x  (where x is overwritten with the content of b)
-spec trsv(UpLo::matrix_uplo(), Diag::matrix_diag(), N::integer(), A::vec(), X::vec()) -> ok.
trsv(UpLo, Diag, N, A, X) ->
    trsv(row_maj, UpLo, no_transp, Diag, N, A, N, X, 0, 1).
-spec trsv(Order::matrix_order(), UpLo::matrix_uplo(),
	   Trans::matrix_op(), Diag::matrix_diag(),
	   N::integer(), A::vec(), Stride::integer(),
	   X::vec(), StartX::integer(), IncX::integer()) -> ok.
trsv(Ord, Uplo, Trans, Diag, N, A, Stride, X, StartX, IncX) ->
    trmv_impl(Ord, Uplo, Trans, Diag, N, A, Stride, X, StartX, IncX, 2).

%% @doc Packed variant of trsv
-spec tpsv(UpLo::matrix_uplo(), Diag::matrix_diag(), N::integer(), A::vec(), X::vec()) -> ok.
tpsv(UpLo, Diag, N, A, X) ->
    tpsv(row_maj, UpLo, no_transp, Diag, N, A, X, 0, 1).
-spec tpsv(Order::matrix_order(), UpLo::matrix_uplo(),
	   Trans::matrix_op(), Diag::matrix_diag(),
	   N::integer(), A::vec(),
	   X::vec(), StartX::integer(), IncX::integer()) -> ok.
tpsv(Ord, Uplo, Trans, Diag, N, A, X, StartX, IncX) ->
    trmv_impl(Ord, Uplo, Trans, Diag, N, A, 0, X, StartX, IncX, 3).

%% @doc Performs a rank-1 update of a general matrix.
%%   alpha*x*y'+A = A
-spec ger(M::integer(), N::integer(),
	  Alpha::float(), X::vec(), Y::vec(), A::vec()) -> ok.
ger(M, N, Alpha, X, Y, A) ->
    ger(row_maj, M, N, Alpha, X, 0, 1, Y, 0, 1, A, N).

-spec ger(Order::matrix_order(),
	  M::integer(), N::integer(), Alpha::float(),
	  X::vec(), StartX::integer(), IncX::integer(),
	  Y::vec(), StartY::integer(), IncY::integer(),
	  A::vec(), Stride::integer()) -> ok.
ger(Ord, M, N, Alpha,
    X, StartX, IncX,
    Y, StartY, IncY,
    A, Stride) ->
    gemv_impl(Ord, no_transp, M, N, Alpha, A, Stride,
	      X, StartX, IncX, 0.0, Y, StartY, IncY, 3).

%% @doc Performs a rank-1 update of a symmetric matrix.
%%   alpha*x*x'+A = A
-spec syr(UpLo::matrix_uplo(), N::integer(), Alpha::float(), X::vec(), A::vec()) -> ok.
syr(UpLo, N, Alpha, X, A) ->
    syr(row_maj, UpLo, N, Alpha, X, 0, 1, A, N).
-spec syr(Order::matrix_order(), UpLo::matrix_uplo(),
	  N::integer(), Alpha::float(),
	  X::vec(), StartX::integer(), IncX::integer(),
	  A::vec(), Stride::integer()) -> ok.
syr(Ord, Uplo, N, Alpha, X, StartX, IncX, A, Stride) ->
    trmv_impl(Ord, Uplo, no_transp, unit, N, A, Stride, X, StartX, IncX, 4, Alpha).

%% @doc Performs a rank-1 update of a symmetric matrix.
%%   alpha*x*x'+A = A
-spec spr(UpLo::matrix_uplo(), N::integer(), Alpha::float(), X::vec(), A::vec()) -> ok.
spr(UpLo, N, Alpha, X, A) ->
    spr(row_maj, UpLo, N, Alpha, X, 0, 1, A).
-spec spr(Order::matrix_order(), UpLo::matrix_uplo(),
	  N::integer(), Alpha::float(),
	  X::vec(), StartX::integer(), IncX::integer(),
	  A::vec()) -> ok.
spr(Ord, Uplo, N, Alpha, X, StartX, IncX, A) ->
    trmv_impl(Ord, Uplo, no_transp, unit, N, A, 0, X, StartX, IncX, 5, Alpha).

%% @doc Performs a rank-2 update of a symmetric matrix.
%%   alpha*x*y'+ alpha*y*x' + A = A
-spec syr2(Uplo::matrix_uplo(), N::integer(),
	   Alpha::float(), X::vec(), Y::vec(), A::vec()) -> ok.
syr2(UpLo, N, Alpha, X, Y, A) ->
    syr2(row_maj, UpLo, N, Alpha, X, 0, 1, Y, 0, 1, A, N).

-spec syr2(Order::matrix_order(),UpLo::matrix_uplo(),
	   N::integer(), Alpha::float(),
	   X::vec(), StartX::integer(), IncX::integer(),
	   Y::vec(), StartY::integer(), IncY::integer(),
	   A::vec(), Stride::integer()) -> ok.
syr2(Ord, UpLo, N, Alpha,
     X, StartX, IncX,
     Y, StartY, IncY,
     A, Stride) ->
    gemv_impl(Ord, UpLo, 0, N, Alpha, A, Stride,
	      X, StartX, IncX, 0.0, Y, StartY, IncY, 4).

%% @doc Performs a rank-2 update of a packed symmetric matrix.
%%   alpha*x*y'+ alpha*y*x' + A = A
-spec spr2(Uplo::matrix_uplo(), N::integer(),
	   Alpha::float(), X::vec(), Y::vec(), A::vec()) -> ok.
spr2(UpLo, N, Alpha, X, Y, A) ->
    spr2(row_maj, UpLo, N, Alpha, X, 0, 1, Y, 0, 1, A).

-spec spr2(Order::matrix_order(),UpLo::matrix_uplo(),
	   N::integer(), Alpha::float(),
	   X::vec(), StartX::integer(), IncX::integer(),
	   Y::vec(), StartY::integer(), IncY::integer(),
	   A::vec()) -> ok.
spr2(Ord, UpLo, N, Alpha,
     X, StartX, IncX,
     Y, StartY, IncY,
     A) ->
    gemv_impl(Ord, UpLo, 0, N, Alpha, A, 0,
	      X, StartX, IncX, 0.0, Y, StartY, IncY, 5).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

on_load() ->
    LibBaseName = "eblas",
    PrivDir = code:priv_dir(blas),
    Lib = filename:join([PrivDir, LibBaseName]),
    erlang:load_nif(Lib, {0.1}).
