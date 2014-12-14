%%%-------------------------------------------------------------------
%%% @author Dan Gudmundsson <dgud@erlang.org>
%%% @copyright (C) 2014, Dan Gudmundsson
%%% @doc  A wrapper for the defacto standard BLAS library
%%%
%%%       Most of these functions use desctructive operations.
%%%
%%% @end
%%% Created :  9 Dec 2014 by Dan Gudmundsson <dgud@erlang.org>
%%%-------------------------------------------------------------------
-module(blasd).

-export([from_list/1, to_list/1, to_tuple_list/2, size/1]).
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
	  iamax/1, iamax/4
	  %% Level 2
	  %%dgemv/8, dgemv/13
	]).

-compile({no_auto_import,[size/1]}).
-opaque(vec).
-type vec() :: binary().
%-type matrix_op() :: normal|transpose|conjugate.

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

-spec size(Vec::vec()) -> N::integer().
size(_Vec) -> ?nif_stub.

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
    N = size(XY),
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
    N = size(X),
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
    N = size(X),
    copy(N, X, 0, 1).

%% @doc Copy values in the vector
%% Start at StartX pos and step Xinc
-spec copy(N::integer(), X::vec(), XStart::integer(), Xinc::integer()) -> vec().
copy(_N, _X, _XStart, _Xinc) -> ?nif_stub.

%% @doc Compute AX+Y=Y
-spec axpy(A::float(), X::vec(), Y::vec()) -> ok.
axpy(A, X, Y) ->
    N = size(X),
    N =:= size(Y) orelse error(badarg),
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
    N = size(X),
    N =:= size(Y) orelse error(badarg),
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
    N = size(X),
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
    N = size(X),
    asum(N, X, 0, 1).

-spec asum(N::integer(),
	    X::vec(), XStart::integer(), Xinc::integer()) -> float().
asum(N, X, XStart, Xinc) ->
    one_vec(N, 0.0, X, XStart, Xinc, 2).

%% @doc Locates the max value
%%  (index from Xstart)
-spec iamax(X::vec()) -> {Index::integer(), Max::float()}.
iamax(X) ->
    N = size(X),
    iamax(N, X, 0, 1).

-spec iamax(N::integer(),
	     X::vec(), XStart::integer(), Xinc::integer()) ->
		    {Index::integer(), Max::float()}.
iamax(N, X, XStart, Xinc) ->
    one_vec(N, 0.0, X, XStart, Xinc, 3).

one_vec(_N, _Alpha, _X, _XStart, _Xinc, _Op) -> ?nif_stub.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Level 2


%% @doc Perform a M*N matrix vector operation
%%   normal:     Alpha*A*x+Beta*Y = Y
%%   transpose:  Alpha*A'*x+Beta*Y = Y
%%   conjugate:  Alpha*conjug(A')*x+Beta*Y = Y
%% where:
%%   Trans is one of normal, transpose, conjugate
%%   M is the number of rows
%%   N is the number of columns
%%   A is an Lda*N matrix
%%   Lda On entry, LDA specifies the first dimension of A as declared
%%       in the calling program. Lda must be at least max( 1, m )
%%   X and Y are vectors fo length ( 1 + ( N - 1 )*abs( Incx ) ) when Trans is normal
%%           and otherwise ( 1 + ( M - 1 )*abs( Incx ) )
%% -spec gemv(Trans::matrix_op(),
%% 	    M::integer(), N::integer(),
%% 	    Alpha::float(), A::vec(),
%% 	    X::vec(), Beta::float(), Y::vec()) -> ok.
%% gemv(Trans, M, N, Alpha, A, X, Beta, Y) ->
%%     gemv(Trans, M, N, Alpha, A, M, X, 0, 1, Beta, Y, 0, 1).
%% -spec gemv(Trans::matrix_op(),
%% 	    M::integer(), N::integer(),
%% 	    Alpha::float(), A::vec(), Lda::integer(),
%% 	    X::vec(), StartX::integer(), IncX::integer(),
%% 	    Beta::float(),
%% 	    Y::vec(), StartY::integer(), IncY::integer()) -> ok.
%% gemv(_Trans, _M, _N, _Alpha, _A, _Lda, _X, _StartX, _IncX, _Beta, _Y, _StartY, _IncY) -> ?nif_stub.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

on_load() ->
    LibBaseName = "eblas",
    PrivDir = code:priv_dir(blas),
    Lib = filename:join([PrivDir, LibBaseName]),
    erlang:load_nif(Lib, {0.1}).
