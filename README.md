blas
=====

This project is the continuation of ddgud's BLAS wrapper. It features scheduling control (execution in dirty/clean nifs), as well as type checking, and array overflow detection (to avoid sigsev related crashes).


Usage
-----

sasum example:

```erlang
Bin = chain:ltb(s, [1,2,4,3]),  
% ltb(T, L):
% transforms list L into a binary,
% using encoding T: s->single, d->double,
%                   c->single complex
%                   z->single double

Ptr = blas:new(Bin),           
% new(Binary):
% creates a mutable c binary, of initial
% content Bin.

Sum = blas:run({sasum, 4, Ptr, 1 }).
% run({Blas_fct_name, arg0, arg1,...}).
% Executes blas Blas_fct_name with given arguments.
```

caxpy example:

```erlang
% List representation of X,Y,A
[X,Y,A]   =  [[1,2,1,3,1,4],[3,2,3,4,3,6],[1,0]],
% Binary representation of X,Y,A
[Xb,Yb,Ab] = lists:map(fun(L)->chain:ltb(c, L) end, [X,Y,A]),
% C_Binary representation of X,Y,A
[Xc,Yc,Ac] = lists:map(fun blas:new/1, [Xb,Yb,Ab]),

ok    = blas:run({caxpy, 3, Ac, Xc, 1, Yc, 1}),

% Reset Yc
blas:copy(Yb, Yc),
ok    = blas:run({caxpy, 3, Ab, Xb, 1, Yc, 1}).
```

Available
----

_axpy,_copy,_swap,_scal,_dot,_nrm2,_asum,i_amax,_rot,_rotg,_rotm,_rotmg

TODO
----

MISSING LEVEL 1  
amin,  

LEVEL2 TODO  
sgbmv, dgbmv, cgbmv, zgbmv,
ssbmv,dsbmv,
stbmv, dtbmv, ctbmv,ztbmv,
stbsv, dtbsv, ctbsv, ztbsv,
stpmv, dtpmv, ctpmv, ztpmv,
stpsv, dtpsv, ctpsv, ztpsv,
ssymv, dymv,
chemv, zhemv,
sspmv, dspmv,
sspr, dspr,
chpr, zhpr,
sspr2, dspr2,
chpr2, zhpr2,
chbmv, zhbmv,
chpmv, zhpmv,
sgemv, dgemv, cgemv, zgemv,

sger, dger, cgeru, cgerc, zgeru, zgerc,
cher, zher, cher2, zher2, 
strmv, dtrmv, ctrmv, ztrmv,
strsv, dtrsv, ctsv, ztrsv  


LEVEL 3 TODO  
sgemm, dgemm, cgemm, cgemm3m, zgemm, zgemm3m,
chemm, zhemm,
cherk, zherk, cher2k, zher2k,
ssymm, dsymm, csymm, zsymm,

ssyrk, dsyrk, csyrk, zsyrk,
ssyr2k, dsyr2k, csyr2k, zsyr2k,
strmm, dtrmm, ctrmm, ztrmm,
strsm, dtrsm, ctrsm, ztrsm  