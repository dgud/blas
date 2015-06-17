#include <stdio.h>
#include <string.h>
#include "erl_nif.h"
#include <atlas/cblas.h>

#define MAX(a,b) (((a)>(b))?(a):(b))

static ERL_NIF_TERM atom_ok;
static ERL_NIF_TERM atom_true;

static ERL_NIF_TERM atom_rowmaj;
static ERL_NIF_TERM atom_colmaj;

static ERL_NIF_TERM atom_notransp;
static ERL_NIF_TERM atom_transpose;
static ERL_NIF_TERM atom_conjugatet;

static ERL_NIF_TERM atom_upper;
static ERL_NIF_TERM atom_lower;

static ERL_NIF_TERM atom_left;
static ERL_NIF_TERM atom_right;

static ERL_NIF_TERM atom_nonunit;
static ERL_NIF_TERM atom_unit;

static ERL_NIF_TERM make_cont(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM from_list(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM to_values(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM to_idx_list(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM cont_size(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM cont_update(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM drotg(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM drot(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM dcopy(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM l1d_1cont(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM l1d_2cont(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM dgemv(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM dtrmv(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM dgemm(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);

static ErlNifFunc nif_funcs[] = {
    {"make_cont", 2, make_cont},
    {"from_list", 1, from_list},
    {"to_tuple_list_impl", 4, to_values},
    {"cont_size", 1, cont_size},
    {"values", 2, to_idx_list},
    {"update", 2, cont_update},
    {"update", 3, cont_update},

    {"rotg", 2, drotg},
    {"rot",  9, drot},
    {"copy", 4, dcopy},
    {"copy", 7, dcopy},
    {"one_vec", 6, l1d_1cont},
    {"two_vec", 9, l1d_2cont},

    {"gemv_impl", 15, dgemv},
    {"trmv_impl", 11, dtrmv},
    {"trmv_impl", 12, dtrmv},

    {"gemm_impl", 15, dgemm},
    {"trmm_impl", 15, dgemm},
};

static ErlNifResourceType *avec_r;

typedef struct {
    double v[1];
} Avec;

static ERL_NIF_TERM mk_avec(ErlNifEnv *env, unsigned int n, Avec **avec) {
    ERL_NIF_TERM term;
    *avec = enif_alloc_resource(avec_r, n*sizeof(double));
    term = enif_make_resource(env, *avec);
    enif_release_resource(*avec);
    return term;
}

#define AVEC_SIZE(AV) 	(enif_sizeof_resource(AV) / 8)

/* ---------------------------------------------------*/

static ERL_NIF_TERM make_cont(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    unsigned int n;
    ERL_NIF_TERM res;
    Avec *avec = NULL;
    double *data;

    if(!enif_get_uint(env, argv[0], &n)) return enif_make_badarg(env);
    res = mk_avec(env, n, &avec);

    if(enif_is_binary(env, argv[1])) {
	ErlNifBinary bin;
	enif_inspect_binary(env, argv[1], &bin);
	if(n*sizeof(double) != bin.size)
	    return enif_make_badarg(env);
	memcpy(avec->v, bin.data, sizeof(double)*n);
    } else if(enif_is_identical(argv[1], atom_true)) {
	data = avec->v;
	memset(data, 0, sizeof(double)*n);
    }

    return res;
}

static ERL_NIF_TERM from_list(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    unsigned int len;
    int dim, i,j, tmp;
    ERL_NIF_TERM hd, tail, res;
    const ERL_NIF_TERM *current;
    Avec *avec = NULL;
    double *data;

    if(!enif_get_list_length(env, argv[0], &len) || len == 0)
	return enif_make_badarg(env);

    enif_get_list_cell(env, argv[0], &hd, &tail);
    if(!enif_get_tuple(env, hd, &dim, &current))
	dim = 1;
    res = mk_avec(env, len*dim, &avec);

    enif_consume_timeslice(env, (len*dim)/1000);

    data = avec->v;
    if(dim == 1) {
	for(i=0; i<len; i++) {
	    if(!enif_get_double(env, hd, data))
		return enif_make_badarg(env);
	    data++;
	    enif_get_list_cell(env, tail, &hd, &tail);
	}
    } else {
	for(i=0; i<len; i++) {
	    if(!enif_get_tuple(env, hd, &tmp, &current) || tmp != dim)
		return enif_make_badarg(env);
	    for(j=0; j<dim; j++) {
		if(!enif_get_double(env, current[j], data))
		    return enif_make_badarg(env);
		data++;
	    }
	    enif_get_list_cell(env, tail, &hd, &tail);
	}
    }
    return res;
}

static ERL_NIF_TERM to_values(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    ERL_NIF_TERM tail, *tmp;
    Avec *avec = NULL;
    double *arr;
    unsigned int idx, n, dim, i, j, max;

    if(!enif_get_uint(env, argv[0], &idx)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[1], &n))   return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[2], &dim))  return enif_make_badarg(env);

    if(!enif_get_resource(env, argv[3], avec_r, (void **) &avec))
	return enif_make_badarg(env);
    max = AVEC_SIZE(avec);

    enif_consume_timeslice(env, max/1000);

    if(idx+n > max) return enif_make_badarg(env);

    if(n == 1 && dim == 0)  /* get_value() */
	return enif_make_double(env, avec->v[idx]);

    if(dim == 0) { /* to_list */
	tail = enif_make_list(env, 0);
	arr = avec->v + idx + n-1;
	for(i=0; i < max; i++) {
	    tail = enif_make_list_cell(env, enif_make_double(env, *arr), tail);
	    arr -= 1;
	}
	return tail;
    }
    if(dim == n) { /* to_tuple */
	tmp = (ERL_NIF_TERM*) malloc(sizeof(ERL_NIF_TERM)*n);
	arr = avec->v;
	for(i=0; i < n; i++) {
	    tmp[i] = enif_make_double(env, arr[idx+i]);
	}
	tail = enif_make_tuple_from_array(env, tmp, n);
	free(tmp);
	return tail;
    }

    /* list of tuples */
    if(n % dim != 0)
	return enif_make_badarg(env);

    arr = avec->v + idx + n-dim;
    tmp = (ERL_NIF_TERM*) malloc(sizeof(ERL_NIF_TERM)*dim);
    tail = enif_make_list(env, 0);

    for(i=0; i < max / dim; i++) {
	for(j=0; j < dim; j++, arr++) {
	    tmp[j] = enif_make_double(env, *arr);
	}
	tail = enif_make_list_cell(env, enif_make_tuple_from_array(env, tmp, dim), tail);
	arr -= 2*dim;
    }
    free(tmp);
    return tail;
}


static ERL_NIF_TERM to_idx_list(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    ERL_NIF_TERM hd, tail, *tmp;
    Avec *avec = NULL;
    double *arr;
    unsigned int idx, n, i;

    if(!enif_get_list_length(env, argv[0], &n) || n == 0)
	return enif_make_badarg(env);
    if(!enif_get_resource(env, argv[1], avec_r, (void **) &avec))
	return enif_make_badarg(env);

    enif_consume_timeslice(env, n/1000);
    tmp = (ERL_NIF_TERM*) malloc(sizeof(ERL_NIF_TERM)*n);
    arr = avec->v;
    tail = argv[0];
    for(i=0; i < n; i++) {
	enif_get_list_cell(env, tail, &hd, &tail);
	if(!enif_get_uint(env, hd, &idx) || idx >= AVEC_SIZE(avec))
	    return enif_make_badarg(env);
	tmp[i] = enif_make_tuple2(env, hd, enif_make_double(env, arr[idx]));
    }
    return enif_make_list_from_array(env, tmp, n);
}

static ERL_NIF_TERM cont_size(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    Avec *avec = NULL;

    if(!enif_get_resource(env, argv[0], avec_r, (void **) &avec)) {
	return enif_make_badarg(env);
    }

    return enif_make_uint(env, AVEC_SIZE(avec));
}

static ERL_NIF_TERM cont_update(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    ERL_NIF_TERM hd, tail;
    const ERL_NIF_TERM *curr;
    Avec *avec = NULL;
    double *arr, temp;
    int i=0, n=0;
    unsigned int idx, max;

    if(argc == 2) { /* tuple list of values */
	if(!enif_is_list(env, argv[0]))
	    return enif_make_badarg(env);
	if(!enif_get_resource(env, argv[1], avec_r, (void **) &avec))
	    return enif_make_badarg(env);
	arr = avec->v;
	max = AVEC_SIZE(avec);
	tail = argv[0];
	while(!enif_is_empty_list(env, tail)) {
	    if(!enif_get_list_cell(env, tail, &hd, &tail))
		return enif_make_badarg(env);
	    if(!enif_get_tuple(env, hd, &i, &curr) || i != 2)
		return enif_make_badarg(env);
	    if(!enif_get_uint(env, curr[0], &idx) || idx >= max)
		return enif_make_badarg(env);
	    if(!enif_get_double(env, curr[1], &temp))
		return enif_make_badarg(env);
	    arr[idx] = temp;
	    n++;
	}
	enif_consume_timeslice(env, n/1000);
	return atom_ok;
    }
    /* update(Idx, V|[Vs], Vec) */
    if(!enif_get_uint(env, argv[0], &idx))
	return enif_make_badarg(env);
    if(!enif_is_list(env, argv[1])) {
	/* update(Idx, Double, Vec) */
	if(!enif_get_double(env, argv[1], &temp))
	    return enif_make_badarg(env);
	if(!enif_get_resource(env, argv[2], avec_r, (void **) &avec))
	    return enif_make_badarg(env);
	max = AVEC_SIZE(avec);
	if(idx >= max)
	    return enif_make_badarg(env);
	arr = avec->v;
	arr[idx] = temp;
	return atom_ok;
    }
    /* update(Idx, [Values], Vec) */
    tail = argv[1];
    if(!enif_get_resource(env, argv[2], avec_r, (void **) &avec))
	return enif_make_badarg(env);
    max = AVEC_SIZE(avec);
    arr = avec->v+idx;
    while(!enif_is_empty_list(env, tail)) {
	if(!enif_get_list_cell(env, tail, &hd, &tail))
	    return enif_make_badarg(env);
	if(!enif_get_double(env, hd, &temp))
	    return enif_make_badarg(env);
	*arr = temp;
	arr++;
	idx++;
	n++;
	if(idx > max)
	    return enif_make_badarg(env);
    }
    enif_consume_timeslice(env, n/1000);
    return atom_ok;
}



/* ---------------------------------------------------*/

static ERL_NIF_TERM drotg(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{/* (a, b) */
    double a,b,c,s;

    if(!enif_get_double(env, argv[0], &a) || !enif_get_double(env, argv[1], &b)) {
	return enif_make_badarg(env);
    }

    cblas_drotg(&a,&b,&c,&s);
    return enif_make_tuple4(env,
			    enif_make_double(env, a), enif_make_double(env, b),
			    enif_make_double(env, c), enif_make_double(env, s));

}

static ERL_NIF_TERM drot(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    unsigned int n, xs, ys;
    int xi, yi;
    Avec *ax, *ay;
    double *x, *y, c, s;

    if(!enif_get_uint(env, argv[0], &n)) return enif_make_badarg(env);
    if(!enif_get_resource(env, argv[1], avec_r, (void **) &ax)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[2], &xs)) return enif_make_badarg(env);
    if(!enif_get_int(env, argv[3], &xi)) return enif_make_badarg(env);
    if(!enif_get_resource(env, argv[4], avec_r, (void **) &ay)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[5], &ys)) return enif_make_badarg(env);
    if(!enif_get_int(env, argv[6], &yi)) return enif_make_badarg(env);
    if(!enif_get_double(env, argv[7], &c)) return enif_make_badarg(env);
    if(!enif_get_double(env, argv[8], &s)) return enif_make_badarg(env);

    x = ax->v + xs;
    y = ay->v + ys;
    /* array limit checks */
    if((xs+n*abs(xi)-1) > AVEC_SIZE(ax)) return enif_make_badarg(env);
    if((ys+n*abs(yi)-1) > AVEC_SIZE(ax)) return enif_make_badarg(env);

    enif_consume_timeslice(env, n/1000);
    cblas_drot(n, x, xi, y, xi, c, s);

    return atom_ok;
}

static ERL_NIF_TERM l1d_1cont(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    unsigned int op, n, xs;
    int xi, tmp;
    Avec *ax;
    double *x, alpha;
    ERL_NIF_TERM res = atom_ok;

    if(!enif_get_uint(env, argv[0], &n)) return enif_make_badarg(env);
    if(!enif_get_double(env, argv[1], &alpha)) return enif_make_badarg(env);
    if(!enif_get_resource(env, argv[2], avec_r, (void **) &ax)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[3], &xs)) return enif_make_badarg(env);
    if(!enif_get_int(env, argv[4], &xi)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[5], &op)) return enif_make_badarg(env);
    x = ax->v + xs;
    /* array limit checks */
    if((xs+n*abs(xi)-1) > AVEC_SIZE(ax)) return enif_make_badarg(env);

    switch(op) {
    case 0:
	cblas_dscal(n, alpha, x, xi);
	break;
    case 1:
	res = enif_make_double(env, cblas_dnrm2(n, x, xi));
	break;
    case 2:
	res = enif_make_double(env, cblas_dasum(n, x, xi));
	break;
    case 3:
	tmp = cblas_idamax(n, x, xi);
	res = enif_make_tuple2(env,
			       enif_make_uint(env, tmp),
			       enif_make_double(env, x[tmp]));
	break;
    }
    enif_consume_timeslice(env, n/1000);
    return res;
}


static ERL_NIF_TERM dcopy(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    unsigned int n, xs, ys = 0;
    int xi, yi=1;
    Avec *ax, *ay;
    double *x, *y;
    ERL_NIF_TERM res;

    if(!enif_get_uint(env, argv[0], &n)) return enif_make_badarg(env);
    if(!enif_get_resource(env, argv[1], avec_r, (void **) &ax)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[2], &xs)) return enif_make_badarg(env);
    if(!enif_get_int(env, argv[3], &xi)) return enif_make_badarg(env);
    /* array limit checks */
    if((xs+n*abs(xi)-1) > AVEC_SIZE(ax)) return enif_make_badarg(env);

    if(argc == 4) {  /* copy to new vector */
	res = mk_avec(env, n, &ay);
    } else {
	if(!enif_get_resource(env, argv[4], avec_r, (void **) &ay)) return enif_make_badarg(env);
	if(!enif_get_uint(env, argv[5], &ys)) return enif_make_badarg(env);
	if(!enif_get_int(env, argv[6], &yi)) return enif_make_badarg(env);
	if((ys+n*abs(yi)-1) > AVEC_SIZE(ay)) return enif_make_badarg(env);
	res = atom_ok;
    }

    x = ax->v + xs;
    y = ay->v + ys;
    cblas_dcopy(n, x, xi, y, yi);
    enif_consume_timeslice(env, n/1000);

    return res;
}

static ERL_NIF_TERM l1d_2cont(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    unsigned int op, n, xs, ys;
    int xi, yi;
    Avec *ax, *ay;
    double *x, *y, alpha, res;

    if(!enif_get_uint(env, argv[0], &n)) return enif_make_badarg(env);
    if(!enif_get_double(env, argv[1], &alpha)) return enif_make_badarg(env);
    if(!enif_get_resource(env, argv[2], avec_r, (void **) &ax)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[3], &xs)) return enif_make_badarg(env);
    if(!enif_get_int(env, argv[4], &xi)) return enif_make_badarg(env);
    if(!enif_get_resource(env, argv[5], avec_r, (void **) &ay)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[6], &ys)) return enif_make_badarg(env);
    if(!enif_get_int(env, argv[7], &yi)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[8], &op)) return enif_make_badarg(env);
    x = ax->v + xs;
    y = ay->v + ys;
    /* array limit checks */
    if((xs+n*abs(xi)-1) > AVEC_SIZE(ax)) return enif_make_badarg(env);
    if((ys+n*abs(yi)-1) > AVEC_SIZE(ay)) return enif_make_badarg(env);

    enif_consume_timeslice(env, n/1000);
    switch(op) {
    case 0:
	cblas_daxpy(n, alpha, x, xi, y, yi);
	break;
    case 1:
	cblas_dswap(n, x, xi, y, yi);
	break;
    case 2:
	res = cblas_ddot(n, x, xi, y, yi);
	return enif_make_double(env, res);
    }
    return atom_ok;
}

/* ---------------------------------------------------*/

static ERL_NIF_TERM dgemv(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    unsigned int m, n, lda, xs, ys, op;
    int xi, yi;
    Avec *aa, *ax, *ay;
    double *a, *x, *y, alpha, beta;
    enum CBLAS_ORDER order = 0;
    enum CBLAS_TRANSPOSE trans = 0;
    enum CBLAS_UPLO uplo = 0;

    if(enif_is_identical(argv[0], atom_rowmaj)) order = CblasRowMajor;
    else if(enif_is_identical(argv[0], atom_colmaj)) order = CblasColMajor;
    else return enif_make_badarg(env);

    if(enif_is_identical(argv[1], atom_notransp)) trans = CblasNoTrans;
    else if(enif_is_identical(argv[1], atom_transpose)) trans = CblasTrans;
    else if(enif_is_identical(argv[1], atom_conjugatet)) trans = CblasConjTrans;
    else if(enif_is_identical(argv[1], atom_lower)) uplo = CblasLower;
    else if(enif_is_identical(argv[1], atom_upper)) uplo = CblasUpper;
    else return enif_make_badarg(env);

    if(!enif_get_uint(env, argv[2], &m)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[3], &n)) return enif_make_badarg(env);
    if(!enif_get_double(env, argv[4], &alpha)) return enif_make_badarg(env);
    if(!enif_get_resource(env, argv[5], avec_r, (void **) &aa)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[6], &lda)) return enif_make_badarg(env);
    if(!enif_get_resource(env, argv[7], avec_r, (void **) &ax)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[8], &xs)) return enif_make_badarg(env);
    if(!enif_get_int(env, argv[9], &xi)) return enif_make_badarg(env);
    if(!enif_get_double(env, argv[10], &beta)) return enif_make_badarg(env);
    if(!enif_get_resource(env, argv[11], avec_r, (void **) &ay)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[12], &ys)) return enif_make_badarg(env);
    if(!enif_get_int(env, argv[13], &yi)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[14], &op)) return enif_make_badarg(env);

    a = aa->v;
    x = ax->v + xs;
    y = ay->v + ys;

    /* array limit checks */
    if((xs+n*abs(xi)-1) > AVEC_SIZE(ax)) return enif_make_badarg(env);
    if((ys+n*abs(yi)-1) > AVEC_SIZE(ay)) return enif_make_badarg(env);

    switch(op) {
    case 0:
	cblas_dgemv(order, trans, m, n, alpha, a, lda, x, xi, beta, y, yi); break;
    case 1:
	cblas_dsymv(order, uplo, n, alpha, a, lda, x, xi, beta, y, yi); break;
    case 2:
	cblas_dspmv(order, uplo, n, alpha, a, x, xi, beta, y, yi); break;
    case 3:
	cblas_dger(order, m, n, alpha, x, xi, y, yi, a, lda); break;
    case 4:
	cblas_dsyr2(order, uplo, n, alpha, x, xi, y, yi, a, lda); break;
    case 5:
	cblas_dspr2(order, uplo, n, alpha, x, xi, y, yi, a); break;
    }
    enif_consume_timeslice(env, MAX(1,m)*n/1000);
    return atom_ok;
}

static ERL_NIF_TERM dtrmv(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    unsigned int n, lda, xs, op;
    int xi;
    Avec *aa, *ax;
    double *a, *x, alpha;
    enum CBLAS_ORDER order = 0;
    enum CBLAS_TRANSPOSE trans = 0;
    enum CBLAS_UPLO uplo = 0;
    enum CBLAS_DIAG diag = 0;

    if(enif_is_identical(argv[0], atom_rowmaj)) order = CblasRowMajor;
    else if(enif_is_identical(argv[0], atom_colmaj)) order = CblasColMajor;
    else return enif_make_badarg(env);

    if(enif_is_identical(argv[1], atom_lower)) uplo = CblasLower;
    else if(enif_is_identical(argv[1], atom_upper)) uplo = CblasUpper;
    else return enif_make_badarg(env);

    if(!enif_get_uint(env, argv[10], &op)) return enif_make_badarg(env);
    if(op < 4) {
	if(enif_is_identical(argv[2], atom_notransp)) trans = CblasNoTrans;
	else if(enif_is_identical(argv[2], atom_transpose)) trans = CblasTrans;
	else if(enif_is_identical(argv[2], atom_conjugatet)) trans = CblasConjTrans;
	else return enif_make_badarg(env);

	if(enif_is_identical(argv[3], atom_nonunit)) diag = CblasNonUnit;
	else if(enif_is_identical(argv[3], atom_unit)) diag = CblasUnit;
	else return enif_make_badarg(env);
    }

    if(!enif_get_uint(env, argv[4], &n)) return enif_make_badarg(env);
    if(!enif_get_resource(env, argv[5], avec_r, (void **) &aa)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[6], &lda)) return enif_make_badarg(env);
    if(!enif_get_resource(env, argv[7], avec_r, (void **) &ax)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[8], &xs)) return enif_make_badarg(env);
    if(!enif_get_int(env, argv[9], &xi)) return enif_make_badarg(env);
    if(argc > 11)
	if(!enif_get_double(env, argv[11], &alpha)) return enif_make_badarg(env);

    a = aa->v;
    x = ax->v + xs;

    /* array limit checks */
    if((xs+n*abs(xi)-1) > AVEC_SIZE(ax)) return enif_make_badarg(env);

    switch(op) {
    case 0:
	cblas_dtrmv(order, uplo, trans, diag, n, a, lda, x, xi);break;
    case 1:
	cblas_dtpmv(order, uplo, trans, diag, n, a, x, xi); break;
    case 2:
	cblas_dtrsv(order, uplo, trans, diag, n, a, lda, x, xi);break;
    case 3:
	cblas_dtpsv(order, uplo, trans, diag, n, a, x, xi);break;
    case 4:
	cblas_dsyr(order, uplo, n, alpha, x, xi, a, lda); break;
    case 5:
	cblas_dspr(order, uplo, n, alpha, x, xi, a); break;
    }

    enif_consume_timeslice(env, n*n/1000);
    return atom_ok;
}

/* ---------------------------------------------------*/

static ERL_NIF_TERM dgemm(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    unsigned int m, n, k, lda, ldb, ldc, op;
    Avec *aa, *ab, *ac;
    double alpha, beta, *a, *b, *c = NULL;
    enum CBLAS_ORDER order = 0;
    enum CBLAS_TRANSPOSE ta = 0, tb = 0;
    enum CBLAS_UPLO uplo = 0;
    enum CBLAS_SIDE side = 0;
    enum CBLAS_DIAG diag = 0;

    if(enif_is_identical(argv[0], atom_rowmaj)) order = CblasRowMajor;
    else if(enif_is_identical(argv[0], atom_colmaj)) order = CblasColMajor;
    else return enif_make_badarg(env);

    if(!enif_get_uint(env, argv[3], &m)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[4], &n)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[5], &k)) return enif_make_badarg(env);

    if(!enif_get_double(env, argv[6], &alpha)) return enif_make_badarg(env);

    if(!enif_get_resource(env, argv[7], avec_r, (void **) &aa)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[8], &lda)) return enif_make_badarg(env);

    if(!enif_get_resource(env, argv[9], avec_r, (void **) &ab)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[10], &ldb)) return enif_make_badarg(env);

    if(!enif_get_double(env, argv[11], &beta)) return enif_make_badarg(env);

    if(!enif_get_uint(env, argv[12], &op)) return enif_make_badarg(env);
    if(op < 3) {
	if(!enif_get_resource(env, argv[13], avec_r, (void **) &ac))
	    return enif_make_badarg(env);
	if(!enif_get_uint(env, argv[14], &ldc)) return enif_make_badarg(env);
	c = ac->v;
    }
    a = aa->v;
    b = ab->v;
    switch(op) {
    case 0: {
	if(enif_is_identical(argv[1], atom_notransp)) ta = CblasNoTrans;
	else if(enif_is_identical(argv[1], atom_transpose)) ta = CblasTrans;
	else if(enif_is_identical(argv[1], atom_conjugatet)) ta = CblasConjTrans;
	else return enif_make_badarg(env);

	if(enif_is_identical(argv[2], atom_notransp)) tb = CblasNoTrans;
	else if(enif_is_identical(argv[2], atom_transpose)) tb = CblasTrans;
	else if(enif_is_identical(argv[2], atom_conjugatet)) tb = CblasConjTrans;
	else return enif_make_badarg(env);

	cblas_dgemm(order, ta,tb, m,n,k, alpha, a,lda, b,ldb, beta, c,ldc);
	break;}
    case 1: {
	if(enif_is_identical(argv[1], atom_left)) side = CblasLeft;
	else if(enif_is_identical(argv[1], atom_right)) side = CblasRight;
	else return enif_make_badarg(env);

	if(enif_is_identical(argv[2], atom_lower)) uplo = CblasLower;
	else if(enif_is_identical(argv[2], atom_upper)) uplo = CblasUpper;
	else return enif_make_badarg(env);

	cblas_dsymm(order, side, uplo, m,n, alpha, a,lda, b,ldb, beta, c,ldc);
	break; }
    case 2: {
	if(enif_is_identical(argv[1], atom_lower)) uplo = CblasLower;
	else if(enif_is_identical(argv[1], atom_upper)) uplo = CblasUpper;
	else return enif_make_badarg(env);

	if(enif_is_identical(argv[2], atom_notransp)) tb = CblasNoTrans;
	else if(enif_is_identical(argv[2], atom_transpose)) tb = CblasTrans;
	else if(enif_is_identical(argv[2], atom_conjugatet)) tb = CblasConjTrans;

	// fprintf(stderr, "%d %d,%d,%d %.2f %.2f %d,%d,%d\r\n", __LINE__, m,n,k, alpha, beta, lda,ldb,ldc);
	cblas_dsyr2k(order, uplo, tb, n, k, alpha, a,lda, b,ldb, beta, c,ldc);
	break; }
    case 3: {
	if(enif_is_identical(argv[1], atom_lower)) uplo = CblasLower;
	else if(enif_is_identical(argv[1], atom_upper)) uplo = CblasUpper;
	else return enif_make_badarg(env);

	if(enif_is_identical(argv[2], atom_notransp)) tb = CblasNoTrans;
	else if(enif_is_identical(argv[2], atom_transpose)) tb = CblasTrans;
	else if(enif_is_identical(argv[2], atom_conjugatet)) tb = CblasConjTrans;

	if(enif_is_identical(argv[13], atom_left)) side = CblasLeft;
	else if(enif_is_identical(argv[13], atom_right)) side = CblasRight;
	else return enif_make_badarg(env);

	if(enif_is_identical(argv[14], atom_nonunit)) diag = CblasNonUnit;
	else if(enif_is_identical(argv[14], atom_unit)) diag = CblasUnit;
	else return enif_make_badarg(env);

	cblas_dtrmm(order,side,uplo,tb,diag, m,n, alpha, a,lda, b, ldb);
	break; }
    case 4: {
	if(enif_is_identical(argv[1], atom_lower)) uplo = CblasLower;
	else if(enif_is_identical(argv[1], atom_upper)) uplo = CblasUpper;
	else return enif_make_badarg(env);

	if(enif_is_identical(argv[2], atom_notransp)) tb = CblasNoTrans;
	else if(enif_is_identical(argv[2], atom_transpose)) tb = CblasTrans;
	else if(enif_is_identical(argv[2], atom_conjugatet)) tb = CblasConjTrans;

	if(enif_is_identical(argv[13], atom_left)) side = CblasLeft;
	else if(enif_is_identical(argv[13], atom_right)) side = CblasRight;
	else return enif_make_badarg(env);

	if(enif_is_identical(argv[14], atom_nonunit)) diag = CblasNonUnit;
	else if(enif_is_identical(argv[14], atom_unit)) diag = CblasUnit;
	else return enif_make_badarg(env);

	cblas_dtrsm(order,side,uplo,tb,diag, m,n, alpha, a,lda, b, ldb);
	break; }
    case 5: {
	if(enif_is_identical(argv[1], atom_lower)) uplo = CblasLower;
	else if(enif_is_identical(argv[1], atom_upper)) uplo = CblasUpper;
	else return enif_make_badarg(env);

	if(enif_is_identical(argv[2], atom_notransp)) tb = CblasNoTrans;
	else if(enif_is_identical(argv[2], atom_transpose)) tb = CblasTrans;
	else if(enif_is_identical(argv[2], atom_conjugatet)) tb = CblasConjTrans;

	cblas_dsyrk(order,uplo,tb, n,k, alpha, a,lda, beta, b,ldb);
	break; }
    }

    enif_consume_timeslice(env, MAX(m,k)*n/1000);
    return atom_ok;
}

/* ---------------------------------------------------*/

static int load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info)
{
    atom_ok = enif_make_atom(env,"ok");
    atom_true = enif_make_atom(env,"true");

    atom_notransp = enif_make_atom(env,"no_transp");
    atom_transpose = enif_make_atom(env,"transp");
    atom_conjugatet = enif_make_atom(env,"conj_transp");

    atom_rowmaj = enif_make_atom(env,"row_maj");
    atom_colmaj = enif_make_atom(env,"col_maj");

    atom_upper = enif_make_atom(env,"upper");
    atom_lower = enif_make_atom(env,"lower");

    atom_left = enif_make_atom(env,"left");
    atom_right = enif_make_atom(env,"right");

    atom_nonunit = enif_make_atom(env,"non_unit");
    atom_unit = enif_make_atom(env,"unit");

    avec_r  = enif_open_resource_type(env, "eblas", "avec", NULL, ERL_NIF_RT_CREATE, NULL);
    return 0;
}

static int upgrade(ErlNifEnv* env, void** priv_data, void** old_priv_data,
		   ERL_NIF_TERM load_info)
{
    return 0;
}

static void unload(ErlNifEnv* env, void* priv_data)
{

}

ERL_NIF_INIT(blasd_raw,nif_funcs,load,NULL,upgrade,unload)
