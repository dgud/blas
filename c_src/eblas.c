#include <stdio.h>
#include "erl_nif.h"
#include <atlas/cblas.h>

static ERL_NIF_TERM atom_ok;

static ERL_NIF_TERM from_list(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM to_tuple_list(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM size(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM drotg(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM drot(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM dcopy(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM l1d_1vec(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM l1d_2vec(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);

static ErlNifFunc nif_funcs[] = {
    {"from_list", 1, from_list},
    {"to_tuple_list", 2, to_tuple_list},
    {"size", 1, size},

    {"rotg", 2, drotg},
    {"rot",  9, drot},
    {"copy", 4, dcopy},
    {"one_vec", 6, l1d_1vec},
    {"two_vec", 9, l1d_2vec},

};

static ErlNifResourceType *avec_r;

typedef struct {
    unsigned int n;
    unsigned int dim; /* unused alignment only 64b*/
    double v[];
} Avec;

static ERL_NIF_TERM mk_avec(ErlNifEnv *env, unsigned int n, Avec **avec) {
    ERL_NIF_TERM term;
    *avec = enif_alloc_resource(avec_r, sizeof(avec) + n*sizeof(double));
    term = enif_make_resource(env, *avec);
    enif_release_resource(*avec);
    (*avec)->n = n;
    return term;
}

/* ---------------------------------------------------*/

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

static ERL_NIF_TERM to_tuple_list(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    ERL_NIF_TERM tail, *tmp;
    Avec *avec = NULL;
    int i, j;
    unsigned int dim;
    double *ptr;

    if(!enif_get_uint(env, argv[0], &dim)) return enif_make_badarg(env);
    if(!enif_get_resource(env, argv[1], avec_r, (void **) &avec)) {
	return enif_make_badarg(env);
    }
    if(avec->n % dim != 0) return enif_make_badarg(env);

    tail = enif_make_list(env, 0);
    ptr = avec->v + avec->n-dim;
    if(dim == 1) {
	for(i=0; i < avec->n; i++) {
	    tail = enif_make_list_cell(env, enif_make_double(env, *ptr), tail);
	    ptr -= 1;
	}
    } else {
	tmp = (ERL_NIF_TERM*) malloc(sizeof(ERL_NIF_TERM)*dim);
	for(i=0; i < avec->n / dim; i++) {
	    for(j=0; j < dim; j++, ptr++) {
		tmp[j] = enif_make_double(env, *ptr);
	    }
	    tail = enif_make_list_cell(env, enif_make_tuple_from_array(env, tmp, dim), tail);
	    ptr -= 2*dim;
	}
	free(tmp);
    }
    return tail;
}

static ERL_NIF_TERM size(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    Avec *avec = NULL;

    if(!enif_get_resource(env, argv[0], avec_r, (void **) &avec)) {
	return enif_make_badarg(env);
    }

    return enif_make_uint(env, avec->n);
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
    unsigned int n, xs, xi, ys, yi;
    Avec *ax, *ay;
    double *x, *y, c, s;

    if(!enif_get_uint(env, argv[0], &n)) return enif_make_badarg(env);
    if(!enif_get_resource(env, argv[1], avec_r, (void **) &ax)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[2], &xs)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[3], &xi)) return enif_make_badarg(env);
    if(!enif_get_resource(env, argv[4], avec_r, (void **) &ay)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[5], &ys)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[6], &yi)) return enif_make_badarg(env);
    if(!enif_get_double(env, argv[7], &c)) return enif_make_badarg(env);
    if(!enif_get_double(env, argv[8], &s)) return enif_make_badarg(env);

    x = ax->v + xs;
    y = ay->v + ys;
    /* array limit checks */
    if((xs+n*xi-1) > (ax->n)) return enif_make_badarg(env);
    if((ys+n*yi-1) > (ay->n)) return enif_make_badarg(env);

    cblas_drot(n, x, xi, y, xi, c, s);

    return atom_ok;
}

static ERL_NIF_TERM l1d_1vec(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
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
    if((xs+n*xi-1) > ax->n) return enif_make_badarg(env);

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
    return res;
}


static ERL_NIF_TERM dcopy(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    unsigned int i, n, xs, xi;
    Avec *ax, *resv;
    double *x, *y;
    ERL_NIF_TERM res;

    if(!enif_get_uint(env, argv[0], &n)) return enif_make_badarg(env);
    if(!enif_get_resource(env, argv[1], avec_r, (void **) &ax)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[2], &xs)) return enif_make_badarg(env);
    if(!enif_get_uint(env, argv[3], &xi)) return enif_make_badarg(env);

    x = ax->v + xs;
    /* array limit checks */
    if((xs+n*xi-1) > ax->n) return enif_make_badarg(env);

    res = mk_avec(env, n, &resv);
    y = resv->v;

    for(i=0; i < n; i++) {
	*y = *x;
	y += 1;
	x += xi;
    }

    return res;
}

static ERL_NIF_TERM l1d_2vec(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
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
    if((xs+n*xi-1) > ax->n) return enif_make_badarg(env);
    if((ys+n*yi-1) > ay->n) return enif_make_badarg(env);

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

/* static ERL_NIF_TERM dgemv(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) */
/* { */
/*     unsigned int n, xs, xi, ys, yi; */
/*     Avec *aa, *ax, *ay; */
/*     double *a, *x, *y, alpha, beta; */
/*     char trans; */

/*     if(!enif_get_atom(env, argv[0], &op)) return enif_make_badarg(env); */
/*     if(!enif_get_uint(env, argv[0], &m)) return enif_make_badarg(env); */
/*     if(!enif_get_uint(env, argv[0], &n)) return enif_make_badarg(env); */
/*     if(!enif_get_double(env, argv[1], &alpha)) return enif_make_badarg(env); */
/*     if(!enif_get_resource(env, argv[2], avec_r, (void **) &aa)) return enif_make_badarg(env); */
/*     if(!enif_get_uint(env, argv[0], &lda)) return enif_make_badarg(env); */
/*     if(!enif_get_resource(env, argv[2], avec_r, (void **) &ax)) return enif_make_badarg(env); */
/*     if(!enif_get_uint(env, argv[3], &xs)) return enif_make_badarg(env); */
/*     if(!enif_get_uint(env, argv[4], &xi)) return enif_make_badarg(env); */
/*     if(!enif_get_double(env, argv[1], &beta)) return enif_make_badarg(env); */
/*     if(!enif_get_resource(env, argv[5], avec_r, (void **) &ay)) return enif_make_badarg(env); */
/*     if(!enif_get_uint(env, argv[6], &ys)) return enif_make_badarg(env); */
/*     if(!enif_get_uint(env, argv[7], &yi)) return enif_make_badarg(env); */
/*     x = ax->v + xs; */
/*     y = ay->v + ys; */

/*     if(op) ... */

/*     /\* array limit checks *\/ */
/*     if((xs+n*xi-1) > ax->n) return enif_make_badarg(env); */
/*     if((ys+n*yi-1) > ay->n) return enif_make_badarg(env); */

/*     cblas_dgemv(trans, m, n, alpha, a, lda, x, xi, beta, y, yi); */

/*     return atom_ok; */
/* } */


/* ---------------------------------------------------*/

static int load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info)
{
    atom_ok = enif_make_atom(env,"ok");
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

ERL_NIF_INIT(blasd,nif_funcs,load,NULL,upgrade,unload)
