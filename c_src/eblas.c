#include "eblas.h"
#include "string.h"
#include <cblas.h>
#include <complex.h>
#include "string.h"

//Various utility functions
//--------------------------------

int translate(ErlNifEnv* env, const ERL_NIF_TERM* terms, const etypes* format, ...){
    va_list valist;
    va_start(valist, format);
    int valid = 1;
    int* i_dest;

    for(int curr=0; format[curr] != e_end; curr++){
        debug_write("Unwrapping term number %i\n", curr);
        switch(format[curr]){
            case e_int:
                valid = enif_get_int(env, terms[curr], va_arg(valist, int*));
            break;
            case e_uint:
                i_dest = va_arg(valist, int*);
                valid = enif_get_int(env, terms[curr], i_dest) && *i_dest >=0;
            break;
            
            case e_float:
                double val;
                float* dest = va_arg(valist, float*);
                valid = enif_get_double(env, terms[curr], &val);
                if(valid)
                    *dest = (float) val;    
            break;
            case e_double:
                valid = enif_get_double(env, terms[curr], va_arg(valist, double*));
            break;

            case e_ptr:
                valid = get_c_binary(env, terms[curr], va_arg(valist, c_binary*));
            break;
            case e_cste_ptr:
                valid = get_cste_binary(env, terms[curr], va_arg(valist, cste_c_binary*));
            break;

            case e_layout:
                i_dest = va_arg(valist, int*);
                if      (enif_is_identical(terms[curr], atomRowMajor)) *i_dest = CblasRowMajor;
                else if (enif_is_identical(terms[curr], atomColMajor)) *i_dest = CblasColMajor;
                else valid = 0;
            break;

            case e_transpose: 
                i_dest = va_arg(valist, int*);
                if      (enif_is_identical(terms[curr], atomNoTrans))  *i_dest = CblasNoTrans;
                else if (enif_is_identical(terms[curr], atomTrans))    *i_dest = CblasTrans;
                else if (enif_is_identical(terms[curr], atomConjTrans))*i_dest = CblasConjTrans;
                else valid = 0;
            break;

            case e_uplo: 
                i_dest = va_arg(valist, int*);
                if      (enif_is_identical(terms[curr], atomUpper)) *i_dest = CblasUpper;
                else if (enif_is_identical(terms[curr], atomLower)) *i_dest = CblasLower;
                else valid = 0;
            break;

            case e_diag: 
                i_dest = va_arg(valist, int*);
                if      (enif_is_identical(terms[curr], atomNonUnit)) *i_dest = CblasNonUnit;
                else if (enif_is_identical(terms[curr], atomUnit))    *i_dest = CblasUnit;
                else valid = 0;
                break;

            case e_side:
                i_dest = va_arg(valist, int*);
                if      (enif_is_identical(terms[curr], atomLeft))  *i_dest = CblasLeft;
                else if (enif_is_identical(terms[curr], atomRight)) *i_dest = CblasRight;
                else valid = 0;
                break;

            default:
                valid = 0;
            break;
        }

        if(!valid){
            va_end(valist);
            return curr + 1;
        }
    }
    
    va_end(valist);
    return 0;
}

// C_binary definitions/functions

//Used for debug purpose.
//Likely thread unsafe.
//Usage: debug_write("A double: %lf, an int:%d", double_val, int_val);
int debug_write(const char* fmt, ...){
    FILE* fp = fopen("priv/debug.txt", "a");
    va_list args;

    va_start(args, fmt);
    vfprintf(fp, fmt, args);
    va_end(args);

    fclose(fp);
    return 1;
}

int get_c_binary(ErlNifEnv* env, const ERL_NIF_TERM term, c_binary* result){
    int arity;
    const ERL_NIF_TERM* terms;
    void* resource;

    // Return false if: incorect record size, resource size does not match
    int b =  enif_get_tuple(env, term, &arity, &terms)
                && arity == 4
                && enif_get_uint(env, terms[1], &result->size)
                && enif_get_uint(env, terms[2], &result->offset)
                && enif_get_resource(env, terms[3], c_binary_resource, &resource)
                && result->size == enif_sizeof_resource(resource);

    result->ptr = (unsigned char*) resource;
    
    return b;
}

// *UnIVeRSAl pointer to an erlang type; currently, binary/double/c_binarry.
int get_cste_binary(ErlNifEnv* env, const ERL_NIF_TERM term, cste_c_binary* result){
    if(enif_is_binary(env, term)){
        // Read a binary.
        ErlNifBinary ebin;
        if(!enif_inspect_binary(env, term, &ebin))
            return 0;
        
        result->size    = ebin.size;
        result->offset  = 0;
        result->ptr     = ebin.data;
    }
    else{
        // Read a cbin.
        c_binary cbin;
        if(get_c_binary(env, term, &cbin)){
            result->size    = cbin.size;
            result->offset  = cbin.offset;
            result->ptr     = (const unsigned char*) cbin.ptr;
        }
        else{
            // Read a double.
            if(!enif_get_double(env, term, &result->tmp))
                return 0;

            result->size    = 8;
            result->offset  = 0;
            result->ptr     = (const unsigned char*) &result->tmp;
        }
    }
    return 1;
}


int in_bounds(int elem_size, int n_elem, int inc, c_binary b){
    int end_offset = b.offset + (elem_size*n_elem*inc);
    debug_write("end max offset: %i  offset: %u\n", end_offset, b.size);
    return (elem_size > 0 && end_offset >= 0 && end_offset <= b.size)? 0:20;
}

int in_cste_bounds(int elem_size, int n_elem, int inc, cste_c_binary b){
    int end_offset = b.offset + (elem_size*n_elem*inc);
    debug_write("end max offset: %i  offset: %u\n", end_offset, b.size);
    return (elem_size > 0 && end_offset >= 0 && end_offset <= b.size)?0:20;
}

void set_cste_c_binary(cste_c_binary *ccb, etypes type, unsigned char* ptr){
    //e_int, e_double, e_float_complex, e_double_complex, 
    switch(type){
        case e_int:            ccb->size = sizeof(int);        break;
        case e_double:         ccb->size = sizeof(double);     break;
        case e_float_complex:  ccb->size = sizeof(float)*2;    break;
        case e_double_complex: ccb->size = sizeof(double)*2;   break;
        default:               ccb->size = 0;                  break;
    }

    ccb->type   = type;
    ccb->offset = 0;
    ccb->ptr    = ptr;
}

ERL_NIF_TERM cste_c_binary_to_term(ErlNifEnv* env, cste_c_binary ccb){
    ERL_NIF_TERM result = -1;

    switch(ccb.type){
        case e_int:     int    vali = *(int*)    ccb.ptr; result = enif_make_int(env, vali);    break;
        case e_double:  double vald = *(double*) ccb.ptr; result = enif_make_double(env, vald); break;

        case e_float_complex:
        case e_double_complex:
             ErlNifBinary bin;

            debug_write("Creating binarry...\n");
            if(enif_alloc_binary(ccb.size, &bin)){
                memcpy(bin.data, ccb.ptr, ccb.size);
                debug_write("Finished copying!\n");
                if(!(result = enif_make_binary(env, &bin))){
                    enif_release_binary(&bin);
                    result = enif_make_badarg(env);
                }
            }
        break;

        default:
            result = enif_make_badarg(env);
        break;
    }
    return result;
}



ERL_NIF_TERM new(ErlNifEnv* env, int argc, const ERL_NIF_TERM* argv){
    int size = 0;
    if(!enif_get_int(env, argv[0], &size)) return enif_make_badarg(env);

    void* ptr = enif_alloc_resource(c_binary_resource, size);
    ERL_NIF_TERM resource = enif_make_resource(env, ptr);
    enif_release_resource(ptr);
    return resource; 
}

ERL_NIF_TERM copy(ErlNifEnv* env, int argc, const ERL_NIF_TERM* argv){
    ErlNifBinary bin;
    c_binary cbin;

    if(!enif_inspect_binary(env, argv[0], &bin)|| !get_c_binary(env, argv[1], &cbin))
         return enif_make_badarg(env);

    memcpy(cbin.ptr + cbin.offset, bin.data, bin.size);

    return enif_make_atom(env, "ok");
}

ERL_NIF_TERM to_binary(ErlNifEnv* env, int argc, const ERL_NIF_TERM* argv){
    c_binary cbin;
    ErlNifBinary bin;
    unsigned size;

    if(!enif_get_uint(env, argv[0], &size)
        || !get_c_binary(env, argv[1], &cbin)
        || !enif_alloc_binary(size, &bin))
        return enif_make_badarg(env);

    memcpy(bin.data, cbin.ptr + cbin.offset, size);

    return enif_make_binary(env, &bin);
}

// UNWRAPPER

// https://stackoverflow.com/questions/7666509/hash-function-for-string
unsigned long hash(char *str){
    unsigned long hash = 5381;
    int c;

    while ((c = *str++))
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}


bytes_sizes pick_size(long hash, blas_names names[], bytes_sizes sizes []){
    for(int curr=0; names[curr] != blas_name_end; curr++)
        if(names[curr]==hash)
            return sizes[curr];
    
    return 0;
}


ERL_NIF_TERM unwrapper(ErlNifEnv* env, int argc, const ERL_NIF_TERM* argv){
    int narg;
    const ERL_NIF_TERM* elements;
    char name[20];

    if(!enif_get_tuple(env, *argv, &narg, &elements)
        || !enif_get_atom(env, elements[0], name, 20, ERL_NIF_LATIN1)
    ){
        return enif_make_badarg(env);
    }

    
    int error;
    narg--;
    elements++;
    unsigned long hash_name = hash(name);
    ERL_NIF_TERM result = 0;
    //debug_write("%s=%lu\n", name, hash_name);
    switch(hash_name){

        case saxpy: case daxpy: case caxpy: case zaxpy: {
            int n; cste_c_binary alpha; cste_c_binary x; int incx; c_binary y; int incy;
            bytes_sizes type = pick_size(hash_name, (blas_names []){saxpy, daxpy, caxpy, zaxpy, blas_name_end}, (bytes_sizes[]){s_bytes, d_bytes, c_bytes, z_bytes, no_bytes});
            
            if( !(error = narg == 6? 0:21)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &n, &alpha, &x, &incx, &y, &incy))
                && !(error = in_cste_bounds(type, 1, 1, alpha)) && !(error = in_cste_bounds(type, n, incx, x))  && !(error = in_bounds(type, n, incy, y))
            )
             switch(hash_name){
                case saxpy: cblas_saxpy(n, *(double*)get_cste_ptr(alpha), get_cste_ptr(x), incx, get_ptr(y), incy); break;
                case daxpy: cblas_daxpy(n, *(double*)get_cste_ptr(alpha), get_cste_ptr(x), incx, get_ptr(y), incy); break;
                case caxpy: cblas_caxpy(n,           get_cste_ptr(alpha), get_cste_ptr(x), incx, get_ptr(y), incy); break;
                case zaxpy: cblas_zaxpy(n,           get_cste_ptr(alpha), get_cste_ptr(x), incx, get_ptr(y), incy); break;
                default: error = -2; break;
            }

        break;}

        case scopy: case dcopy: case ccopy: case zcopy:  {
            int n;  cste_c_binary x; int incx; c_binary y; int incy;
            bytes_sizes type = pick_size(hash_name, (blas_names []){scopy, dcopy, ccopy, zcopy,  blas_name_end}, (bytes_sizes[]){s_bytes, d_bytes, c_bytes, z_bytes, no_bytes});
            
            if( !(error = narg == 5? 0:21)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &n, &x, &incx, &y, &incy))
                && !(error = in_cste_bounds(type, n, incx, x)) && !(error = in_bounds(type, n, incy, y))
            )
            switch(hash_name){
                case scopy: cblas_scopy(n, get_cste_ptr(x), incx, get_ptr(y), incy); break;
                case dcopy: cblas_dcopy(n, get_cste_ptr(x), incx, get_ptr(y), incy); break;
                case ccopy: cblas_ccopy(n, get_cste_ptr(x), incx, get_ptr(y), incy); break;
                case zcopy: cblas_zcopy(n, get_cste_ptr(x), incx, get_ptr(y), incy); break;
                default: error = -2; break;
            }

        break;}

        case sswap: case dswap: case cswap: case zswap:  {
            int n;  c_binary x; int incx; c_binary y; int incy;
            bytes_sizes type = pick_size(hash_name, (blas_names []){sswap, dswap, cswap, zswap, blas_name_end}, (bytes_sizes[]){s_bytes, d_bytes, c_bytes, z_bytes, no_bytes});
            
            if( !(error = narg == 5? 0:21)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &n, &x, &incx, &y, &incy))
                && !(error = in_bounds(type, n, incx, x)) && !(error = in_bounds(type, n, incy, y))
            )
            switch(hash_name){
                case sswap: cblas_sswap(n, get_ptr(x), incx, get_ptr(y), incy); break;
                case dswap: cblas_dswap(n, get_ptr(x), incx, get_ptr(y), incy); break;
                case cswap: cblas_cswap(n, get_ptr(x), incx, get_ptr(y), incy); break;
                case zswap: cblas_zswap(n, get_ptr(x), incx, get_ptr(y), incy); break;
                default: error = -2; break;
            }
            
        break;}

        case sscal: case dscal: case cscal: case zscal: case csscal: case zdscal:  {
            int n;  cste_c_binary alpha; c_binary x; int incx;
            bytes_sizes type = pick_size(hash_name, (blas_names []){sscal, dscal, cscal, zscal,  csscal, zdscal, blas_name_end}, (bytes_sizes[]){s_bytes, d_bytes, c_bytes, z_bytes, c_bytes, z_bytes, no_bytes});
            
            if( !(error = narg == 4? 0:21)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_ptr, e_int, e_end}, &n, &alpha, &x, &incx))
                && !(error = in_cste_bounds(type, 1, 1, alpha) ) && !(error = in_bounds(type, n, incx, x))
            )
            switch(hash_name){
                case sscal:  cblas_sscal(n, *(double*) get_cste_ptr(alpha), get_ptr(x), incx); break;
                case dscal:  cblas_dscal(n, *(double*) get_cste_ptr(alpha), get_ptr(x), incx); break;
                case cscal:  cblas_cscal(n,            get_cste_ptr(alpha), get_ptr(x), incx); break;
                case zscal:  cblas_zscal(n,            get_cste_ptr(alpha), get_ptr(x), incx); break;
                case csscal: cblas_sscal(n, *(double*) get_cste_ptr(alpha), get_ptr(x), incx); break;
                case zdscal: cblas_dscal(n, *(double*) get_cste_ptr(alpha), get_ptr(x), incx); break;
                default: error = -2; break;
            }
            
        break;}

        case sdot: case ddot: case dsdot: case cdotu: case zdotu: case cdotc: case zdotc: {
            cste_c_binary dot_result;

            int n;  cste_c_binary x; int incx; cste_c_binary y; int incy;
            bytes_sizes type = pick_size(hash_name, (blas_names []){sdot, ddot, dsdot, cdotu, zdotu, cdotc, zdotc, blas_name_end}, (bytes_sizes[]){s_bytes, d_bytes, s_bytes, c_bytes, z_bytes, c_bytes, z_bytes, no_bytes});
            
            if( !(error = narg == 5? 0:21)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_end}, &n, &x, &incx, &y, &incy))
                && !(error = in_cste_bounds(type, n, incx, x) ) && !(error = in_cste_bounds(type, n, incy, y))
            ){
                switch(hash_name){
                    case sdot:                   double f_result  = cblas_sdot (n, get_cste_ptr(x), incx, get_cste_ptr(y), incy); set_cste_c_binary(&dot_result, e_double, (unsigned char*) &f_result);  break;
                    case ddot:                   double d_result  = cblas_ddot (n, get_cste_ptr(x), incx, get_cste_ptr(y), incy); set_cste_c_binary(&dot_result, e_double, (unsigned char*) &d_result);  break;
                    case dsdot:                  double ds_result = cblas_dsdot(n, get_cste_ptr(x), incx, get_cste_ptr(y), incy); set_cste_c_binary(&dot_result, e_double, (unsigned char*) &ds_result); break;
                    case cdotu: openblas_complex_float  c_result  = cblas_cdotu(n, get_cste_ptr(x), incx, get_cste_ptr(y), incy); set_cste_c_binary(&dot_result, e_float_complex,  (unsigned char*) &c_result);  break;
                    case zdotu: openblas_complex_double z_result  = cblas_zdotu(n, get_cste_ptr(x), incx, get_cste_ptr(y), incy); set_cste_c_binary(&dot_result, e_double_complex, (unsigned char*) &z_result);  break;
                    case cdotc: openblas_complex_float  cd_result = cblas_cdotc(n, get_cste_ptr(x), incx, get_cste_ptr(y), incy); set_cste_c_binary(&dot_result, e_float_complex,  (unsigned char*) &cd_result); break;
                    case zdotc: openblas_complex_double zd_result = cblas_zdotc(n, get_cste_ptr(x), incx, get_cste_ptr(y), incy); set_cste_c_binary(&dot_result, e_double_complex, (unsigned char*) &zd_result); break;
                    default: error = -2; break;
                }

                result = cste_c_binary_to_term(env, dot_result);
            }
            
        break;}

        case sdsdot: {
            cste_c_binary dot_result;

            int n;  cste_c_binary b; cste_c_binary x; int incx; cste_c_binary y; int incy;
            bytes_sizes type = s_bytes;
            
            if( !(error = narg == 6? 0:21)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_end}, &n, &b, &x, &incx, &y, &incy))
                && !(error = in_cste_bounds(type, n, incx, x) ) && !(error = in_cste_bounds(type, n, incy, y))
            ){
                double f_result  = cblas_sdsdot (n, *(double*) get_cste_ptr(b), get_cste_ptr(x), incx, get_cste_ptr(y), incy); set_cste_c_binary(&dot_result, e_double, (unsigned char*) &f_result);
                result = cste_c_binary_to_term(env, dot_result);
            }
        break;}

        case snrm2: case dnrm2: case scnrm2: case dznrm2: case sasum: case dasum: case scasum: case dzasum: case isamax: case idamax: case icamax: case izamax:
        case isamin : case idamin: case  icamin: case  izamin: case ismax: case idmax: case icmax: case izmax: case ismin: case idmin: case  icmin: case  izmin: {
            cste_c_binary u_result;
            double d_result;
            int i_result;

            int n;  cste_c_binary x; int incx;
            bytes_sizes type;
            switch(hash_name){
                case snrm2:  case sasum:  case isamax: case isamin: case ismax: case ismin: type = s_bytes;  break;   
                case dnrm2:  case dasum:  case idamax: case idamin: case idmax: case idmin: type = d_bytes;  break;
                case scnrm2: case scasum: case icamax: case icamin: case icmax: case icmin: type = c_bytes;  break;
                case dznrm2: case dzasum: case izamax: case izamin: case izmax: case izmin: type = z_bytes;  break;
                default:                                                                    type = no_bytes; break;
            }
            if( !(error = narg == 3? 0:21)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_end}, &n, &x, &incx))
                && !(error = in_cste_bounds(type, n, incx, x))
            ){
                switch(hash_name){
                    case snrm2:  d_result  = cblas_snrm2 (n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    case dnrm2:  d_result  = cblas_dnrm2 (n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    case scnrm2: d_result  = cblas_scnrm2(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    case dznrm2: d_result  = cblas_dznrm2(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    
                    case dasum:  d_result  = cblas_dasum (n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    case sasum:  d_result  = cblas_sasum (n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    case scasum: d_result  = cblas_scasum(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    case dzasum: d_result  = cblas_dzasum(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    
                    case isamax: i_result  = cblas_isamax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case idamax: i_result  = cblas_idamax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case icamax: i_result  = cblas_icamax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case izamax: i_result  = cblas_izamax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;

                    case isamin: i_result  = cblas_isamin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case idamin: i_result  = cblas_idamin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case icamin: i_result  = cblas_icamin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case izamin: i_result  = cblas_izamin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;

                    case ismax: i_result  = cblas_ismax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case idmax: i_result  = cblas_idmax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case icmax: i_result  = cblas_icmax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case izmax: i_result  = cblas_izmax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;

                    case ismin: i_result  = cblas_ismin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case idmin: i_result  = cblas_idmin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case icmin: i_result  = cblas_icmin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case izmin: i_result  = cblas_izmin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;

                    default: error = -2; break;
                }
                result = cste_c_binary_to_term(env, u_result);
            }
            
        break;}

        case srot: case drot: case csrot: case zdrot:  {
            int n;  c_binary x; int incx; c_binary y; int incy; cste_c_binary c; cste_c_binary s;
            bytes_sizes type = pick_size(hash_name, (blas_names []){srot, drot, csrot, zdrot, blas_name_end}, (bytes_sizes[]){s_bytes, d_bytes, c_bytes, z_bytes, no_bytes});
            
            if( !(error = narg == 7? 0:21)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_end}, &n, &x, &incx, &y, &incy, &c, &s))
                && !(error = in_bounds(type, n, incx, x)) && !(error = in_bounds(type, n, incy, y))
            )
            switch(hash_name){
                case srot:  cblas_srot(n, get_ptr(x),  incx, get_ptr(y), incy, *(float*)  get_cste_ptr(c), *(float*)  get_cste_ptr(s)); break;
                case drot:  cblas_drot(n, get_ptr(x),  incx, get_ptr(y), incy, *(double*) get_cste_ptr(c), *(double*) get_cste_ptr(s)); break;
                case csrot: cblas_csrot(n, get_ptr(x), incx, get_ptr(y), incy, *(float*)  get_cste_ptr(c), *(float*)  get_cste_ptr(s)); break;
                case zdrot: cblas_zdrot(n, get_ptr(x), incx, get_ptr(y), incy, *(double*) get_cste_ptr(c), *(double*) get_cste_ptr(s)); break;
                default: error = -2; break;
            }
            
        break;}

        case srotg: case drotg: case crotg: case zrotg:  {
            c_binary a; c_binary b; c_binary c; c_binary s;
            bytes_sizes type = pick_size(hash_name, (blas_names []){srotg, drotg, crotg, zrotg, blas_name_end}, (bytes_sizes[]){s_bytes, d_bytes, c_bytes, z_bytes, no_bytes});
            
            if( !(error = narg == 4? 0:21)
                && !(error = translate(env, elements, (etypes[]) {e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &a, &b, &c, &s))
                && !(error = in_bounds(type, 1, 1, a)) && !(error = in_bounds(type, 1, 1, b)) && !(error = in_bounds(type, 1, 1, c)) && !(error = in_bounds(type, 1, 1, s))
            )
            switch(hash_name){
                case srotg: cblas_srotg(get_ptr(a), get_ptr(b), get_ptr(c), get_ptr(s)); break;
                case drotg: cblas_drotg(get_ptr(a), get_ptr(b), get_ptr(c), get_ptr(s)); break;
                case crotg: cblas_crotg(get_ptr(a), get_ptr(b), get_ptr(c), get_ptr(s)); break;
                case zrotg: cblas_zrotg(get_ptr(a), get_ptr(b), get_ptr(c), get_ptr(s)); break;
                default: error = -2; break;
            }
            
        break;}

        case srotm: case drotm:  {
            int n; c_binary x; int incx; c_binary y; int incy; cste_c_binary param;
            bytes_sizes type = pick_size(hash_name, (blas_names []){srotm, drotm, blas_name_end}, (bytes_sizes[]){s_bytes, d_bytes, no_bytes});
            
            if( !(error = narg == 6? 0:21)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &n, &x, &incx, &y, &incy, &param))
                && !(error = in_bounds(type, n, incx, x)) && !(error = in_bounds(type, n, incy, y)) && !(error = in_cste_bounds(type, 5, 1, param))
            )
            switch(hash_name){
                case srotm: cblas_srotm(n, get_ptr(x), incx, get_ptr(y), incy, get_cste_ptr(param)); break;
                case drotm: cblas_srotm(n, get_ptr(x), incx, get_ptr(y), incy, get_cste_ptr(param)); break;
                default: error = -2; break;
            }
            
        break;}

        case srotmg: case drotmg:  {
            c_binary d1; c_binary d2; c_binary b1; cste_c_binary b2; c_binary param;
            bytes_sizes type = pick_size(hash_name, (blas_names []){srotmg, drotmg, blas_name_end}, (bytes_sizes[]){s_bytes, d_bytes, no_bytes});
            
            if( !(error = narg == 5? 0:21)
                && !(error = translate(env, elements, (etypes[]) {e_ptr, e_ptr, e_ptr, e_cste_ptr, e_ptr, e_end}, &d1, &d2, &b1, &b2, &param))
                && !(error = in_bounds(type, 1, 1, d1)) && !(error = in_bounds(type, 1, 1, d2)) && !(error = in_bounds(type, 1, 1, b1)) && !(error = in_cste_bounds(type, 1, 1, b2)) && !(error = in_bounds(type, 5, 1, param))
            )
            switch(hash_name){
                case srotmg: cblas_srotmg(get_ptr(d1), get_ptr(d2), get_ptr(b1), *(float*) get_cste_ptr(b2),  get_ptr(param)); break;
                case drotmg: cblas_srotmg(get_ptr(d1), get_ptr(d2), get_ptr(b1), *(double*)get_cste_ptr(b2),  get_ptr(param)); break;
                default: error = -2; break;
            }
            
        break;}

        // BLAS LEVEL 2
        case sgemv: case dgemv: case cgemv: case zgemv: {
            int layout; int trans; int m; int n; cste_c_binary alpha; cste_c_binary a; int lda; cste_c_binary x; int incx; cste_c_binary beta; c_binary y; int incy;
            bytes_sizes type = pick_size(hash_name, (blas_names []){sgemv, dgemv, cgemv, zgemv, blas_name_end}, (bytes_sizes[]){s_bytes, d_bytes, c_bytes, z_bytes, no_bytes});
            
            if( !(error = narg == 12?0:21)
                && ! (error = translate(env, elements, (etypes[]) {e_layout, e_transpose, e_uint, e_uint, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end},
                                                        &layout, &trans, &m, &n, &alpha, &a, &lda, &x, &incx, &beta, &y, &incy))
                && !(error = in_cste_bounds(type, lda, layout==CblasColMajor?n:m, a)) &&!(error = in_cste_bounds(type, n, incx, x)) && !(error = in_bounds(type, n, incy, y)))
            switch(hash_name){
                case sgemv: cblas_sgemv(layout, trans, m, n, *(double*)get_cste_ptr(alpha), get_cste_ptr(a), lda, get_cste_ptr(x), incx, *(double*)get_cste_ptr(beta), get_ptr(y), incy); break;
                case dgemv: cblas_dgemv(layout, trans, m, n, *(double*)get_cste_ptr(alpha), get_cste_ptr(a), lda, get_cste_ptr(x), incx, *(double*)get_cste_ptr(beta), get_ptr(y), incy); break;
                case cgemv: cblas_cgemv(layout, trans, m, n,           get_cste_ptr(alpha), get_cste_ptr(a), lda, get_cste_ptr(x), incx,           get_cste_ptr(beta), get_ptr(y), incy); break;
                case zgemv: cblas_zgemv(layout, trans, m, n,           get_cste_ptr(alpha), get_cste_ptr(a), lda, get_cste_ptr(x), incx,           get_cste_ptr(beta), get_ptr(y), incy); break;
                default: error = -2; break;
            }
        break;}

        case sgbmv: case dgbmv: case cgbmv: case zgbmv: {
            int layout; int trans; int m; int n; int kl; int ku; cste_c_binary alpha; cste_c_binary a; int lda; cste_c_binary x; int incx; cste_c_binary beta; c_binary y; int incy;
            bytes_sizes type = pick_size(hash_name, (blas_names []){sgbmv, dgbmv, cgbmv, zgbmv, blas_name_end}, (bytes_sizes[]){s_bytes, d_bytes, c_bytes, z_bytes, no_bytes});
            
            if( !(error = narg == 14?0:21)
                && ! (error = translate(env, elements, (etypes[]) {e_layout, e_transpose, e_uint, e_uint, e_uint, e_uint, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end},
                                                        &layout, &trans, &m, &n, &kl, &ku, &alpha, &a, &lda, &x, &incx, &beta, &y, &incy))
                && !(error = in_cste_bounds(type, lda, layout==CblasColMajor?n:m, a)) &&!(error = in_cste_bounds(type, n, incx, x)) && !(error = in_bounds(type, n, incy, y)))
            switch(hash_name){
                case sgbmv: cblas_sgbmv(layout, trans, m, n, kl, ku, *(double*)get_cste_ptr(alpha), get_cste_ptr(a), lda, get_cste_ptr(x), incx, *(double*)get_cste_ptr(beta), get_ptr(y), incy); break;
                case dgbmv: cblas_dgbmv(layout, trans, m, n, kl, ku, *(double*)get_cste_ptr(alpha), get_cste_ptr(a), lda, get_cste_ptr(x), incx, *(double*)get_cste_ptr(beta), get_ptr(y), incy); break;
                case cgbmv: cblas_cgbmv(layout, trans, m, n, kl, ku,           get_cste_ptr(alpha), get_cste_ptr(a), lda, get_cste_ptr(x), incx,           get_cste_ptr(beta), get_ptr(y), incy); break;
                case zgbmv: cblas_zgbmv(layout, trans, m, n, kl, ku,           get_cste_ptr(alpha), get_cste_ptr(a), lda, get_cste_ptr(x), incx,           get_cste_ptr(beta), get_ptr(y), incy); break;
                default: error = -2; break;
            }
        break;}

        case ssbmv: case dsbmv: {
            int layout; int uplo; int m; int n; cste_c_binary alpha; cste_c_binary a; int lda; cste_c_binary x; int incx; cste_c_binary beta; c_binary y; int incy;
            bytes_sizes type = pick_size(hash_name, (blas_names []){ssbmv, dsbmv, blas_name_end}, (bytes_sizes[]){s_bytes, d_bytes, c_bytes, z_bytes, no_bytes});
            
            if( !(error = narg == 12?0:21)
                && ! (error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_uint, e_uint, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end},
                                                        &layout, &uplo, &m, &n, &alpha, &a, &lda, &x, &incx, &beta, &y, &incy))
                && !(error = in_cste_bounds(type, lda, layout==CblasColMajor?n:m, a)) &&!(error = in_cste_bounds(type, n, incx, x)) && !(error = in_bounds(type, n, incy, y)))
            switch(hash_name){
                case ssbmv: cblas_ssbmv(layout, uplo, m, n, *(double*)get_cste_ptr(alpha), get_cste_ptr(a), lda, get_cste_ptr(x), incx, *(double*)get_cste_ptr(beta), get_ptr(y), incy); break;
                case dsbmv: cblas_dsbmv(layout, uplo, m, n, *(double*)get_cste_ptr(alpha), get_cste_ptr(a), lda, get_cste_ptr(x), incx, *(double*)get_cste_ptr(beta), get_ptr(y), incy); break;
                default: error = -2; break;
            }
        break;}

        default:
            error = -1;
        break;
    }

    switch(error){
        case -1:
            debug_write("%s=%lu,\n", name, hash_name);
            return enif_raise_exception(env, enif_make_atom(env, "Unknown blas."));
        case 0:
            return !result? enif_make_atom(env, "ok"): result;
        break;
        case 1 ... 19:
            char buff[50];
            sprintf(buff, "Could not translate argument %i.", error - 1);
            return enif_raise_exception(env, enif_make_atom(env, buff));
        break;
        case 20:
            return enif_raise_exception(env, enif_make_atom(env, "Array overflow."));
        break;
        case 21:
            return enif_raise_exception(env, enif_make_atom(env, "Invalid number of arguments."));
        break;

        default:
            debug_write("In default error.\n");
            return enif_make_badarg(env);
        break;
    }
}

int load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info){
    c_binary_resource = enif_open_resource_type(env, "c_binary", "c_binary_resource", NULL, ERL_NIF_RT_CREATE, NULL);
    debug_write("\nNew session\n-----------\n");

    atomRowMajor    = enif_make_atom(env, "blasRowMajor");
    atomColMajor    = enif_make_atom(env, "blasColMajo");
    atomNoTrans     = enif_make_atom(env, "blasNoTrans");
    atomTrans       = enif_make_atom(env, "blasTrans");
    atomConjTrans   = enif_make_atom(env, "blasConjTrans");
    atomUpper       = enif_make_atom(env, "blasUpper");
    atomLower       = enif_make_atom(env, "blasLower");
    atomNonUnit     = enif_make_atom(env, "blasNonUnit");
    atomUnit        = enif_make_atom(env, "blasUnit");
    atomLeft        = enif_make_atom(env, "blasLeft");
    atomRight       = enif_make_atom(env, "blasRight");

    return 0;
}

ErlNifFunc nif_funcs[] = { 
    {"new_nif", 1, new},
    {"copy_nif", 2, copy},
    {"bin_nif", 2, to_binary},

    {"dirty_unwrapper", 1, unwrapper},
    {"clean_unwrapper", 1, unwrapper}
};


ERL_NIF_INIT(blas, nif_funcs, load, NULL, NULL, NULL)