#ifndef EWB_INCLUDED
#define EWB_INCLUDED
#include "erl_nif.h"
#include <cblas.h>

// Types translator
ERL_NIF_TERM atomRowMajor, atomColMajor, atomNoTrans, atomTrans, atomConjTrans, atomUpper,atomLower, atomNonUnit, atomUnit, atomLeft, atomRight;

typedef enum types {e_int, e_uint, e_float, e_double, e_ptr, e_cste_ptr, e_float_complex, e_double_complex, e_layout, e_transpose, e_uplo, e_diag, e_side, e_end} etypes;
int translate(ErlNifEnv* env, const ERL_NIF_TERM* terms, const etypes* format, ...);


// C binary definition
// --------------------------------------------

typedef struct{
    unsigned int size;
    unsigned int offset;
    unsigned char* ptr;
} c_binary;

inline void* get_ptr(c_binary cb){return (void*) cb.ptr + cb.offset;}
int get_c_binary(ErlNifEnv* env, const ERL_NIF_TERM term, c_binary* result);
int in_bounds(int elem_size, int n_elem, int inc, c_binary b);

typedef struct{
    unsigned int size;
    unsigned int offset;
    const unsigned char* ptr;
    double tmp;
    etypes type;
} cste_c_binary;

inline const void* get_cste_ptr(cste_c_binary cb){return (void*) cb.ptr + cb.offset;}
int get_cste_binary(ErlNifEnv* env, const ERL_NIF_TERM term, cste_c_binary* result);
int in_cste_bounds(int elem_size, int n_elem, int inc, cste_c_binary b);

void set_cste_c_binary(cste_c_binary* ccb, etypes type, unsigned char* ptr);
ERL_NIF_TERM cste_c_binary_to_term(ErlNifEnv* env, cste_c_binary ccb);


// Private stuff
int debug_write(const char* fmt, ...);
ErlNifResourceType *c_binary_resource;
int load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info);
int upgrade(ErlNifEnv* caller_env, void** priv_data, void** old_priv_data, ERL_NIF_TERM load_info);
int unload(ErlNifEnv* caller_env, void* priv_data);

ERL_NIF_TERM new(ErlNifEnv* env, int argc, const ERL_NIF_TERM* argv);
ERL_NIF_TERM copy(ErlNifEnv* env, int argc, const ERL_NIF_TERM* argv);
ERL_NIF_TERM to_binary(ErlNifEnv* env, int argc, const ERL_NIF_TERM* argv);



// Blas wrapper
// --------------------------------------------

unsigned long hash(char *str);
int load_ebw(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info);
ERL_NIF_TERM unwrapper(ErlNifEnv* env, int argc, const ERL_NIF_TERM* argv);



#endif