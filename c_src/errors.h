
typedef enum errors {
    ERROR_SIGSEV    = 20,   // Array overflow
    ERROR_N_ARG     = 21,   // Invalid number of arguments
    ERROR_NOT_FOUND = 404,  // Switch case branch not solved.
    ERROR_NO_BLAS   = -1,   // Invalid blas name
    ERROR_NONE      = 0    // No error.
} errors;

inline errors test_n_arg(int narg, int expected){
    return narg == expected? ERROR_NONE:ERROR_N_ARG;
}