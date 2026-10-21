#include "numerics.h"
#include <cmath>
#include <vector>

int nb_power(int n, const double* A, const double* x0, int max_iter,
             double tol, double* lambda, double* v, int* iterations,
             double* residual) {
    // TODO: real normalized power iteration, including the initial/final tests.
    (void)n; (void)A; (void)x0; (void)max_iter; (void)tol;
    (void)lambda; (void)v; (void)iterations; (void)residual;
    return -99;
}

int nb_deflate(int n, const double* A, double lambda, const double* v, double* B) {
    // TODO: rank-one deflation, without changing the input arrays.
    (void)n; (void)A; (void)lambda; (void)v; (void)B;
    return -99;
}

int nb_qr(int m, int n, const double* A, double* Q, double* R) {
    // TODO: thin modified Gram-Schmidt; initialize the lower triangle of R.
    (void)m; (void)n; (void)A; (void)Q; (void)R;
    return -99;
}

int nb_least_squares(int m, int n, const double* A, const double* b, double* x) {
    // TODO: call nb_qr, compute Q^T*b, and back substitute.
    (void)m; (void)n; (void)A; (void)b; (void)x;
    return -99;
}
