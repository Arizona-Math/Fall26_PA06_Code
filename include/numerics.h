#ifndef NUMBOOK_PA06_NUMERICS_H
#define NUMBOOK_PA06_NUMERICS_H
// C ABI. Matrices are contiguous ROW-MAJOR doubles. Buffers do not overlap.
// All sizes/inputs satisfy the README assumptions. Inputs are preserved.
#ifdef __cplusplus
extern "C" {
#endif
// Normalize x0. Test the initial vector and each subsequent normalized vector.
// residual = ||A*v - lambda*v||_2, lambda = v^T*A*v, ||v||_2 = 1.
// Stop when residual <= tol*||A||_F; iterations counts completed updates.
// Return 0 on convergence, 1 on exhausted budget. Always return final outputs.
int nb_power(int n, const double* A, const double* x0, int max_iter,
             double tol, double* lambda, double* v, int* iterations,
             double* residual);
// v is a UNIT vector. B = A - lambda*v*v^T; A and v are preserved.
int nb_deflate(int n, const double* A, double lambda, const double* v, double* B);
// Thin modified Gram-Schmidt: A[m,n] = Q[m,n]*R[n,n], m >= n.
// Full column rank, with conditioning appropriate to double-precision MGS.
int nb_qr(int m, int n, const double* A, double* Q, double* R);
// Use nb_qr and back substitution; never form A^T*A or a matrix inverse.
int nb_least_squares(int m, int n, const double* A, const double* b, double* x);
// The last three routines return 0 on success. A zero pivot may return 2.
// -99 is reserved for the unimplemented starter, not a valid student result.
#ifdef __cplusplus
}
#endif
#endif
