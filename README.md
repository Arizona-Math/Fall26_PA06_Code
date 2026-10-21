# PA06: Eigenvalues, QR, and seasonal forecasting

Math 589A. Numbook Chapter 4: power iteration, deflation, QR decomposition,
and least squares. This assignment continues the companion-matrix problems
from Fall 2024/2025 and the earlier QR-only seasonal-forecasting project.

There are two linked Gradescope submissions: **PA06 Code (100 points)** and
**PA06 Written (100 points)**. See Gradescope for due dates and their combined
course weight. Complete the provided fixed-layout Written PDF; do not insert
pages before its continuation pages. Submit code separately as described below.
Each written part asks for one result: put that result in **The Answer**, with
the supporting calculation in the scratch area. A formula or value is often
enough; prose answers should use at most two sentences. Report only the
quantities requested in the PDF, rather than copying the full experiment output.

## Start here

Use Linux or WSL, a C++17 compiler, Make, Python 3, NumPy, and Matplotlib.
The shared-library build targets Linux (`libnumerics.so`), including the
Gradescope environment. No pybind11, Cython, or Python development headers
are needed. On a course HPC system load its compiler and Python/NumPy modules;
use an allocated compute session for the timing experiment, following the
course's HPC instructions. A local Linux/WSL environment is sufficient for
debugging. Record where your reported experiments ran.

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
make                 # the untouched starter compiles
make test            # it fails until the TODOs are completed
make experiments     # tables, a plot, and measurements.json under build/
make submission      # submission.zip, containing only your three source files
```

If your HPC Python already supplies these packages, use that environment
instead. Keep `.venv/` and all generated artifacts out of version control.

Complete three files:

| File | Student work |
|---|---|
| `src/numerics.cpp` | Four C++ numerical routines: power iteration, rank-one deflation, thin MGS QR, QR least squares |
| `python/bindings.py` | Two `ctypes` wrappers: `power` and `qr`; the deflation and least-squares wrappers are supplied examples |
| `python/modeling.py` | `build_design` and recursive `forecast`; companion matrices, synthetic division, and `fit` are supplied |

The routines are short mathematical algorithms. Assume the stated valid
inputs; no general input-validation framework is required. Preserve the
interfaces in `include/numerics.h` and the Python docstrings.

## Mathematical background

### Roots, recurrences, and companion matrices

For a monic polynomial `p(z)=z^n+c_(n-1)z^(n-1)+...+c_0`, put ones on the
superdiagonal of its companion matrix and `[-c_0,...,-c_(n-1)]` in the last
row. Its characteristic polynomial is `p`. For each root `z`, the vector
`[1,z,...,z^(n-1)]` is an eigenvector. The state
`[u_t,...,u_(t+n-1)]` advances by this matrix precisely when

```math
u_{t+n}=-c_0u_t-c_1u_{t+1}-\cdots-c_{n-1}u_{t+n-1}.
```

Thus roots are modes of a difference equation: their moduli describe growth
or decay per step, while negative or nonreal roots produce oscillation.
The original Fall 2025 example is

```math
p(z)=z^5-3z^3+2z^2-5z+6.
```

With a simple real eigenvalue strictly largest in modulus and a starting
vector having its component in an eigenbasis, normalized power iteration
approaches its eigendirection. A negative dominant eigenvalue alternates
the sign. The asymptotic directional error depends on the ratio of the next
largest modulus to the dominant modulus. A nonnormal matrix may also have
large transient effects. A small eigenpair residual verifies an approximate
eigenpair, not dominance. A dominant conjugate pair does not meet the
ordinary real power method's convergence assumption.

### Deflation

For an exact real unit eigenvector `v` of `C` with eigenvalue `lambda`, form
`B=C-lambda*v*v.T`. Then `B*v=0`. If `C*w=mu*w` and `mu!=0,lambda`, the
transformed vector

```math
w'=w-\frac{\lambda}{\mu}(v^Tw)v
```

satisfies `B*w'=mu*w'`. For distinct nonzero eigenvalues these vectors and
`v` form a basis. Consequently the deflation replaces one eigenvalue by
zero and retains the other eigenvalues. Companion matrices are nonsymmetric:
**the other eigenvectors generally change**. Check the roots in the original
polynomial; do not require deflated eigenvectors to satisfy the original
matrix eigenproblem. For symmetric matrices, orthogonality makes the other
eigenvectors unchanged in exact arithmetic.

### QR and fitting

For full-column-rank `A` of size `m` by `n`, thin QR has `Q` of size `m` by
`n`, `R` of size `n` by `n`, and `Q.T@Q=I`. Modified Gram-Schmidt uses the
**updated residual column** in each projection coefficient. Minimize
`||A*x-b||_2` by computing `Q.T@b` and solving `R*x=Q.T@b` by back substitution.
Check both reconstruction and orthogonality. The textbook identity
`cond_2(A.T@A)=cond_2(A)^2` helps explain why forming normal equations can
lose accuracy. No matrix inverse or normal equations belong in the assigned
solver. Orthogonal factorization does not remove sensitivity in the data.

## C++ contracts

All arrays contain finite double-precision values, sizes are positive,
and input/output buffers do not overlap. Matrices use contiguous row-major
storage: entry `(i,j)` is at offset `i*n+j`. Inputs are not modified. No global
state or saved answers are needed. Allocating a few work vectors once per
call is fine. Use ordinary loops and `std::vector`, not a numerical library.

1. `nb_power`: `A` is a real square matrix; `x0` is nonzero. Normalize it.
   At each state compute `lambda=v.T@A@v` and `residual=||A*v-lambda*v||_2`.
   Stop when `residual <= tol*||A||_F`. Check the initial state and the final
   state, even when `max_iter=0`. Count completed normalized updates, not
   residual checks. Return status **0** when the residual condition holds,
   or **1** when the nonnegative iteration budget is exhausted; always return
   the final normalized vector, quotient, count, and actual residual.
   `tol>=0`; zero is used for a fixed-budget timing experiment. Inputs need
   not guarantee convergence: the rotation example should exhaust its budget.
2. `nb_deflate`: input `v` is a unit vector. Write
   `B[i,j]=A[i,j]-lambda*v[i]*v[j]` without changing `A` or `v`.
3. `nb_qr`: `m>=n`, full column rank, and moderate condition number. Implement
   **modified Gram-Schmidt** with a positive diagonal of `R`; write zeros below
   its diagonal. Graded inputs do not require rank-revealing pivoting or
   rank-deficiency detection. Returning status 2 on a zero pivot is allowed.
4. `nb_least_squares`: call `nb_qr`, form `Q.T@b`, and back substitute. All
   inputs satisfy the preceding QR contract. Return 0 on success; propagate
   a nonzero QR status.

For power iteration, near-zero eigenpair residual is a stopping condition,
not proof that the start selected the dominant mode. Tests include a supplied
start that is already a nondominant eigenvector. Do not replace supplied starts
with random ones. Use `max_iter` rather than a hard-coded iteration count.

## Calling the library from Python

The header declares an `extern "C"` interface. The Makefile builds a shared
library with `-fPIC -shared`; C linkage gives predictable exported symbols.
The Python module loads it with `ctypes.CDLL`. The supplied declarations
match the header's C types. The `NUMBOOK_LIB` environment variable overrides
the path and must remain supported (the grader uses it).

Complete `power` and `qr` by adapting the supplied wrappers. Accept ordinary
array-like input, including float32 arrays, Fortran-order arrays and strided
views, and prepare **C-contiguous float64** buffers. Keep their Python owners
alive throughout the call. Allocate output arrays of the declared shapes,
pass their pointers, check status, and return normal Python/NumPy results.
`power` returns `(lambda, v, iterations, residual, converged)`; status 1
returns `converged=False` without raising. Other unexpected statuses raise
`RuntimeError`. `qr` returns `(Q,R)`. Wrappers must call the C++ routines;
reimplementing them in NumPy is not this interface exercise. The grader may
substitute a diagnostic backend to check argument transport and return values.

Only the wrappers handle Python representation. C++ assumes the promised
valid contiguous arrays and never retains pointers after returning.

## The forecasting model and exact indexing

For monthly data use

```math
y_t\approx c+dt+\sum_{j=1}^N a_j y_{t-j}
    +\alpha\cos(2\pi t/12)+\beta\sin(2\pi t/12).
```

The coefficient order is `[c,d,a_1,...,a_N,alpha,beta]`. **Time is one-based**:
`y[0]` is `y_1`. For `build_design(y,N,start)`, include all rows from
`t=start` through `t=len(y)`, inclusive. The first two columns are 1 and
the original time index `t`; use lags in order 1 through N, then cosine,
then sine. Do not restart the time index at `start`. Inputs have
`N in {0,1,2}` and `N<start<=len(y)`.

`forecast(history,coef,N,horizon)` begins at `t=len(history)+1`. Copy the
history and append each predicted value before computing the next prediction.
Thus later lags use previous forecasts. Do not access the held-out data.
Preserve both input arrays; zero horizon returns an empty vector.

Use Tucson electricity (`elecuse`), not water use. Train on the first 132
observations (January 2007--December 2017), using `start=3` for both N=0 and
N=2. Forecast the last 12 months (2018) recursively. Both training matrices
have 130 rows; they have 4 and 6 columns. Compare to the seasonal-naive
forecast given by the 12 monthly values from 2017. Report RMSE, not squared
error, with units MWh/capita. The `data/README.md` records provenance and units.
Phoenix is supplied as an optional second example; it ends in June 2018,
so it must not be treated as having the same final calendar year as Tucson.

The difference between two forecasts with identical forcing and coefficients
obeys `e_t=a_1*e_(t-1)+a_2*e_(t-2)`. Its roots solve `z^2-a_1*z-a_2=0`.
Compute them with the quadratic formula or a library diagnostic; a complex
conjugate pair need not be found by ordinary real power iteration. All roots
strictly inside the unit circle imply decay of every initial-history
perturbation. This statement concerns the homogeneous error dynamics, not
future measurement errors or an unchanged future trend.

## Required experiments

`scripts/experiments.py` supplies reproducible drivers and independent checks.
Read the code and check its outputs. Q6.5 gives a rotation case to verify
independently through your Python/C++ interface.

- **Roots:** at tolerances 1e-8 and 1e-12, recover the three real roots using
  matrix deflation. Restart with all ones at every stage. Record iterations,
  deflated eigenpair residuals, original polynomial residuals, synthetic
  division remainders, and the two roots of the remaining quadratic.
- **Forecasts:** fit N=0 and N=2, report coefficients and both RMSEs, plot the
  held-out forecasts, and compare the seasonal-naive baseline. Report condition
  number, QR reconstruction and orthogonality errors, and a library comparison.
- **Timing:** compare the supplied NumPy power loop and the wrapped C++ routine
  on the same dense symmetric matrices, n=64 and 256, 200 updates, tol=0.
  Use one BLAS thread, warm up, and take the median of three calls. Set
  `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` before starting
  Python. The driver fixes the data seed. Include allocation, normalization,
  and residual checks in the timed call; exclude building and data generation.
  Independently check the final residuals and report actual iteration counts.
  There is no speed threshold or performance leaderboard for PA06.

Numerical residual stopping is not a bound on root error for a nonnormal
matrix. Compare methods with appropriately interpreted residuals. C++ need
not beat NumPy's optimized BLAS, and local timings are not Gradescope timings.
The driver retains the full diagnostic record. In the Written PDF, give only
the requested conclusions and their supporting calculations. For the Q4.5
comparison plot, add the seasonal-naive series (`train[-12:]`) to the supplied
plot of observations and the two fitted forecasts.

## Allowed tools and grading

Implement the four C++ kernels yourself; no Eigen, BLAS/LAPACK, built-in
eigensolvers, QR, inverse, or normal-equation solver inside those routines.
The Python wrappers do conversion and calls, not numerical substitutes.
Use NumPy array operations to assemble models and forecasts. Library
`eigvals`, `roots`, `qr`, `cond`, and `lstsq` are welcome for independent
verification and the explicitly allowed quadratic-root diagnostic. The
supplied synthetic-division helper may be reused. AI assistance and previous
course code may be used with acknowledgment, understanding, and verification.

| Code component | Points |
|---|---:|
| Power: signed eigenpairs, residual stopping, nonsymmetric example | 25 |
| Rank-one deflation | 10 |
| Rectangular modified Gram-Schmidt QR | 20 |
| QR least squares | 15 |
| Python wrappers (tested independently of your numerical kernels) | 15 |
| Design matrix and recursive forecast | 15 |

Upload `submission.zip` from `make submission`, or exactly one each of
`numerics.cpp`, `bindings.py`, and `modeling.py`. Do not upload compiled
libraries, plots, PDFs, the whole repository, or duplicate copies of sources.
Keep function names and signatures unchanged. The grader uses its own header
and builds your C++ file; it does not run a submitted Makefile. All input
preservation, output shapes, and iteration-budget rules above are tested.

The written submission contains the mathematics, tables, plot, explanation
of the Python/C++ interface, and verification record. It is not part of the
programming autograder. Do not move written answers out of designated regions.
