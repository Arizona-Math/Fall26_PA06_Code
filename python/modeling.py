"""Polynomial helpers and seasonal difference equations.

Complete build_design() and forecast(). fit() already calls the C++ solver.
Time is one-based: y[0] is y_1. Coefficients have the fixed order
[c, d, a_1, ..., a_N, alpha, beta]. The seasonal period is 12.
"""
import numpy as np
from bindings import least_squares


def companion(coefficients):
    """Monic polynomial, coefficients in descending powers (supplied)."""
    p = np.asarray(coefficients, dtype=float)
    n = len(p) - 1
    C = np.zeros((n, n))
    C[:-1, 1:] = np.eye(n - 1)
    C[-1] = -p[:0:-1]
    return C


def synthetic_division(coefficients, root):
    """Return quotient and remainder; Horner recursion from earlier work."""
    b = np.array(coefficients, dtype=float, copy=True)
    for j in range(1, len(b)):
        b[j] += root * b[j - 1]
    return b[:-1], b[-1]


def build_design(y, N, start=3):
    """Return A,b for observations t=start,...,len(y), INCLUSIVE.

    Contract: N is 0, 1, or 2; N < start <= len(y).
    Rows: [1, t, y_(t-1), ..., y_(t-N), cos(2*pi*t/12), sin(2*pi*t/12)].
    A.shape = (len(y)-start+1, N+4); b.shape = (len(y)-start+1,).
    No future observations; do not alter y. Use the same start for models
    that will be compared. The assignment uses start=3 for N=0 and N=2.
    """
    raise NotImplementedError("Construct the rectangular design matrix")


def fit(y, N, start=3):
    A, b = build_design(y, N, start)
    coef = least_squares(A, b)
    return coef, A @ coef - b


def forecast(history, coef, N, horizon=12):
    """Return horizon recursive forecasts for times T+1,...,T+horizon.

    history has length T >= max(N,1). coef has length N+4 in the order
    above. Append each prediction to a private working history so subsequent
    lags use predictions. Time and seasonal phase continue from training.
    Preserve both inputs. horizon=0 returns an empty float64 vector.
    """
    raise NotImplementedError("Forecast without using held-out observations")
