"""ctypes boundary. Complete power() and qr(); the other wrappers are examples.

NUMBOOK_LIB may name an alternate shared library (used by the grader).
No compilation, fitting, or experiments run on import.
"""
import ctypes as ct
import os
from pathlib import Path
import numpy as np

_LIB = None
_P = ct.POINTER(ct.c_double)


def _library():
    global _LIB
    if _LIB is None:
        default = Path(__file__).resolve().parents[1] / "build/libnumerics.so"
        _LIB = ct.CDLL(os.environ.get("NUMBOOK_LIB", str(default)))
        _LIB.nb_power.argtypes = [ct.c_int, _P, _P, ct.c_int, ct.c_double,
                                  _P, _P, ct.POINTER(ct.c_int), _P]
        _LIB.nb_deflate.argtypes = [ct.c_int, _P, ct.c_double, _P, _P]
        _LIB.nb_qr.argtypes = [ct.c_int, ct.c_int, _P, _P, _P]
        _LIB.nb_least_squares.argtypes = [ct.c_int, ct.c_int, _P, _P, _P]
        for name in ("nb_power", "nb_deflate", "nb_qr", "nb_least_squares"):
            getattr(_LIB, name).restype = ct.c_int
    return _LIB


def _ptr(a):
    return a.ctypes.data_as(_P)


def _array(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def _check(status):
    if status != 0:
        raise RuntimeError(f"C++ routine returned status {status}")


def power(A, x0, max_iter=10000, tol=1e-12):
    """Return (lambda, unit_vector, iterations, residual, converged).

    Accept array-like inputs, including float32 and noncontiguous views.
    Convert to C-contiguous float64, retain their owners during the C call,
    allocate outputs, and call nb_power. Status 1 is a normal exhausted
    iteration budget: return its final outputs with converged=False.
    Preserve inputs. Do NOT implement the iteration in Python here.
    """
    raise NotImplementedError("Complete the ctypes wrapper for nb_power")


def qr(A):
    """Return thin (Q,R) from nb_qr for an m-by-n full-rank A, m >= n.

    Allocate Q with shape (m,n), R with shape (n,n), both C-contiguous
    float64. Pass dimensions in the declared order; check the status.
    """
    raise NotImplementedError("Complete the ctypes wrapper for nb_qr")


def deflate(A, eigenvalue, v):
    A, v = _array(A), _array(v)
    B = np.empty_like(A)
    _check(_library().nb_deflate(A.shape[0], _ptr(A), float(eigenvalue),
                               _ptr(v), _ptr(B)))
    return B


def least_squares(A, b):
    A, b = _array(A), _array(b)
    m, n = A.shape
    x = np.empty(n, dtype=np.float64)
    _check(_library().nb_least_squares(m, n, _ptr(A), _ptr(b), _ptr(x)))
    return x
