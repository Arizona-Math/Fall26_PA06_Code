"""Reproducible experiment driver; complete the student routines first.

PA06_PYTHON optionally selects an instructor/reference Python directory.
Reports measured outputs; does not silently substitute library algorithms.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import json
import platform
import sys
import time
import statistics
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get("PA06_PYTHON", str(ROOT / "python")))
import bindings as nb
import modeling as model


def numpy_power(A, x0, max_iter, tol):
    x = x0 / np.linalg.norm(x0)
    scale = np.linalg.norm(A, "fro")
    for k in range(max_iter+1):
        y = A @ x
        lam = x @ y
        res = np.linalg.norm(y-lam*x)
        if res <= tol*scale or k == max_iter:
            return float(lam), x, k, float(res), bool(res <= tol*scale)
        x = y / np.linalg.norm(y)


def run(output=None):
    out = Path(output or ROOT / "build/experiments")
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / "mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = {"environment": {"platform": platform.platform(),
              "processor": platform.processor(), "numpy": np.__version__,
              "python": platform.python_version(),
              "threads": {k: os.environ.get(k, "unset") for k in
                          ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}}}
    p = np.array([1., 0., -3., 2., -5., 6.])
    C = model.companion(p)
    report["roots"] = []
    for tol in (1e-8, 1e-12):
        B, q = C.copy(), p.copy()
        rows = []
        for stage in range(3):
            lam, v, k, residual, ok = nb.power(B, np.ones(5), 10000, tol)
            if not ok:
                raise RuntimeError(f"Power iteration exhausted at stage {stage+1}")
            rows.append({"stage": stage+1, "lambda": lam, "iterations": k,
                         "deflated_residual": residual,
                         "original_vector_residual": float(np.linalg.norm(C@v-lam*v)),
                         "polynomial_residual": float(abs(np.polyval(p, lam)))})
            B = nb.deflate(B, lam, v)
            q, remainder = model.synthetic_division(q, lam)
            rows[-1]["division_remainder"] = float(remainder)
        z = np.emath.sqrt(q[1]**2-4*q[0]*q[2])
        pair = [(-q[1]+z)/(2*q[0]), (-q[1]-z)/(2*q[0])]
        report["roots"].append({"tol": tol, "stages": rows,
                    "quadratic": q.tolist(),
                    "complex_roots": [[float(z.real), float(z.imag)] for z in pair]})

    data = np.genfromtxt(ROOT / "data/tucsonAZ.csv", delimiter=",", names=True)
    assert np.array_equal(data["year"][-12:], np.full(12, 2018))
    assert np.array_equal(data["month"][-12:], np.arange(1, 13))
    y = data["elecuse"]
    train, test = y[:-12], y[-12:]
    report["forecast"] = []
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(np.arange(109, 145), y[108:], "k.-", label="observed")
    for N in (0, 2):
        A, b = model.build_design(train, N, start=3)
        coef, residual = model.fit(train, N, start=3)
        pred = model.forecast(train, coef, N, 12)
        Q, R = nb.qr(A)
        ref = np.linalg.lstsq(A, b, rcond=None)[0]
        roots = np.roots(np.r_[1., -coef[2:2+N]]) if N else np.array([])
        report["forecast"].append({"N": N, "shape": list(A.shape), "coefficients": coef.tolist(),
            "training_rmse": float(np.sqrt(np.mean(residual**2))),
            "forecast_rmse": float(np.sqrt(np.mean((pred-test)**2))),
            "condition_number": float(np.linalg.cond(A)),
            "qr_relative_residual": float(np.linalg.norm(A-Q@R)/np.linalg.norm(A)),
            "orthogonality_error": float(np.linalg.norm(Q.T@Q-np.eye(A.shape[1]), 2)),
            "reference_coefficient_error": float(np.linalg.norm(coef-ref)),
            "recurrence_roots": [[float(z.real), float(z.imag)] for z in roots],
            "forecasts": pred.tolist()})
        ax.plot(np.arange(133, 145), pred, ".-", label=f"recursive forecast N={N}")
    baseline = train[-12:]
    report["seasonal_naive_rmse"] = float(np.sqrt(np.mean((baseline-test)**2)))
    ax.axvline(132.5, color="0.5", linestyle="--")
    ax.set(xlabel="Month (January 2007 = 1)", ylabel="Electricity use (MWh/capita)",
           title="Tucson: train through 2017, forecast 2018")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "forecast.png", dpi=180); plt.close(fig)

    report["timings"] = []
    rng = np.random.default_rng(58906)
    for n in (64, 256):
        U, _ = np.linalg.qr(rng.normal(size=(n,n)))
        A = np.ascontiguousarray((U*np.linspace(1., 5., n))@U.T)
        x0 = np.ones(n)
        results = {}
        for label, method in (("numpy", numpy_power), ("cpp_ctypes", nb.power)):
            method(A, x0, 20, 0.)  # warm-up; fixed budget, no convergence assumed
            samples=[]
            for _ in range(3):
                start=time.perf_counter(); answer=method(A,x0,200,0.)
                samples.append(time.perf_counter()-start)
            lam, v, k, residual, ok = answer
            assert k == 200
            assert np.isclose(residual, np.linalg.norm(A@v-lam*v), rtol=1e-8, atol=1e-12)
            results[label] = {"seconds": statistics.median(samples), "iterations": k,
                              "lambda": lam, "residual": residual}
        report["timings"].append({"n": n, "methods": results})
    (out / "measurements.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv)>1 else None)
