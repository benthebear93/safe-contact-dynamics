"""Shared SLSQP execution for the example CBF quadratic programs."""

import time

import numpy as np
from scipy import optimize


def minimize_qp(H, g, A, b, lbx, ubx, x0, *, maxiter, timing=None):
    """Minimize 0.5*x.T@H@x + g.T@x with A@x <= b and box bounds.

    Inputs are already normalized by the caller. Initialization, feasibility
    tolerances, exception handling and fallback controls stay in each wrapper.
    Return SciPy's result unchanged; optionally record the solve duration.
    """

    def objective(x):
        return 0.5 * float(x @ H @ x) + float(g @ x)

    def jacobian(x):
        return H @ x + g

    constraint = optimize.LinearConstraint(A, lb=-np.inf * np.ones(A.shape[0]), ub=b)
    t0 = time.perf_counter()
    result = optimize.minimize(
        objective,
        x0=x0,
        jac=jacobian,
        method="SLSQP",
        bounds=optimize.Bounds(lbx, ubx),
        constraints=[constraint],
        options={"maxiter": maxiter, "ftol": 1e-9, "disp": False},
    )
    t1 = time.perf_counter()
    if timing is not None:
        timing["total_s"] += t1 - t0
        timing["by_solver_s"]["scipy_slsqp"] = timing["by_solver_s"].get(
            "scipy_slsqp", 0.0
        ) + (t1 - t0)
        timing["calls"] += 1
    return result
