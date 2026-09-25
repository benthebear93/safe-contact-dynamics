"""ctypes interface to the hopper residual and exact C++ Jacobians."""

import ctypes
from pathlib import Path

import numpy as np

_LIB = None
_VECTOR = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
_MATRIX = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")


def _get_lib():
    global _LIB
    if _LIB is None:
        path = Path(__file__).resolve().parents[2] / "build/libhopper_residual.so"
        if not path.exists():
            raise ImportError(f"{path} not found. Run bash scripts/build_residuals.sh")
        lib = ctypes.CDLL(str(path))
        lib.hopper_residual.argtypes = [
            _VECTOR,
            _VECTOR,
            _VECTOR,
            ctypes.c_double,
            _VECTOR,
        ]
        lib.hopper_residual.restype = None
        for name in ("hopper_rz", "hopper_rtheta"):
            function = getattr(lib, name)
            function.argtypes = [_VECTOR, _VECTOR, _VECTOR, _MATRIX]
            function.restype = None
        _LIB = lib
    return _LIB


def make_hopper_residual(model):
    """Return Simulator callbacks; retain the existing twelve-variable layout."""
    dims = (model.nq, model.nu, model.nw, model.nc, model.nb, model.nf)
    if dims != (4, 2, 0, 1, 2, 1):
        raise ValueError("C++ hopper expects nq=4, nu=2, nw=0, nc=1, nb=2, nf=1")
    lib = _get_lib()

    def inputs(z, theta):
        z = np.ascontiguousarray(z, dtype=np.float64).reshape(-1)
        theta = np.ascontiguousarray(theta, dtype=np.float64).reshape(-1)
        if z.size != 12 or theta.size != 12:
            raise ValueError("Hopper z and theta must each contain 12 values")
        params = np.array(
            [model.mb, model.ml, model.jb, model.jl, model.gravity], dtype=np.float64
        )
        return params, z, theta

    def residual(z, theta, kappa):
        out = np.empty(12)
        lib.hopper_residual(*inputs(z, theta), float(kappa), out)
        return out

    def jacobian_z(z, theta, reg=None):
        out = np.empty((12, 12))
        lib.hopper_rz(*inputs(z, theta), out)
        return out

    def jacobian_theta(z, theta):
        out = np.empty((12, 12))
        lib.hopper_rtheta(*inputs(z, theta), out)
        return out

    return residual, jacobian_z, jacobian_theta
