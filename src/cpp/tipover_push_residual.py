from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


_LIB_NAME = "libtipover_push_residual.so"
_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LIB = _ROOT / "build" / _LIB_NAME


def _load_lib(path: Path | None = None) -> ctypes.CDLL:
    lib_path = path if path is not None else _DEFAULT_LIB
    if not lib_path.exists():
        raise ImportError(
            f"{_LIB_NAME} not found at {lib_path}. "
            "Build it with: g++ -O3 -std=c++17 -shared -fPIC cpp/tipover_push_residual.cpp -o build/libtipover_push_residual.so"
        )
    return ctypes.CDLL(str(lib_path))


_LIB = None

_ND_1D = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
_ND_2D = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")


def _get_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = _load_lib()
        _LIB.tipover_push_set_params.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ]
        _LIB.tipover_push_set_params.restype = None

        _LIB.tipover_push_residual.argtypes = [_ND_1D, _ND_1D, ctypes.c_double, _ND_1D]
        _LIB.tipover_push_residual.restype = None

        _LIB.tipover_push_rz.argtypes = [_ND_1D, _ND_1D, _ND_2D]
        _LIB.tipover_push_rz.restype = None

        _LIB.tipover_push_rtheta.argtypes = [_ND_1D, _ND_1D, _ND_2D]
        _LIB.tipover_push_rtheta.restype = None
    return _LIB


def make_tipover_push_residual(model, use_cpp_jac: bool = True):
    """Use native residuals and Jacobians; use_cpp_jac is a compatibility argument."""
    if (
        model.nq != 7
        or model.nu != 2
        or model.nc != 5
        or model.nb != 10
        or model.nw != 0
        or model.nf != 2
    ):
        raise ValueError(
            "C++ tipover_push residual expects nq=7, nu=2, nc=5, nb=10, nw=0, nf=2."
        )

    lib = _get_lib()
    lib.tipover_push_set_params(
        model.mb,
        model.mp,
        model.I,
        model.box_half_width,
        model.box_half_height,
        model.pusher_radius,
        model.gravity,
        model.mu_floor,
        model.mu_pusher,
    )

    def r(z, theta, kappa):
        z_arr = np.ascontiguousarray(z, dtype=np.float64).reshape(-1)
        theta_arr = np.ascontiguousarray(theta, dtype=np.float64).reshape(-1)
        out = np.zeros(47, dtype=np.float64)
        lib.tipover_push_residual(z_arr, theta_arr, float(kappa), out)
        return out

    def rz(z, theta):
        z_arr = np.ascontiguousarray(z, dtype=np.float64).reshape(-1)
        theta_arr = np.ascontiguousarray(theta, dtype=np.float64).reshape(-1)
        out = np.zeros((47, 47), dtype=np.float64)
        lib.tipover_push_rz(z_arr, theta_arr, out)
        return out

    def rtheta(z, theta):
        z_arr = np.ascontiguousarray(z, dtype=np.float64).reshape(-1)
        theta_arr = np.ascontiguousarray(theta, dtype=np.float64).reshape(-1)
        out = np.zeros((47, 19), dtype=np.float64)
        lib.tipover_push_rtheta(z_arr, theta_arr, out)
        return out

    return r, rz, rtheta
