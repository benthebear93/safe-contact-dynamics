from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


_LIB_NAME = "libpusher_residual.so"
_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LIB = _ROOT / "build" / _LIB_NAME


def _load_lib(path: Path | None = None) -> ctypes.CDLL:
    lib_path = path if path is not None else _DEFAULT_LIB
    if not lib_path.exists():
        raise ImportError(
            f"{_LIB_NAME} not found at {lib_path}. "
            "Build it with: bash scripts/build_residuals.sh"
        )
    return ctypes.CDLL(str(lib_path))


_LIB = None

_ND_1D = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
_ND_2D = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")


def _get_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = _load_lib()
        _LIB.pusher_set_params.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ]
        _LIB.pusher_set_params.restype = None

        _LIB.pusher_residual.argtypes = [_ND_1D, _ND_1D, ctypes.c_double, _ND_1D]
        _LIB.pusher_residual.restype = None

        _LIB.pusher_rz.argtypes = [_ND_1D, _ND_1D, _ND_2D]
        _LIB.pusher_rz.restype = None

        _LIB.pusher_rtheta.argtypes = [_ND_1D, _ND_1D, _ND_2D]
        _LIB.pusher_rtheta.restype = None
    return _LIB


def make_pusher_residual(pusher):
    if (
        pusher.nq != 7
        or pusher.nu != 2
        or pusher.nc != 5
        or pusher.nb != 20
        or pusher.nw != 0
        or pusher.nf != 2
    ):
        raise ValueError(
            "C++ pusher residual expects nq=7, nu=2, nc=5, nb=20, nw=0, nf=2."
        )

    lib = _get_lib()
    lib.pusher_set_params(
        pusher.mb,
        pusher.mp,
        pusher.I,
        pusher.r_box,
        pusher.r_pusher,
        pusher.gravity,
    )

    def r(z, theta, kappa):
        z_arr = np.ascontiguousarray(z, dtype=np.float64).reshape(-1)
        theta_arr = np.ascontiguousarray(theta, dtype=np.float64).reshape(-1)
        out = np.zeros(67, dtype=np.float64)
        lib.pusher_residual(z_arr, theta_arr, float(kappa), out)
        return out

    def rz(z, theta):
        z_arr = np.ascontiguousarray(z, dtype=np.float64).reshape(-1)
        theta_arr = np.ascontiguousarray(theta, dtype=np.float64).reshape(-1)
        out = np.zeros((67, 67), dtype=np.float64)
        lib.pusher_rz(z_arr, theta_arr, out)
        return out

    def rtheta(z, theta):
        z_arr = np.ascontiguousarray(z, dtype=np.float64).reshape(-1)
        theta_arr = np.ascontiguousarray(theta, dtype=np.float64).reshape(-1)
        out = np.zeros((67, 19), dtype=np.float64)
        lib.pusher_rtheta(z_arr, theta_arr, out)
        return out

    return r, rz, rtheta
