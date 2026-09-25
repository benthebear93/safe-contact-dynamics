from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


_LIB_NAME = "libparticle_residual.so"
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
        _LIB.particle_set_params.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ]
        _LIB.particle_set_params.restype = None

        _LIB.particle_residual.argtypes = [_ND_1D, _ND_1D, ctypes.c_double, _ND_1D]
        _LIB.particle_residual.restype = None

        _LIB.particle_rz.argtypes = [_ND_1D, _ND_1D, _ND_2D]
        _LIB.particle_rz.restype = None

        _LIB.particle_rtheta.argtypes = [_ND_1D, _ND_1D, _ND_2D]
        _LIB.particle_rtheta.restype = None
    return _LIB


def make_particle_residual(particle):
    if particle.nq != 3 or particle.nu != 3 or particle.nc != 1 or particle.nb != 4:
        raise ValueError(
            "C++ particle residual expects nq=3, nu=3, nc=1, nb=4."
        )
    if particle.nw != 0 or particle.nf != 1:
        raise ValueError("C++ particle residual expects nw=0, nf=1.")

    lib = _get_lib()
    lib.particle_set_params(particle.mb, particle.gravity, particle.r)

    def r(z, theta, kappa):
        z_arr = np.ascontiguousarray(z, dtype=np.float64).reshape(-1)
        theta_arr = np.ascontiguousarray(theta, dtype=np.float64).reshape(-1)
        out = np.zeros(15, dtype=np.float64)
        lib.particle_residual(z_arr, theta_arr, float(kappa), out)
        return out

    def rz(z, theta):
        z_arr = np.ascontiguousarray(z, dtype=np.float64).reshape(-1)
        theta_arr = np.ascontiguousarray(theta, dtype=np.float64).reshape(-1)
        out = np.zeros((15, 15), dtype=np.float64)
        lib.particle_rz(z_arr, theta_arr, out)
        return out

    def rtheta(z, theta):
        z_arr = np.ascontiguousarray(z, dtype=np.float64).reshape(-1)
        theta_arr = np.ascontiguousarray(theta, dtype=np.float64).reshape(-1)
        out = np.zeros((15, 11), dtype=np.float64)
        lib.particle_rtheta(z_arr, theta_arr, out)
        return out

    return r, rz, rtheta
