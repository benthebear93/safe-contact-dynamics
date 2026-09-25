import numpy as np
from dataclasses import dataclass
from src.helper import friction_dim, nc_impact_check


def _pp(name, arr):
    """Pretty-print an index list with its length."""
    try:
        n = len(arr)
    except TypeError:
        arr = list(arr)
        n = len(arr)
    print(f"{name:>10} (len={n}): {arr}")


class IndicesZ:
    def __init__(self, q, gamma, s_gamma, psi, b, s_psi, s_b):
        self.q = q
        self.gamma = gamma
        self.s_gamma = s_gamma
        self.psi = psi
        self.b = b
        self.s_psi = s_psi
        self.s_b = s_b

    def initialize_z(self, z, q):
        eps = 3.16227766e-2
        z[self.q] = q
        # z[self.gamma] = 1.0
        # z[self.s_gamma] = 1.0
        # z[self.psi] = 1.0
        # z[self.b] = 0.1
        # z[self.s_psi] = 1.0
        # z[self.s_b] = 0.1
        z[self.gamma] = eps
        z[self.s_gamma] = eps
        z[self.psi] = eps
        z[self.b] = eps
        z[self.s_psi] = eps
        z[self.s_b] = eps
        return z


class IndicesTheta:
    def __init__(self, model, nf=1):
        nq = model.nq
        nu = model.nu
        nw = model.nw
        nf = model.nf

        self.q1 = np.arange(0, nq)
        self.q2 = np.arange(nq, 2 * nq)
        self.u = np.arange(2 * nq, 2 * nq + nu)
        self.w = np.arange(2 * nq + nu, 2 * nq + nu + nw)
        self.f = np.arange(2 * nq + nu + nw, 2 * nq + nu + nw + nf)
        self.h = np.arange(2 * nq + nu + nw + nf, 2 * nq + nu + nw + nf + 1)

    def initialize_theta(self, theta, q1, q2, u, w, f, h):
        theta[self.q1] = q1
        theta[self.q2] = q2
        theta[self.u] = u
        theta[self.w] = w
        theta[self.f] = f
        theta[self.h] = h
        return theta


@dataclass
class IndicesOptimization:
    nz: int
    n_delta: int
    ortz: list[list[int]]
    ort_delta: list[list[int]]
    socz: list[list[list[int]]]
    soc_delta: list[list[list[int]]]
    equr: list[int]
    ortr: list[int]
    socr: list[int]
    socri: list[list[int]]
    bil: list[int]


def index_q2(model, env, quat=False):
    nq = model.nq
    nquat = model.nquat if hasattr(model, "nquat") else 0
    return list(range(nq - nquat))


def index_gamma1(model, env, quat=False):
    nq = model.nq
    nc = model.nc  # model.nc
    nquat = model.nquat if hasattr(model, "nquat") else 0
    off = nq - nquat
    return list(range(off, off + nc))


def index_b1(model, env, quat=False):
    nq = model.nq
    nc = model.nc  # model.nc
    nquat = model.nquat if hasattr(model, "nquat") else 0
    nb = nc_impact_check(env, nc)
    off = (nq - nquat) + nc
    return list(range(off, off + nb))


def index_psi1(model, env, quat=False):
    nq = model.nq
    nc = model.nc  # model.nc
    nquat = model.nquat if hasattr(model, "nquat") else 0
    # nb = nc * friction_dim(env)
    nb = nc_impact_check(env, nc)
    off = (nq - nquat) + nc + nb
    return list(range(off, off + nc))


def index_s1(model, env, quat=False):
    # slack variable for normal force
    nq = model.nq
    nc = model.nc  # model.nc
    nquat = model.nquat if hasattr(model, "nquat") else 0
    # nb = nc * friction_dim(env)
    nb = nc_impact_check(env, nc)
    off = (nq - nquat) + nc + nb + nc
    return list(range(off, off + nc))


def index_eta1(model, env, quat=False):
    # slack variable for friction force
    nq = model.nq
    nc = model.nc  # model.nc
    nquat = model.nquat if hasattr(model, "nquat") else 0
    # nb = nc * friction_dim(env)
    nb = nc_impact_check(env, nc)
    off = (nq - nquat) + nc + nb + nc + nc
    return list(range(off, off + nb))


def index_s2(model, env, quat=False):
    # slack variable for contact point velocity
    nq = model.nq
    nc = model.nc  # model.nc
    nquat = model.nquat if hasattr(model, "nquat") else 0
    # nb = nc * friction_dim(env)
    nb = nc_impact_check(env, nc)
    off = (nq - nquat) + nc + nb + nc + nc + nb
    return list(range(off, off + nc))


# θ indices


# Residual indices


def index_dyn(model, env, quat=False):  # dynamics
    nq = model.nq
    nquat = model.nquat if hasattr(model, "nquat") else 0
    return list(range(nq - nquat))


def index_imp(model, env, quat=False):  # impact
    nq = model.nq
    nc = model.nc
    nquat = model.nquat if hasattr(model, "nquat") else 0
    off = nq - nquat
    return list(range(off, off + nc))


def index_mdp(model, env, quat=False):  # friction reltate velocity?
    nq = model.nq
    nc = model.nc
    nquat = model.nquat if hasattr(model, "nquat") else 0
    nb = nc_impact_check(env, nc)
    off = (nq - nquat) + nc
    return list(range(off, off + nb))


def index_fri(model, env, quat=False):  # friction
    nq = model.nq
    nc = model.nc
    nquat = model.nquat if hasattr(model, "nquat") else 0
    nb = nc_impact_check(env, nc)
    off = (nq - nquat) + nc + nb
    return list(range(off, off + nc))


def index_bimp(model, env, quat=False):  # bilinear impact
    nq = model.nq
    nc = model.nc
    nquat = model.nquat if hasattr(model, "nquat") else 0
    nb = nc_impact_check(env, nc)
    off = (nq - nquat) + nc + nb + nc
    return list(range(off, off + nc))


def index_bmdp(model, env, quat=False):  # bilinear vel?
    nq = model.nq
    nc = model.nc
    nquat = model.nquat if hasattr(model, "nquat") else 0
    nb = nc_impact_check(env, nc)
    off = (nq - nquat) + nc + nb + nc + nc
    return list(range(off, off + nb))


def index_bfri(model, env, quat=False):  # bilinear friction
    nq = model.nq
    nc = model.nc  # model.nc
    nquat = model.nquat if hasattr(model, "nquat") else 0
    nb = nc * friction_dim(env)
    nb = nc_impact_check(env, nc)
    off = (nq - nquat) + nc + nb + nc + nc + nb
    return list(range(off, off + nc))


# Aggregated Indices


def linearization_var_index(model, env, quat=False, verbose=False):
    iq2 = index_q2(model, env, quat=quat)
    igamma1 = index_gamma1(model, env, quat=quat)
    ib1 = index_b1(model, env, quat=quat)
    ipsi1 = index_psi1(model, env, quat=quat)
    is1 = index_s1(model, env, quat=quat)
    ieta1 = index_eta1(model, env, quat=quat)
    is2 = index_s2(model, env, quat=quat)
    ix = iq2
    iy1 = igamma1 + ib1 + ipsi1
    iy2 = is1 + ieta1 + is2
    if verbose:
        _pp("iq2", ix)
        _pp("igamma1 + ib1 + ipsi1", iy1)
        _pp("is1 + ieta1 + is2", is1 + ieta1 + is2)
    return ix, iy1, iy2


def linearization_term_index(model, env, quat=False, verbose=False):
    # --- helpers ------------------------------------------------------------
    def _pp(name, arr):
        """Pretty-print an index list with its length."""
        try:
            n = len(arr)
        except TypeError:
            arr = list(arr)
            n = len(arr)
        print(f"{name:>10} (len={n}): {arr}")

    # dyn = [dyn]
    # rst = [s1  - ..., ≡ ialt
    #        η1  - ...,
    #        s2  - ...,]
    # bil = [γ1 .* s1 .- κ;
    #        b1 .* η1 .- κ;
    #        ψ1 .* s2 .- κ]
    # Base index blocks
    idyn = index_dyn(model, env, quat=quat)  # dynamics
    iimp = index_imp(model, env, quat=quat)  # impact
    imdp = index_mdp(model, env, quat=quat)  # mdp
    ifri = index_fri(model, env, quat=quat)  # frictin
    ibimp = index_bimp(model, env, quat=quat)  # bilinear (slack for impact)
    ibmdp = index_bmdp(model, env, quat=quat)  # bilinear (slack for mdp)
    ibfri = index_bfri(model, env, quat=quat)  # bilinear (slack for friction)

    # print(
    #     f"idyn {idyn, len(idyn)} iimp {iimp, len(iimp)} imdp {imdp, len(imdp)} \n ifri {ifri, len(ifri)} ibimp {ibimp, len(ibimp)} ibmdp {ibmdp, len(ibmdp)} ibfri {ibfri, len(ibfri)}"
    # )
    # Groupings
    irst = iimp + imdp + ifri  # residual stack (impact + mdp + friction)
    ibil = ibimp + ibmdp + ibfri  # bilinear residual stack
    ialt = iimp  # alternative/selected block (as in your code)

    if verbose:
        print("\n=== Linearization term indices ===")
        print(f"quat mode: {quat}")
        _pp("idyn", idyn)
        _pp("iimp", iimp)
        _pp("ifri", ifri)
        _pp("imdp", imdp)
        _pp("ibimp", ibimp)
        _pp("ibfri", ibfri)
        _pp("ibmdp", ibmdp)

        print("\n=== Grouped blocks ===")
        _pp("irst = iimp+imdp+ifri", irst)
        _pp("ibil = ibimp+ibmdp+ibfri", ibil)
        _pp("ialt = iimp", ialt)
        print()

    return idyn, irst, ibil, ialt


