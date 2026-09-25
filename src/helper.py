import numpy as np
from typing import List, Tuple
from abc import ABC
from autograd import numpy as anp


def rotation_matrix_yaw(a):
    """3D rotation matrix for yaw about z-axis (right-handed)."""
    c = anp.cos(a)
    s = anp.sin(a)
    return anp.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def friction_dim(env):
    if env.rotation_type == "R2" and env.cone_type == "LinearizedCone":
        return 2
    elif env.rotation_type == "R3" and env.cone_type == "LinearizedCone":
        return 4
    elif env.rotation_type == "R2" and env.cone_type == "NonlinearCone":
        return 1
    elif env.rotation_type == "R3" and env.cone_type == "NonlinearCone":
        return 2
    else:
        raise ValueError("Unsupported Environment type combination")


def nc_impact_check(env, nc):
    try:
        nc_impact = env.nc_impact
    except:
        nc_impact = 0

    nb = nc * friction_dim(env)
    if nc_impact != 0:
        nb = nc * friction_dim(env) - 2
    return nb


def num_var(model, env, quat=False):
    nq = model.nq
    nc = model.nc
    nquat = 0
    nb = nc_impact_check(env, nc)
    # nc * friction_dim(env)
    return (nq - nquat) + nc + nb + nc + nc + nb + nc


def num_data(model):
    nq = model.nq
    nu = model.nu
    nw = model.nw
    nf = model.nf
    return nq + nq + nu + nw + nf + 1


class Space(ABC):
    """Abstract base class for optimization space."""

    pass


class Euclidean(Space):
    def __init__(self, n: int):
        """
        Represents Euclidean space of dimension n.
        """
        self.n = n


def soc_step_length_single(
    lmbd, delta, rho_vec, idx, tau=0.99, eps=1e-14, verbose=False
):
    # Nesterov–Todd scaling
    # check Section 8.2 CVXOPT
    # Adding to slack ϵ to make sure that we never get out of the cone

    l0 = lmbd[0]
    le = lmbd[idx]
    de = delta[idx]

    ll = max(l0**2 - np.dot(le, le), 1.0e-25) + eps
    ld = (
        l0 * delta[0] - np.dot(le, de) + eps
    )  # check how much soc margin changes in step dir

    rho_s = ld / ll  # normalize margin of step
    rho_vec[:] = de / np.sqrt(ll)  # normalize tail dir vector with margin
    beta = (ld / np.sqrt(ll) + delta[0]) / (l0 / np.sqrt(ll) + 1.0) / ll
    rho_vec[:] -= beta * le
    # # we make sre that the inverse always exists with ϵ,
    # # if norm(ρv) - ρs) is negative (Δ is pushing towards a more positive cone)
    #     # the computation is ignored and we get the maximum value for α = 1.0
    # # else we have α = τ / norm(ρv) - ρs)
    # # we add ϵ to the denumerator to ensure strict positivity and avoid 1e-16 errors.
    alpha = 1.0
    if np.linalg.norm(rho_vec) - rho_s > 0.0:
        alpha = min(alpha, tau / (np.linalg.norm(rho_vec) - rho_s))
    return alpha


def soc_step_length(
    z: np.ndarray,
    delta: np.ndarray,
    socz: list[list[list[int]]],
    soc_delta: list[list[list[int]]],
    zsoc: list[list[list[np.ndarray]]],
    delta_soc: list[list[np.ndarray]],
    rho_vec: list[list[np.ndarray]],
    idx: list[list[list[int]]],
    tau: float = 0.99,
    verbose: bool = False,
    soc_step_length_func=None,
    return_limit: bool = False,
) -> float:
    """
    Compute the step length for second-order cone constraints.

    Assumes soc_step_length_func is provided externally and compatible with:
    soc_step_length_func(zsoc_i, delta_soc_i, rho_vec_i, idx_i, tau, verbose)
    """
    alpha = 1.0
    limit_info = None
    num_cone = len(socz)
    for i in range(num_cone):
        for j in range(len(socz[i])):
            zsoc[i][j][:] = z[socz[i][j]]
            delta_soc[i][j][:] = -delta[soc_delta[i][j]]
            alpha_i = soc_step_length_single(
                zsoc[i][j],
                delta_soc[i][j],
                rho_vec[i][j],
                idx[i][j],
                tau=tau,
                verbose=verbose,
            )
            if alpha_i < alpha:
                alpha = alpha_i
                if return_limit:
                    z_vec = np.array(zsoc[i][j], copy=True)
                    dz_vec = np.array(delta_soc[i][j], copy=True)
                    limit_info = {
                        "cone": i,
                        "side": j,
                        "alpha": float(alpha_i),
                        "z0": float(z_vec[0]) if z_vec.size else None,
                        "z_norm": (
                            float(np.linalg.norm(z_vec[1:])) if z_vec.size > 1 else None
                        ),
                        "dz0": float(dz_vec[0]) if dz_vec.size else None,
                        "dz_norm": (
                            float(np.linalg.norm(dz_vec[1:]))
                            if dz_vec.size > 1
                            else None
                        ),
                        "z": z_vec.tolist(),
                        "dz": dz_vec.tolist(),
                    }
    if return_limit:
        return alpha, limit_info
    return alpha


def centering(
    z: np.ndarray,
    dz: np.ndarray,
    ort_z: List[List[int]],
    ort_dz: List[List[int]],
    soc_z: List[List[List[int]]],
    soc_dz: List[List[List[int]]],
    zort: List[np.ndarray],
    dzort: List[np.ndarray],
    zsoc: List[List[np.ndarray]],
    dzsoc: List[List[np.ndarray]],
    alpha_aff: float,
) -> Tuple[float, float]:
    mu = 0.0
    mu_aff = 0.0
    # print("ort_z", ort_z[0])
    n = 0 if len(ort_z[0]) == 0 else len(ort_z[0])

    for cone in soc_z:
        if cone[0].size > 0:
            n += int(cone[0].size)

    zo1 = z[ort_z[0]]
    zo2 = z[ort_z[1]]
    do1 = dz[ort_dz[0]]
    do2 = dz[ort_dz[1]]
    mu = np.dot(zo1, zo2)

    dzort[0][:] = -alpha_aff * do1 + zo1
    dzort[1][:] = -alpha_aff * do2 + zo2
    mu_aff = np.dot(dzort[0], dzort[1])

    for i in range(len(soc_z)):
        zs1 = z[soc_z[i][0]]
        zs2 = z[soc_z[i][1]]
        ds1 = dz[soc_dz[i][0]]
        ds2 = dz[soc_dz[i][1]]

        dzsoc[i][0][:] = -alpha_aff * ds1 + zs1
        dzsoc[i][1][:] = -alpha_aff * ds2 + zs2

        mu += np.dot(zs1, zs2)
        mu_aff += np.dot(dzsoc[i][0], dzsoc[i][1])

    mu /= n
    mu_aff /= n
    sigma = min(max(mu_aff / mu, 0.0), 1.0) ** 3
    return mu, sigma


def general_cone_product(a, z, s, reset=False):
    """
    In-place cone product update: a += cone_product(z, s)
    """
    if reset:
        a[:] = 0.0

    n = len(a)
    a[0] += np.dot(z, s)
    for i in range(1, n):
        a[i] += z[0] * s[i] + s[0] * z[i]

    return a


def general_correction_term(
    residual, delta, ort_res_idx, soc_res_idx, soc_idx, ort_delta, soc_delta
):

    num_ort = len(ort_delta)
    if num_ort <= 2:
        do1 = delta[ort_delta[0]].flatten()  # shape (4,)
        do2 = delta[ort_delta[1]].flatten()  # shape (4,)
        residual[ort_res_idx] += do1 * do2  # .reshape(-1, 1)  # shape (4,1)
    else:
        for i in range(num_ort):
            do1 = delta[ort_delta[i][0]]
            do2 = delta[ort_delta[i][1]]
            residual[ort_res_idx[i]] += do1 * do2

    # NO soc for linear env
    num_cone = len(soc_idx)
    # residual = residual.T
    # residual = residual[0]
    for i in range(num_cone):
        ds1 = delta[soc_delta[i][0]]
        ds2 = delta[soc_delta[i][1]]
        res_soc = residual[soc_idx[i]]
        res_soc = general_cone_product(res_soc, ds2, ds1, reset=False)
        residual[soc_idx[i]] = res_soc
    return residual

