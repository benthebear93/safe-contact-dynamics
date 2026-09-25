import numpy as np
from scipy import sparse
from abc import ABC
from src.helper import friction_dim
from .objective import TrackingObjective, TrackingVelocityObjective
from numpy.lib.stride_tricks import as_strided


class NewtonJacobian(ABC):
    pass


class NewtonJacobianConfigurationForce:
    def __init__(
        self,
        R,
        obj_q2,
        obj_u1,
        obj_gamma1,
        obj_b1,
        obj_q1q2,
        obj_q2q1,
        IV_rows,
        IV_cols,
        ITV_rows,
        ITV_cols,
        q0,
        q0T,
        q1,
        q1T,
        u1,
        u1T,
        reg_pr,
        reg_du,
    ):
        self.R = R
        self.obj_q2 = obj_q2
        self.obj_u1 = obj_u1
        self.obj_gamma1 = obj_gamma1
        self.obj_b1 = obj_b1
        self.obj_q1q2 = obj_q1q2
        self.obj_q2q1 = obj_q2q1
        self.IV_rows = IV_rows  # List of row index arrays for IV positions
        self.IV_cols = IV_cols  # List of col index arrays for IV positions
        self.ITV_rows = ITV_rows
        self.ITV_cols = ITV_cols
        self.q0 = q0
        self.q0T = q0T
        self.q1 = q1
        self.q1T = q1T
        self.u1 = u1
        self.u1T = u1T
        self.reg_pr = reg_pr
        self.reg_du = reg_du


def newton_jacobian_configuration_force(model, env, H):
    nq = model.nq
    nu = model.nu
    nw = model.nw
    nc = model.nc
    nb = nc * friction_dim(env)
    nd = nq + nc + nb
    nr = nq + nu + nc + nb

    iu = np.arange(nu)
    igamma = np.arange(nu, nu + nc)
    ib = np.arange(nu + nc, nu + nc + nb)
    iq = np.arange(nu + nc + nb, nu + nc + nb + nq)

    iz = np.concatenate([iq, igamma, ib])
    itheta = np.concatenate([iq - 2 * nr, iq - nr, iu])

    inu = np.arange(nd)

    R = sparse.lil_matrix((H * (nr + nd), H * (nr + nd)))

    obj_u1 = np.asarray(
        [
            R[
                (t - 1) * nr + iu[0] : (t - 1) * nr + iu[-1] + 1,
                (t - 1) * nr + iu[0] : (t - 1) * nr + iu[-1] + 1,
            ]
            for t in range(1, H + 1)
        ]
    )
    obj_gamma1 = np.asarray(
        [
            R[
                (t - 1) * nr + igamma[0] : (t - 1) * nr + igamma[-1] + 1,
                (t - 1) * nr + igamma[0] : (t - 1) * nr + igamma[-1] + 1,
            ]
            for t in range(1, H + 1)
        ]
    )
    obj_b1 = np.asarray(
        [
            R[
                (t - 1) * nr + ib[0] : (t - 1) * nr + ib[-1] + 1,
                (t - 1) * nr + ib[0] : (t - 1) * nr + ib[-1] + 1,
            ]
            for t in range(1, H + 1)
        ]
    )
    obj_q2 = np.asarray(
        [
            R[
                (t - 1) * nr + iq[0] : (t - 1) * nr + iq[-1] + 1,
                (t - 1) * nr + iq[0] : (t - 1) * nr + iq[-1] + 1,
            ]
            for t in range(1, H + 1)
        ]
    )
    obj_q1q2 = np.asarray(
        [
            R[
                (t - 1) * nr + iq[0] : (t - 1) * nr + iq[-1] + 1,
                t * nr + iq[0] : t * nr + iq[-1] + 1,
            ]
            for t in range(1, H)
        ]
    )
    obj_q2q1 = np.asarray(
        [
            R[
                t * nr + iq[0] : t * nr + iq[-1] + 1,
                (t - 1) * nr + iq[0] : (t - 1) * nr + iq[-1] + 1,
            ]
            for t in range(1, H)
        ]
    )

    IV_rows = [(t - 1) * nr + iz for t in range(1, H + 1)]
    IV_cols = [H * nr + (t - 1) * nd + inu for t in range(1, H + 1)]
    ITV_rows = [H * nr + (t - 1) * nd + inu for t in range(1, H + 1)]
    ITV_cols = [(t - 1) * nr + iz for t in range(1, H + 1)]

    q0 = np.asarray(
        [
            R[
                H * nr + (t - 1) * nd + inu[0] : H * nr + (t - 1) * nd + inu[-1] + 1,
                (t - 3) * nr + iq[0] : (t - 3) * nr + iq[-1] + 1,
            ]
            for t in range(3, H + 1)
        ]
    )
    q0T = np.asarray(
        [
            R[
                (t - 3) * nr + iq[0] : (t - 3) * nr + iq[-1] + 1,
                H * nr + (t - 1) * nd + inu[0] : H * nr + (t - 1) * nd + inu[-1] + 1,
            ]
            for t in range(3, H + 1)
        ]
    )
    q1 = np.asarray(
        [
            R[
                H * nr + (t - 1) * nd + inu[0] : H * nr + (t - 1) * nd + inu[-1] + 1,
                (t - 2) * nr + iq[0] : (t - 2) * nr + iq[-1] + 1,
            ]
            for t in range(2, H + 1)
        ]
    )
    q1T = np.asarray(
        [
            R[
                (t - 2) * nr + iq[0] : (t - 2) * nr + iq[-1] + 1,
                H * nr + (t - 1) * nd + inu[0] : H * nr + (t - 1) * nd + inu[-1] + 1,
            ]
            for t in range(2, H + 1)
        ]
    )
    u1 = np.asarray(
        [
            R[
                H * nr + (t - 1) * nd + inu[0] : H * nr + (t - 1) * nd + inu[-1] + 1,
                (t - 1) * nr + iu[0] : (t - 1) * nr + iu[-1] + 1,
            ]
            for t in range(1, H + 1)
        ]
    )
    u1T = np.asarray(
        [
            R[
                (t - 1) * nr + iu[0] : (t - 1) * nr + iu[-1] + 1,
                H * nr + (t - 1) * nd + inu[0] : H * nr + (t - 1) * nd + inu[-1] + 1,
            ]
            for t in range(1, H + 1)
        ]
    )
    reg_pr = R[0 : H * nr, 0 : H * nr]
    reg_du = R[H * nr : H * nr + H * nd, H * nr : H * nr + H * nd]

    # Optional: Convert to CSC after all initial sets if needed
    # R = R.tocsc()

    return NewtonJacobianConfigurationForce(
        R,
        obj_q2,
        obj_u1,
        obj_gamma1,
        obj_b1,
        obj_q1q2,
        obj_q2q1,
        IV_rows,
        IV_cols,
        ITV_rows,
        ITV_cols,
        q0,
        q0T,
        q1,
        q1T,
        u1,
        u1T,
        reg_pr,
        reg_du,
    )


class NewtonJacobianConfiguration(NewtonJacobian):
    def __init__(
        self,
        R,
        obj_q2,
        obj_u1,
        obj_q1q2,
        obj_q2q1,
        IV,
        ITV,
        q0,
        q0T,
        q1,
        q1T,
        u1,
        u1T,
        reg_pr,
        reg_du,
    ):
        self.R = R
        self.obj_q2 = obj_q2
        self.obj_u1 = obj_u1
        self.obj_q1q2 = obj_q1q2
        self.obj_q2q1 = obj_q2q1
        self.IV = IV
        self.ITV = ITV
        self.q0 = q0
        self.q0T = q0T
        self.q1 = q1
        self.q1T = q1T
        self.u1 = u1
        self.u1T = u1T
        self.reg_pr = reg_pr
        self.reg_du = reg_du


def newton_jacobian_configuration(model, env, H):
    nq = model.nq
    nu = model.nu
    # nw = model.nw
    # nc = model.nc
    nd = nq
    nr = nq + nu  # 6

    # FIXME newton
    # iz = np.array([2, 3, 4, 5])  # 4 row indices
    # inu = np.array([0, 1, 2, 3])  # 1 column index

    R = np.zeros((H * (nr + nd), H * (nr + nd)))

    obj_u1 = [
        R[
            (t - 1) * nr + 0 : (t - 1) * nr + nu, (t - 1) * nr + 0 : (t - 1) * nr + nu
        ]  # 0:2, 0:2, 6:8, 6:8
        for t in range(1, H + 1)
    ]  # d^2/du_t^2

    obj_q2 = [
        R[
            (t - 1) * nr + nu : t * nr, (t - 1) * nr + nu : t * nr
        ]  # 2:6, 2:6, 8:12, 8:12
        for t in range(1, H + 1)
    ]  # d^2/dq_t^2

    obj_q1q2 = [
        R[(t - 1) * nr + nu : t * nr, t * nr + nu : (t + 1) * nr]
        for t in range(1, H)  # 2:6, 8:12, 8:12, 14:18
    ]

    obj_q2q1 = [
        R[t * nr + nu : (t + 1) * nr, (t - 1) * nr + nu : t * nr]
        for t in range(1, H)  # 8:12, 2:6, 14:18, 8:12
    ]

    IV = [
        as_strided(
            R[
                (t - 1) * nr + 2 : (t - 1) * nr + 6,
                H * nr + (t - 1) * nd : H * nr + (t - 1) * nd + nd,
            ],
            shape=(nd,),
            strides=(R.strides[0] + R.strides[1],),
        ).reshape(nd, 1)
        for t in range(1, H + 1)
    ]  # Identity View?
    # FIXME currently using as_strided for mimic the Cartesian

    ITV = [
        as_strided(
            R[H * nr + (t - 1) * nd : H * nr + t * nd, (t - 1) * nr + nu : t * nr],
            shape=(nd,),
            strides=(R.strides[0] + R.strides[1],),
        ).reshape(1, nd)
        for t in range(1, H + 1)
    ]  # Identity Transpose View?
    # FIXME currently using as_strided for mimic the Cartesian

    q0 = [
        R[H * nr + (t - 1) * nd : H * nr + t * nd, (t - 3) * nr + nu : (t - 2) * nr]
        for t in range(3, H + 1)
    ]

    q0T = [
        R[(t - 3) * nr + nu : (t - 2) * nr, H * nr + (t - 1) * nd : H * nr + t * nd]
        for t in range(3, H + 1)
    ]

    q1 = [
        R[H * nr + (t - 1) * nd : H * nr + t * nd, (t - 2) * nr + nu : (t - 1) * nr]
        for t in range(2, H + 1)
    ]
    q1T = [
        R[(t - 2) * nr + nu : (t - 1) * nr, H * nr + (t - 1) * nd : H * nr + t * nd]
        for t in range(2, H + 1)
    ]

    u1 = [
        R[H * nr + (t - 1) * nd : H * nr + t * nd, (t - 1) * nr + 0 : (t - 1) * nr + nu]
        for t in range(1, H + 1)
    ]
    u1T = [
        R[(t - 1) * nr + 0 : (t - 1) * nr + nu, H * nr + (t - 1) * nd : H * nr + t * nd]
        for t in range(1, H + 1)
    ]

    reg_pr = R[: H * nr, : H * nr]  # primal

    reg_du = [
        R[H * nr + i : H * nr + i + 1, H * nr + i : H * nr + i + 1]
        for i in range(H * nd)
    ]  # dual

    return NewtonJacobianConfiguration(
        R,
        obj_q2,
        obj_u1,
        obj_q1q2,
        obj_q2q1,
        IV,
        ITV,
        q0,
        q0T,
        q1,
        q1T,
        u1,
        u1T,
        reg_pr,
        reg_du,
    )


def newton_jacobian(model, env, H, mode="configurationforce"):
    if mode == "configurationforce":
        return newton_jacobian_configuration_force(model, env, H)
    elif mode == "configuration":
        return newton_jacobian_configuration(model, env, H)
    else:
        raise ValueError("mode not implemented")


def initialize_jacobian(jac, obj, H, update_hessian=True):
    # Zero out the entire dense matrix in-place
    jac.R[:, :] = 0.0

    # Update Hessian blocks if needed
    if update_hessian:
        hessian(jac, obj)

    # Implicit dynamics modification: in-place subtraction.
    # configuration uses jac.IV/jac.ITV views, while configurationforce stores
    # index pairs (IV_rows/IV_cols, ITV_rows/ITV_cols).
    if hasattr(jac, "IV") and hasattr(jac, "ITV"):
        for t in range(H):
            jac.IV[t][:] -= 1.0
            jac.ITV[t][:] -= 1.0
    elif (
        hasattr(jac, "IV_rows")
        and hasattr(jac, "IV_cols")
        and hasattr(jac, "ITV_rows")
        and hasattr(jac, "ITV_cols")
    ):
        for t in range(H):
            rows = np.asarray(jac.IV_rows[t], dtype=int).reshape(-1)
            cols = np.asarray(jac.IV_cols[t], dtype=int).reshape(-1)
            for r, c in zip(rows, cols):
                jac.R[int(r), int(c)] -= 1.0
            rows_t = np.asarray(jac.ITV_rows[t], dtype=int).reshape(-1)
            cols_t = np.asarray(jac.ITV_cols[t], dtype=int).reshape(-1)
            for r, c in zip(rows_t, cols_t):
                jac.R[int(r), int(c)] -= 1.0
    return None


def update_jacobian(jac, im_traj, obj, H, beta, window):
    nq = jac.obj_q2[0].shape[0]
    np.set_printoptions(suppress=True, precision=11)

    for i, t in enumerate(window[:-2], start=0):  # Python: 0-based
        if i >= 2:
            jac.q0[i - 2][:] += im_traj.delta_q0[t]
            jac.q0T[i - 2][:] += im_traj.delta_q0[t].T

        if i >= 1:
            jac.q1[i - 1][:] += im_traj.delta_q1[t]
            jac.q1T[i - 1][:] += im_traj.delta_q1[t].T

        jac.u1[i][:] += im_traj.delta_u1[t]
        jac.u1T[i][:] += im_traj.delta_u1[t].T
        # Dual regularization (scalar update, assumed to be 1D slice from R)
        kappa = im_traj.ip[t].kappa[0]  # Julia index 1 → Python 0
        nreg = (
            int(jac.reg_du.shape[0])
            if hasattr(jac.reg_du, "shape")
            else len(jac.reg_du)
        )
        for j in range(nreg):
            try:
                jac.reg_du[j, j] -= beta * kappa
            except Exception:
                jac.reg_du[j] -= beta * kappa

    return None


def jacobian(jac, im_traj, obj, H, beta, window, update_hessian=True):
    initialize_jacobian(jac, obj, H, update_hessian=update_hessian)
    update_jacobian(jac, im_traj, obj, H, beta, window)


def hessian(jac, obj):
    if isinstance(jac, NewtonJacobianConfigurationForce) and isinstance(
        obj, TrackingObjective
    ):
        for t in range(len(obj.u)):
            jac.obj_q2[t] += obj.q[t]
            jac.obj_u1[t] += obj.u[t]
            jac.obj_gamma1[t] += obj.gamma[t]
            jac.obj_b1[t] += obj.b[t]
    elif isinstance(jac, NewtonJacobianConfiguration) and isinstance(
        obj, TrackingObjective
    ):
        for t in range(len(obj.u)):
            jac.obj_q2[t] += obj.q[t]
            jac.obj_u1[t] += obj.u[t]
    elif isinstance(jac, NewtonJacobianConfigurationForce) and isinstance(
        obj, TrackingVelocityObjective
    ):
        for t in range(len(obj.u)):
            jac.obj_q2[t] += obj.q[t]
            jac.obj_u1[t] += obj.u[t]
            jac.obj_gamma1[t] += obj.gamma[t]
            jac.obj_b1[t] += obj.b[t]

            jac.obj_q2[t] += obj.v[t]
            if t == 0:
                continue
            jac.obj_q2[t - 1] += obj.v[t]
            jac.obj_q1q2[t - 1] -= obj.v[t]
            jac.obj_q2q1[t - 1] -= obj.v[t]
    elif isinstance(jac, NewtonJacobianConfiguration) and isinstance(
        obj, TrackingVelocityObjective
    ):
        for t in range(len(obj.u)):
            jac.obj_q2[t] += obj.q[t]
            jac.obj_u1[t] += obj.u[t]

            jac.obj_q2[t] += obj.v[t]
            if t == 0:
                continue
            jac.obj_q2[t - 1] += obj.v[t]
            jac.obj_q1q2[t - 1] -= obj.v[t]
            jac.obj_q2q1[t - 1] -= obj.v[t]
