import numpy as np
from abc import ABC
from src.helper import friction_dim
from src.simulator.trajectory import update_theta

# from newton import delta


def delta(Delta_x, x, x_ref):
    Delta_x[:] = x - x_ref
    return


class NewtonResidual(ABC):
    pass


class NewtonResidualConfigurationForce(NewtonResidual):
    def __init__(self, r, q2, u1, gamma1, b1, rd, rI, q0, q1):
        self.r = r
        self.q2 = q2
        self.u1 = u1
        self.gamma1 = gamma1
        self.b1 = b1
        self.rd = rd
        self.rI = rI
        self.q0 = q0
        self.q1 = q1


def newton_residual_configuration_force(model, env, H):
    nq = model.nq
    nu = model.nu
    nc = model.nc
    nb = nc * friction_dim(env)
    nd = nq + nc + nb
    nr = nq + nu + nc + nb

    r = np.zeros(H * (nr + nd))
    q_off = nu + nc + nb

    u1 = []
    gamma1 = []
    b1 = []
    q2 = []
    rI = []
    rd = []
    q0 = []
    q1 = []

    for t in range(H):
        base = t * nr
        u1.append(r[base : base + nu])
        gamma1.append(r[base + nu : base + nu + nc])
        b1.append(r[base + nu + nc : base + nu + nc + nb])
        q2.append(r[base + q_off : base + q_off + nq])
        # rI order follows dual vector layout [q; gamma; b]
        rI.append(
            np.concatenate(
                [
                    q2[-1],
                    gamma1[-1],
                    b1[-1],
                ]
            )
        )
        rd_base = H * nr + t * nd
        rd.append(r[rd_base : rd_base + nd])

    for t in range(2, H):
        base = (t - 2) * nr + q_off
        q0.append(r[base : base + nq])
    for t in range(1, H):
        base = (t - 1) * nr + q_off
        q1.append(r[base : base + nq])

    return NewtonResidualConfigurationForce(r, q2, u1, gamma1, b1, rd, rI, q0, q1)


class NewtonResidualConfiguration:
    def __init__(self, r, q2, u1, rd, rI, q0, q1):
        self.r = r
        self.q2 = q2
        self.u1 = u1
        self.rd = rd
        self.rI = rI
        self.q0 = q0
        self.q1 = q1


def newton_residual_configuration(model, env, H):
    nq = model.nq
    nu = model.nu
    nc = model.nc
    nd = nq
    nr = nq + nu

    r = np.zeros(H * (nr + nd))

    q2 = []
    u1 = []
    rI = []
    rd = []
    q0 = []
    q1 = []

    for t in range(H):
        base = t * nr
        u1.append(r[base : base + nu])  # view
        q2.append(r[base + nu : base + nu + nq])  # view
        rI.append(r[base + nu : base + nu + nq])  # same as q2

        rd_start = H * nr + t * nd
        rd.append(r[rd_start : rd_start + nd])  # view

    for t in range(2, H):
        q0.append(r[(t - 2) * nr + nu : (t - 2) * nr + nu + nq])  # view
    for t in range(1, H):
        q1.append(r[(t - 1) * nr + nu : (t - 1) * nr + nu + nq])  # view

    return NewtonResidualConfiguration(r, q2, u1, rd, rI, q0, q1)


def newton_residual(model, env, H, mode="configurationforce"):
    if mode == "configurationforce":
        return newton_residual_configuration_force(model, env, H)
    elif mode == "configuration":
        return newton_residual_configuration(model, env, H)
    else:
        raise ValueError("mode not implemented")


def residual(res, core, nu, im_traj, traj, ref_traj, window):
    # store residual of KKT system
    res.r[:] = 0.0
    gradient(res, core.obj, core, traj, ref_traj)
    # nu : lagrangu dual?
    # ∇x​J(x)+t∑​(∂dt​/∂x)⊤νt​=0
    for i, t in enumerate(window[:-2]):
        if i >= 2:
            np.add(res.q2[i - 2], nu[i] @ im_traj.delta_q0[t], out=res.q2[i - 2])
        if i >= 1:
            np.add(res.q2[i - 1], nu[i] @ im_traj.delta_q1[t], out=res.q2[i - 1])
        np.add(res.u1[i], nu[i] @ im_traj.delta_u1[t], out=res.u1[i])
        res.rd[i] += im_traj.d[t]  # primal residual of dynamic constraints
        if hasattr(res, "gamma1") and hasattr(res, "b1"):
            nq = len(res.q2[i])
            nc = len(res.gamma1[i])
            res.q2[i] -= nu[i][:nq]
            res.gamma1[i] -= nu[i][nq : nq + nc]
            res.b1[i] -= nu[i][nq + nc :]
        else:
            res.q2[i] -= nu[i]


def update_traj(traj_cand, traj, nu_cand, nu, Delta, alpha):
    H = traj_cand.H

    for t in range(H):
        traj_cand.q[t + 2] = traj.q[t + 2] - alpha * Delta.q2[t]
        traj_cand.u[t] = traj.u[t] - alpha * Delta.u1[t]

        nu_cand[t] = nu[t] - alpha * Delta.rd[t]

    # Configuration mode updates q/u only; keep gamma/b blocks untouched.
    for t in range(1, traj_cand.H + 1):
        traj_cand.z[t - 1][traj_cand.iq2] = traj_cand.q[t + 1]
    update_theta(traj_cand)


def gradient(res, obj, core, traj, ref_traj):
    """
    Compute gradients for different residual and objective types.

    Parameters:
    - res: Either NewtonResidualConfigurationForce or NewtonResidualConfiguration
    - obj: Either TrackingObjective or TrackingVelocityObjective
    - core: Object containing Delta_q, Delta_u, Delta_gamma, Delta_b
    - traj: Trajectory object with q, u, gamma, b, and H
    - ref_traj: Reference trajectory object with q, u, gamma, b
    """
    # Check if res supports force-related fields (gamma1, b1)
    has_force = hasattr(res, "gamma1") and hasattr(res, "b1")

    # Check if obj is a TrackingVelocityObjective (has v and v_target)
    is_velocity_obj = hasattr(obj, "v") and hasattr(obj, "v_target")

    for t in range(traj.H):
        # Compute deltas
        delta_q_ref = ref_traj.q[t + 2]
        if is_velocity_obj:
            delta_q_ref = [
                delta_q_ref[i] + obj.q_target[t][i] for i in range(len(delta_q_ref))
            ]

        delta(core.Delta_q[t], traj.q[t + 2], delta_q_ref)
        delta(core.Delta_u[t], traj.u[t], ref_traj.u[t])

        if has_force:
            delta(core.Delta_gamma[t], traj.gamma[t], ref_traj.gamma[t])
            delta(core.Delta_b[t], traj.b[t], ref_traj.b[t])

        # Update residuals
        res.q2[t] += obj.q[t] @ core.Delta_q[t]
        res.u1[t] += obj.u[t] @ core.Delta_u[t]

        if has_force:
            res.gamma1[t] += obj.gamma[t] @ core.Delta_gamma[t]
            res.b1[t] += obj.b[t] @ core.Delta_b[t]

        # Handle velocity-related terms if obj is TrackingVelocityObjective
        if is_velocity_obj:
            res.q2[t] += obj.v[t] @ traj.q[t + 2]
            res.q2[t] -= obj.v[t] @ traj.q[t + 1]
            res.q2[t] -= obj.v[t] @ obj.v_target[t] if hasattr(obj, "v_target") else 0

            if t > 0:
                res.q2[t - 1] -= obj.v[t] @ traj.q[t + 2]
                res.q2[t - 1] += obj.v[t] @ traj.q[t + 1]
                res.q2[t - 1] += (
                    obj.v[t] @ obj.v_target[t] if hasattr(obj, "v_target") else 0
                )

