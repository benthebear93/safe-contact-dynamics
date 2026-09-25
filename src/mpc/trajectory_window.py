"""Copy and advance MPC trajectories for the next optimization step."""

import numpy as np


def advance_trajectory(traj, cache, stride):
    """Shift one step and extend the terminal states by the reference stride."""
    _rotate_trajectory(traj, cache)
    _extend_terminal_states(traj, stride)


def _rotate_trajectory(traj, cache):
    H = traj.H

    cache.q[0] = traj.q[0].copy()
    cache.u[0] = traj.u[0].copy()
    cache.w[0] = traj.w[0].copy()
    cache.gamma[0] = traj.gamma[0].copy()
    cache.b[0] = traj.b[0].copy()
    cache.z[0] = traj.z[0].copy()
    cache.theta[0] = traj.theta[0].copy()

    for t in range(1, H + 2):
        traj.q[t - 1] = traj.q[t].copy()

    for t in range(1, H):
        traj.u[t - 1] = traj.u[t].copy()
        traj.w[t - 1] = traj.w[t].copy()
        traj.gamma[t - 1] = traj.gamma[t].copy()
        traj.b[t - 1] = traj.b[t].copy()
        traj.z[t - 1] = traj.z[t].copy()
        traj.theta[t - 1] = traj.theta[t].copy()

    traj.q[-1] = cache.q[0].copy()
    traj.u[-1] = cache.u[0].copy()
    traj.w[-1] = cache.w[0].copy()
    traj.gamma[-1] = cache.gamma[0].copy()
    traj.b[-1] = cache.b[0].copy()
    traj.z[-1] = cache.z[0].copy()
    traj.theta[-1] = cache.theta[0].copy()


def copy_reference_trajectory(traj, ref_traj):
    """Reset the working trajectory from the reference, copying each array."""
    H = ref_traj.H
    for t in range(1, H + 1):
        traj.q[t - 1] = ref_traj.q[t - 1].copy()
        traj.u[t - 1] = ref_traj.u[t - 1].copy()
        traj.w[t - 1] = ref_traj.w[t - 1].copy()
        traj.gamma[t - 1] = ref_traj.gamma[t - 1].copy()
        traj.b[t - 1] = ref_traj.b[t - 1].copy()
        traj.z[t - 1] = ref_traj.z[t - 1].copy()
        traj.theta[t - 1] = ref_traj.theta[t - 1].copy()
    traj.q[H] = ref_traj.q[H].copy()
    traj.q[H + 1] = ref_traj.q[H + 1].copy()


def _extend_terminal_states(traj, stride):
    """Extend the last two configurations and synchronize packed solver data."""
    H = traj.H
    for t in range(H + 1, H + 3):
        traj.q[t - 1] = traj.q[t - H - 1].copy()
        traj.q[t - 1] += stride

        tau = t - 2
        traj.z[tau - 1][traj.iq2] = traj.q[tau + 1]

        traj.theta[tau - 1][traj.iq0] = traj.q[tau - 1]
        traj.theta[tau - 1][traj.iq1] = traj.q[tau]

        tau = t - 3
        traj.theta[tau - 1][traj.iq0] = traj.q[tau - 1]
        traj.theta[tau - 1][traj.iq1] = traj.q[tau]


def compute_reference_stride(model, traj):
    """Return the reference's net x displacement as a configuration offset."""
    stride = np.zeros(model.nq)
    stride[0] = traj.q[-2][0] - traj.q[0][0]
    return stride
