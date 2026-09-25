import copy
import numpy as np
from src.solver.interior_point_solver import (
    InteriorPointOptions,
)
from .newton import NewtonOptions, Newton, newton_solve
from .implicit_dynamics import (
    ImplicitTrajectory,
    set_implicit_trajectory,
    update,
)
from .trajectory_window import (
    advance_trajectory,
    compute_reference_stride,
    copy_reference_trajectory,
)
from src.helper import friction_dim

# np.set_printoptions(linewidth=1000, precision=5, suppress=True)


class CIMPCOptions:
    def __init__(
        self,
        altitude_update=False,
        altitude_impact_threshold=1.0,
        altitude_verbose=False,
        ip_max_time=1e5,
        gains=False,
        control_gain_scaling=100.0,
        velocity_gain_scaling=100.0,
    ):
        self.altitude_update = altitude_update
        self.altitude_impact_threshold = altitude_impact_threshold
        self.altitude_verbose = altitude_verbose
        self.ip_max_time = ip_max_time
        self.gains = gains
        self.control_gain_scaling = control_gain_scaling
        self.velocity_gain_scaling = velocity_gain_scaling


class CIMPC:
    def __init__(
        self,
        u,
        traj,
        traj_cache,
        ref_traj,
        im_traj,
        im_traj_cache,
        H,
        stride,
        altitude,
        phi,
        kappa,
        newton,
        newton_mode,
        s,
        q0,
        N_sample,
        cnt,
        window,
        opts,
        buffer_time,
        next_time_update,
        times_reference,
        total_time_reference,
    ):
        self.u = u
        self.traj = traj
        self.traj_cache = traj_cache
        self.ref_traj = ref_traj
        self.im_traj = im_traj
        self.im_traj_cache = im_traj_cache
        self.H = H
        self.stride = stride
        self.altitude = altitude
        self.phi = phi
        self.kappa = kappa
        self.newton = newton
        self.newton_mode = newton_mode
        self.s = s
        self.N_sample = N_sample
        self.cnt = cnt
        self.window = window
        self.opts = opts
        self.buffer_time = buffer_time
        self.next_time_update = next_time_update
        self.times_reference = times_reference
        self.total_time_reference = total_time_reference
        self.q0 = q0  # Added from constructor


def ci_mpc_policy(
    traj,
    s,
    obj,
    H_mpc=None,
    N_sample=1,
    kappa_mpc=None,
    mode="configurationforce",
    newton_mode="direct",
    n_opts=None,
    mpc_opts=None,
    ip_opts=None,
):
    if H_mpc is None:
        H_mpc = traj.H
    if kappa_mpc is None:
        kappa_mpc = traj.kappa[0]
    if n_opts is None:
        n_opts = NewtonOptions(
            r_tol=3e-4, max_iter=5, verbose=False
        )

    if mpc_opts is None:
        mpc_opts = CIMPCOptions()
    if ip_opts is None:
        ip_opts = InteriorPointOptions(
            gamma_reg=0.1,
            undercut=5.0,
            kappa_tol=kappa_mpc,
            r_tol=1.0e-8,
            diff_sol=True,
            solver="empty_solver",
            max_time=mpc_opts.ip_max_time,
        )

    traj = copy.deepcopy(traj)
    traj_cache = copy.deepcopy(traj)
    ref_traj = copy.deepcopy(traj)
    im_traj = ImplicitTrajectory(
        traj, s, kappa=kappa_mpc, max_time=mpc_opts.ip_max_time, opts=ip_opts, mode=mode
    )
    im_traj_cache = copy.deepcopy(im_traj)

    nq = s.model.nq
    nc = s.model.nc
    nb = nc * friction_dim(s.env)
    if mode == "configurationforce":
        nd = nq + nc + nb
    elif mode == "configuration":
        nd = nq
    else:
        raise ValueError("invalid mode")

    stride = compute_reference_stride(s.model, traj)
    altitude = np.zeros(s.model.nc)  # FIXME
    phi = np.zeros(s.model.nc)

    newton = Newton(s, H_mpc, traj.h, traj, im_traj, obj=obj, opts=n_opts)

    window = np.zeros(H_mpc + 2, dtype=int)  # H_mpc : 10

    total_time_reference = traj.h * (traj.H - 1)

    # reconnect view
    for t in range(traj.H):
        im_traj.d[t] = im_traj.ip[t].z[:nd]
        im_traj.dq2[t] = im_traj.d[t][:nq]

    times_reference = np.linspace(0, total_time_reference, traj.H)

    return CIMPC(
        np.zeros(s.model.nu),
        traj,
        traj_cache,
        ref_traj,
        im_traj,
        im_traj_cache,
        H_mpc,
        stride,
        altitude,
        phi,
        [kappa_mpc],
        newton,
        newton_mode,
        s,
        copy.deepcopy(ref_traj.q[0]),
        N_sample,
        [N_sample],
        window,
        mpc_opts,
        0.0,
        0.0,
        times_reference,
        total_time_reference,
    )


def policy(p, traj, t):
    if t == 0:
        p.cnt[0] = p.N_sample
        p.q0[:] = p.ref_traj.q[0]
        p.altitude[:] = 0.0
        copy_reference_trajectory(p.traj, p.ref_traj)
        set_implicit_trajectory(p.im_traj, p.im_traj_cache)
        reset_window(p.window)

    if p.cnt[0] == p.N_sample:
        # optimize
        q1 = traj.q[t + 1]
        newton_solve(
            p.newton, p.s, p.q0, q1, p.window, p.im_traj, p.traj, warm_start=(t > 0)
        )
        update(p.im_traj, p.traj, p.s, p.kappa[0], p.traj.H)
        # shift trajectory
        advance_trajectory(p.traj, p.traj_cache, p.stride)
        # update
        update_window(p.window, p.ref_traj.H)
        p.q0[:] = q1
        # reset count
        p.cnt[0] = 0

    p.cnt[0] += 1

    # scale by N_sample so repeated application over finer sim steps preserves impulse
    p.u[:] = p.newton.traj.u[0]
    p.u /= p.N_sample

    return p.u


def reset_window(window):
    n = len(window)
    for i in range(n):
        window[i] = i
    return window


def update_window(window, max_window):
    n = len(window)
    for i in range(n):
        window[i] += 1
        if window[i] > max_window - 1:
            window[i] = 1
    return window
