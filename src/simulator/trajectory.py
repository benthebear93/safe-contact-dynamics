import numpy as np
from src.helper import num_data, num_var, nc_impact_check


class Trajectory:
    def __init__(self, model, T, nv=None, nc=None, nb=None, h=None):
        self.nq = model.nq
        self.nu = model.nu
        self.nw = model.nw
        self.nv = nv if nv is not None else model.nq
        self.nc = nc if nc is not None else model.nc
        self.nb = nb if nb is not None else model.nc  # Friction dimension
        # Optional sizes for full optimization/data vectors
        self.nz = model.num_var() if hasattr(model, "num_var") else None
        self.n_theta = num_data(model)
        self.T = T
        self.H = T
        self.h = h
        self.q = [np.zeros(self.nq) for _ in range(T + 2)]
        self.v = [np.zeros(self.nv) for _ in range(T + 1)]  # midpoint velocities
        self.u = [np.zeros(self.nu) for _ in range(T)]
        self.gamma = [np.zeros(self.nc) for _ in range(T)]  # Contact force
        self.s_gamma = [np.zeros(self.nc) for _ in range(T)]
        self.psi = [np.zeros(self.nc) for _ in range(T)]
        self.s_psi = [np.zeros(self.nc) for _ in range(T)]
        self.b = [np.zeros(self.nb) for _ in range(T)]  # Friction force
        self.s_b = [np.zeros(self.nb) for _ in range(T)]  # Friction force
        self.w = [np.zeros(self.nw) for _ in range(T)]  # External disturbances
        self.kappa = [np.zeros(self.nc) for _ in range(T)]
        # Full optimization variables and theta data (if size known)
        if self.nz is not None:
            self.z = [np.zeros(self.nz) for _ in range(T)]
        else:
            self.z = [np.array([]) for _ in range(T)]
        self.theta = [np.zeros(self.n_theta) for _ in range(T)]

        # Indices matching ContactTrajectory layout for export.
        off = 0
        self.iq0 = np.arange(off, off + self.nq)
        off += self.nq
        self.iq1 = np.arange(off, off + self.nq)
        off += self.nq
        self.iu1 = np.arange(off, off + self.nu)
        off += self.nu
        self.iw1 = np.arange(off, off + self.nw)
        off += self.nw
        off = 0
        self.iq2 = np.arange(off, off + self.nq)
        off += self.nq
        self.igamma1 = np.arange(off, off + self.nc)
        off += self.nc
        self.ib1 = np.arange(off, off + self.nb)
        off += self.nb

    def reset(self):
        T = len(self.u)
        for t in range(T):
            self.q[t] = np.zeros(self.nq)
            self.v[t] = np.zeros(self.nv)
            self.u[t] = np.zeros(self.nu)
            self.gamma[t] = np.zeros(self.nc)
            self.s_gamma[t] = np.zeros(self.nc)
            self.psi[t] = np.zeros(self.nc)
            self.s_psi[t] = np.zeros(self.nc)
            self.b[t] = np.zeros(self.nb)
            self.s_b[t] = np.zeros(self.nb)
            self.w[t] = np.zeros(self.nw)
            self.kappa[t] = 1e-8 * np.ones(self.nc)
            if self.nz is not None:
                self.z[t] = np.zeros(self.nz)
            self.theta[t] = np.zeros(self.n_theta)
        self.q[T] = np.zeros(self.nq)
        self.q[T + 1] = np.zeros(self.nq)
        self.v[T] = np.zeros(self.nv)


class GradientTrajectory:
    def __init__(self, model, T, nc=None, nb=None):
        nq = model.nq
        nu = model.nu
        nc = nc if nc is not None else model.nc
        nb = nb if nb is not None else model.nc
        self.dq3_dq1 = [np.zeros((nq, nq)) for _ in range(T)]
        self.dq3_dq2 = [np.zeros((nq, nq)) for _ in range(T)]
        self.dq3_dv1 = [np.zeros((nq, nq)) for _ in range(T)]
        self.dq3_du1 = [np.zeros((nq, nu)) for _ in range(T)]

        self.dgamma1_dq1 = [np.zeros((nc, nq)) for _ in range(T)]
        self.dgamma1_dq2 = [np.zeros((nc, nq)) for _ in range(T)]
        self.dgamma1_dv1 = [np.zeros((nc, nq)) for _ in range(T)]
        self.dgamma1_du1 = [np.zeros((nc, nu)) for _ in range(T)]

        self.db1_dq1 = [np.zeros((nb, nq)) for _ in range(T)]
        self.db1_dq2 = [np.zeros((nb, nq)) for _ in range(T)]
        self.db1_dv1 = [np.zeros((nb, nq)) for _ in range(T)]
        self.db1_du1 = [np.zeros((nb, nu)) for _ in range(T)]

    def reset(self):
        T = len(self.dq3_dq1)
        for t in range(T):
            self.dq3_dq1[t].fill(0.0)
            self.dq3_dq2[t].fill(0.0)
            self.dq3_dv1[t].fill(0.0)
            self.dq3_du1[t].fill(0.0)

            self.dgamma1_dq1[t].fill(0.0)
            self.dgamma1_dq2[t].fill(0.0)
            self.dgamma1_dv1[t].fill(0.0)
            self.dgamma1_du1[t].fill(0.0)

            self.db1_dq1[t].fill(0.0)
            self.db1_dq2[t].fill(0.0)
            self.db1_dv1[t].fill(0.0)
            self.db1_du1[t].fill(0.0)


class ContactTraj:
    def __init__(
        self,
        H,
        h,
        kappa,
        q,
        u,
        w,
        gamma,
        b,
        z,
        theta,
        iq0,
        iq1,
        iu1,
        iw1,
        iq2,
        igamma1,
        ib1,
    ):
        self.H = H
        self.h = h
        self.kappa = kappa
        self.q = q
        self.u = u
        self.w = w
        self.gamma = gamma
        self.b = b
        self.z = z
        self.theta = theta
        self.iq0 = iq0
        self.iq1 = iq1
        self.iu1 = iu1
        self.iw1 = iw1
        self.iq2 = iq2
        self.igamma1 = igamma1
        self.ib1 = ib1


def contact_trajectory(model, env, H, h, kappa=0.0):
    nq = model.nq
    nu = model.nu
    nw = model.nw
    nc = model.nc
    nb = nc * 4  # FIXME
    nz = num_var(model, env)
    n_theta = num_data(model)

    q = [np.zeros(nq) for _ in range(H + 2)]
    u = [np.zeros(nu) for _ in range(H)]
    w = [np.zeros(nw) for _ in range(H)]
    gamma = [np.zeros(nc) for _ in range(H)]
    b = [np.zeros(nb) for _ in range(H)]
    z = [np.zeros(nz) for _ in range(H)]
    theta = [np.append(np.zeros(n_theta - 1), h) for _ in range(H)]
    kappa = [kappa]
    off = 0
    iq0 = np.arange(off, off + nq)
    off += nq
    iq1 = np.arange(off, off + nq)
    off += nq
    iu1 = np.arange(off, off + nu)
    off += nu
    iw1 = np.arange(off, off + nw)
    off += nw
    off = 0
    iq2 = np.arange(off, off + nq)
    off += nq
    igamma1 = np.arange(off, off + nc)
    off += nc
    ib1 = np.arange(off, off + nb)
    off += nb

    return ContactTraj(
        H, h, kappa, q, u, w, gamma, b, z, theta, iq0, iq1, iu1, iw1, iq2, igamma1, ib1
    )


class ContactTrajectory:
    def __init__(self, T, h, model, env):
        self.T = T
        self.H = T
        self.h = h
        self.model = model
        self.env = env
        nb = nc_impact_check(env, model.nc)
        nq = model.nq
        nu = model.nu
        nw = model.nw
        nc = model.nc
        self.q = [np.zeros(model.nq) for _ in range(T + 2)]
        self.u = [np.zeros(model.nu) for _ in range(T)]
        self.w = [np.zeros(model.nw) for _ in range(T)]
        self.gamma = [np.zeros(model.nc) for _ in range(T)]
        self.b = [np.zeros(nb) for _ in range(T)]
        self.kappa = [np.zeros(model.nc) for _ in range(T)]
        self.z = [np.zeros(num_var(model, env)) for _ in range(T)]
        self.theta = [np.zeros(num_data(model)) for _ in range(T)]

        off = 0
        self.iq0 = np.arange(off, off + nq)
        off += nq
        self.iq1 = np.arange(off, off + nq)
        off += nq
        self.iu1 = np.arange(off, off + nu)
        off += nu
        self.iw1 = np.arange(off, off + nw)
        off += nw

        off = 0
        self.iq2 = np.arange(off, off + nq)
        off += nq
        self.igamma1 = np.arange(off, off + nc)
        off += nc
        self.ib1 = np.arange(off, off + nb)
        off += nb


def update_z_step(traj, t):
    if 1 <= t <= traj.H:
        traj.z[t - 1][traj.iq2] = traj.q[t + 1]
        traj.z[t - 1][traj.igamma1] = traj.gamma[t - 1]
        traj.z[t - 1][traj.ib1] = traj.b[t - 1]


def update_z(traj):
    for t in range(1, traj.H + 1):
        update_z_step(traj, t)


def update_theta_step(traj, t):
    if 0 <= t <= traj.H:
        traj.theta[t][traj.iq0] = traj.q[t]
        traj.theta[t][traj.iq1] = traj.q[t + 1]
        traj.theta[t][traj.iu1] = traj.u[t]
        traj.theta[t][traj.iw1] = traj.w[t]


def update_theta(traj):
    for t in range(1, traj.H):
        update_theta_step(traj, t)

