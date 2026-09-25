import copy
import numpy as np
from src.solver.interior_point_solver import (
    InteriorPointOptions,
    interior_point_solve,
    initialize_interior_point_mpc,
    Euclidean,
)
from src.solver.indices import index_q2
from src.helper import friction_dim
from .linearized_step import linearized_step
from .linearized_solver import rlin, RZLin, rzlin, r_theta, RLin, RThetaLin
from .dynamics_context import z_initialize_warmstart


class ImplicitTrajectory:
    def __init__(
        self,
        ref_traj,
        s,
        kappa=None,
        max_time=1e5,
        mode="configuration",
        opts=None,
    ):
        if kappa is None:
            kappa = ref_traj.kappa
        if opts is None:
            opts = InteriorPointOptions(
                undercut=5.0,
                gamma_reg=0.1,
                kappa_tol=kappa[0],
                r_tol=1.0e-8,
                diff_sol=True,
                solver="empty_solver",
                max_time=max_time,
            )

        model = s.model
        env = s.env

        iq2 = index_q2(model, env, quat=False)

        H = ref_traj.H
        nq = model.nq
        nu = model.nu
        nc = model.nc
        nb = nc * friction_dim(env)
        if mode == "configurationforce":
            nd = nq + nc + nb
        elif mode == "configuration":
            nd = nq
        else:
            raise ValueError("invalid mode")

        self.lin = [
            linearized_step(s, ref_traj.z[t], ref_traj.theta[t], kappa)
            for t in range(H)
        ]

        # ip for MPC
        ip = [
            initialize_interior_point_mpc(
                copy.deepcopy(ref_traj.z[t]),
                copy.deepcopy(ref_traj.theta[t]),
                space=Euclidean(len(ref_traj.z[t])),
                idx=model.indices_optimization(),
                r_func=rlin,
                rz_func=rzlin,
                rtheta_func=r_theta,
                r=RLin(
                    s,
                    self.lin[t].z,
                    self.lin[t].theta,
                    self.lin[t].r.T,
                    self.lin[t].rz,
                    self.lin[t].r_theta,
                    verbose=False,
                ),
                rz=RZLin(s, self.lin[t].rz),
                rtheta=RThetaLin(s, self.lin[t].r_theta),
                options=opts,
            )
            for t in range(H)
        ]
        self.ip = ip

        self.mode = mode
        self.iq2 = np.asarray(iq2)

        self.H = H

        self.d = np.asarray([ip[t].z[:nd] for t in range(H)])  # FIXME
        self.d = [ip[t].z[:nd] for t in range(H)]
        self.dq2 = [d[:nq] for d in self.d]

        if mode == "configurationforce":
            self.d_gamma1 = np.asarray([ip[t].z[nq : nq + nc] for t in range(H)])
            self.d_b1 = np.asarray([ip[t].z[nq + nc : nq + nc + nb] for t in range(H)])
        elif mode == "configuration":
            self.d_gamma1 = np.asarray([ip[t].z[0:0] for t in range(H)])
            self.d_b1 = np.asarray([ip[t].z[0:0] for t in range(H)])
        else:
            raise ValueError("invalid mode")

        off = 0
        self.delta_q0 = [ip[t].delta_z[0:nd, off : off + nq] for t in range(H)]
        off += nq

        self.delta_q1 = [ip[t].delta_z[0:nd, off : off + nq] for t in range(H)]
        off += nq

        self.delta_u1 = [ip[t].delta_z[0:nd, off : off + nu] for t in range(H)]
        off += nu


def update(im_traj, ref_traj, s, kappa, H):
    for t in range(H):
        im_traj.ip[t].kappa[0] = kappa

        if t > H - 2:
            continue

    return


def set_implicit_trajectory(im_traj, im_traj_cache):
    H = im_traj.H
    for t in range(H):
        im_traj.lin[t] = im_traj_cache.lin[t]
        im_traj.ip[t].r = im_traj_cache.ip[t].r
        im_traj.ip[t].rz = im_traj_cache.ip[t].rz
        im_traj.ip[t].rtheta = im_traj_cache.ip[t].rtheta
    return


def implicit_dynamics(im_traj, traj, threads=False, window=None):
    if window is None:
        window = list(range(traj.H + 2))

    for i, t in enumerate(window[:-2]):
        z_initialize_warmstart(im_traj.ip[t].z, im_traj.iq2, traj.q[i + 2])
        im_traj.ip[t].theta = traj.theta[i]

    if threads:
        # Implement threading if needed, here sequential
        for t in window[:-2]:
            status = interior_point_solve(im_traj.ip[t])
            if not status:
                print(f"implicit dynamics failure (t = {t})")
    else:
        for t in window[:-2]:
            status = interior_point_solve(im_traj.ip[t])
            if not status and im_traj.ip[t].options.warn:
                print(f"control implicit dynamics failure (t = {t})")

    for i, t in enumerate(window[:-2]):
        im_traj.dq2[t] -= traj.q[i + 2]
        if im_traj.mode == "configurationforce":
            im_traj.d_gamma1[t] -= traj.gamma[i]
            im_traj.d_b1[t] -= traj.b[i]
        elif im_traj.mode == "configuration":
            pass
    return
