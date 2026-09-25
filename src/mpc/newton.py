import copy
import numpy as np
from .objective import TrackingObjective
from src.helper import friction_dim
from .implicit_dynamics import implicit_dynamics
from .newton_jacobian import jacobian, initialize_jacobian, newton_jacobian
from .newton_residual import residual, update_traj, newton_residual
from .newton_indices import newton_indices
from src.simulator.trajectory import contact_trajectory, update_theta_step
from .newton_linear_solver import LUSolver, linear_solve_vector

# np.set_printoptions(precision=10, suppress=False)


class NewtonOptions:
    def __init__(
        self,
        r_tol=1.0e-5,
        max_iter=10,
        max_time=10000.0,
        beta_init=1.0e-5,
        threads=False,
        verbose=False,
        solver="lu_solver",
    ):
        self.r_tol = r_tol
        self.max_iter = max_iter
        self.max_time = max_time
        self.beta_init = beta_init
        self.threads = threads
        self.verbose = verbose
        self.solver = solver


class Newton_class:
    def __init__(
        self,
        jac,
        res,
        res_cand,
        Delta,
        n_u,
        nu_cand,
        traj,
        traj_cand,
        Delta_q,
        Delta_u,
        Delta_gamma,
        Delta_b,
        ind,
        obj,
        solver,
        beta,
        opts,
    ):
        self.jac = jac
        self.res = res
        self.res_cand = res_cand
        self.Delta = Delta
        self.n_u = n_u
        self.nu_cand = nu_cand
        self.traj = traj
        self.traj_cand = traj_cand
        self.Delta_q = Delta_q
        self.Delta_u = Delta_u
        self.Delta_gamma = Delta_gamma
        self.Delta_b = Delta_b
        self.ind = ind
        self.obj = obj
        self.solver = solver
        self.beta = beta
        self.opts = opts


def Newton(s, H, h, traj, im_traj, obj=None, opts=None, kappa=None):
    np.set_printoptions(suppress=True, precision=14)
    if obj is None:
        obj = TrackingObjective(s.model, s.env, H)
    if opts is None:
        opts = NewtonOptions()
    if kappa is None:
        kappa = im_traj.ip[0].kappa[0]

    model = s.model
    env = s.env

    mode = im_traj.mode

    ind = newton_indices(model, env, H, mode=mode)

    nq = model.nq
    nu = model.nu
    nc = model.nc
    nb = nc * friction_dim(env)
    nd = ind.nd

    jac = newton_jacobian(model, env, H, mode=mode)
    # precompute Jacobian for pre-factorization
    window = list(range(H + 2))
    # solve implict dynamic once for formulating jacobian
    implicit_dynamics(im_traj, traj, window=window, threads=opts.threads)

    jacobian(jac, im_traj, obj, H, opts.beta_init, window)

    res = newton_residual(model, env, H, mode=mode)
    res_cand = newton_residual(model, env, H, mode=mode)

    Delta = newton_residual(model, env, H, mode=mode)

    n_u = [np.zeros(nd) for _ in range(H)]
    n_u_cand = copy.deepcopy(n_u)

    traj_new = contact_trajectory(model, env, H, h, kappa=kappa)
    traj_cand = contact_trajectory(model, env, H, h, kappa=kappa)

    Delta_q = [np.zeros(nq) for _ in range(H)]

    Delta_u = [np.zeros(nu) for _ in range(H)]
    Delta_gamma = [np.zeros(nc) for _ in range(H)]
    Delta_b = [np.zeros(nb) for _ in range(H)]

    beta = opts.beta_init

    if opts.solver == "lu_solver":
        solver = LUSolver(jac.R)
    else:
        raise ValueError("Unsupported solver")

    return Newton_class(
        jac,
        res,
        res_cand,
        Delta,
        n_u,
        n_u_cand,
        traj_new,
        traj_cand,
        Delta_q,
        Delta_u,
        Delta_gamma,
        Delta_b,
        ind,
        obj,
        solver,
        beta,
        opts,
    )


def copy_traj(traj, traj_cand, H):
    H_t = traj.H
    H_s = traj_cand.H

    traj.kappa = 2.0e-4  # traj_cand.kappa.copy()

    for t in range(H + 2):
        traj.q[t] = traj_cand.q[t].copy()

    for t in range(H):
        traj.u[t] = traj_cand.u[t].copy()
        traj.w[t] = traj_cand.w[t].copy()
        traj.gamma[t] = traj_cand.gamma[t].copy()
        traj.b[t] = traj_cand.b[t].copy()
        traj.z[t] = traj_cand.z[t].copy()
        traj.theta[t] = traj_cand.theta[t].copy()

    return


def reset(core, ref_traj, q0, q1, window=None, warm_start=False):
    if window is None:
        window = list(range(core.traj.H + 2))

    core.beta = core.opts.beta_init

    if not warm_start:
        for t in range(core.traj.H):
            core.n_u[t].fill(0.0)
            core.nu_cand[t].fill(0.0)

        copy_traj(core.traj, ref_traj, core.traj.H)

    core.traj.q[0] = q0.copy()
    core.traj.q[1] = q1.copy()

    update_theta_step(core.traj, 0)
    update_theta_step(core.traj, 1)

    initialize_jacobian(core.jac, core.obj, core.traj.H)

    copy_traj(core.traj_cand, core.traj, core.traj.H)

    return


def newton_solve(core, s, q0, q1, window, im_traj, ref_traj, warm_start=False):
    elapsed_time = 0.0

    reset(core, ref_traj, q0, q1, window=window, warm_start=warm_start)
    implicit_dynamics(im_traj, core.traj, window=window, threads=core.opts.threads)

    residual(core.res, core, core.n_u, im_traj, core.traj, ref_traj, window)
    r_norm = np.linalg.norm(core.res.r, 1)

    for l in range(1, core.opts.max_iter + 1):
        if r_norm / len(core.res.r) < core.opts.r_tol:
            # print(f"break! {r_norm / len(core.res.r), core.opts.r_tol }")
            break
        jacobian(core.jac, im_traj, core.obj, core.traj.H, core.beta, window)
        linear_solve_vector(core.solver, core.Delta.r, core.jac.R, core.res.r)

        alpha = 1.0
        iter = 0
        update_traj(
            core.traj_cand, core.traj, core.nu_cand, core.n_u, core.Delta, alpha
        )

        implicit_dynamics(
            im_traj, core.traj_cand, window=window, threads=core.opts.threads
        )

        residual(
            core.res_cand, core, core.nu_cand, im_traj, core.traj_cand, ref_traj, window
        )
        r_cand_norm = np.linalg.norm(core.res_cand.r, 1)
        while r_cand_norm**2.0 >= (1.0 - 0.001 * alpha) * r_norm**2.0:
            alpha = 0.5 * alpha

            iter += 1
            if iter > 6:
                break

            update_traj(
                core.traj_cand, core.traj, core.nu_cand, core.n_u, core.Delta, alpha
            )
            implicit_dynamics(
                im_traj, core.traj_cand, window=window, threads=core.opts.threads
            )

            residual(
                core.res_cand,
                core,
                core.nu_cand,
                im_traj,
                core.traj_cand,
                ref_traj,
                window,
            )
            r_cand_norm = np.linalg.norm(core.res_cand.r, 1)

        update_traj(core.traj, core.traj, core.n_u, core.n_u, core.Delta, alpha)
        core.res.r[:] = core.res_cand.r
        r_norm = r_cand_norm

        if iter > 6:
            core.beta = min(core.beta * 1.3, 1.0e2)
        else:
            core.beta = max(1.0e1, core.beta / 1.3)

        if core.opts.verbose:
            print_status(core, im_traj, elapsed_time, alpha, l)


def print_status(core, im_traj, elapsed_time, alpha, l):
    def scn(value, digits=0):
        return f"{value:.{digits}e}"

    print(
        f"     t: {scn(elapsed_time, 0)}"
        f"     r̄: {scn(np.linalg.norm(core.res_cand.r, 1) / len(core.res_cand.r), 0)}"
        f"     r: {scn(np.linalg.norm(core.res.r, 1) / len(core.res.r), 0)}"
        f"     Δ: {scn(np.linalg.norm(core.Delta.r, 1) / len(core.Delta.r), 0)}"
        f"     α: {-int(round(np.log(alpha)))}"
        f"     l: {l}"
        f"     κ: {scn(im_traj.ip[0].kappa[0], 0)}"
    )
