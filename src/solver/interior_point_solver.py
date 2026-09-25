import os
import numpy as np
from src.helper import *
from src.solver.lu import LUSolver, linear_solve


def _mpc_linearized_solver(jacobian):
    """Load the optional MPC backend only for its structured Jacobian."""
    if not hasattr(jacobian, "M"):
        return None
    from src.mpc import linearized_solver

    return linearized_solver


def _residual_vector(r, idx):
    """
    Flatten residual object into a 1D vector matching idx layout.
    Falls back to returning the input if it's already array-like.
    """
    if hasattr(r, "rdyn") and hasattr(r, "rrst") and hasattr(r, "rbil"):
        # Linearized system uses only rdyn and rrst rows in the Jacobian.
        return np.concatenate([r.rdyn, r.rrst])
    return r


class InteriorPointOptions:

    def __init__(
        self,
        r_tol=1.0e-8,
        kappa_tol=1.0e-8,
        kappa_accept_ratio=1.0,
        ls_scale=0.5,
        max_iter=200,
        max_time=1e5,
        max_ls=3,
        diff_sol=False,
        reg=False,
        epsilon_min=0.05,  # ∈ [0.005, 0.25]
        kappa_reg=0.001,  # bilinear constraint violation trigger level
        gamma_reg=0.1,  # regularization scaling parameter
        solver_name="lu_solver",  # can be mapped to a function elsewhere
        undercut=2,  # anp.inf,  # target κ_vio = κ_tol / undercut
        verbose=False,
        warn=False,
        log_path=None,
        log_t=None,
        log_line_search=False,
        log_ls_path=None,
        log_cond=False,
    ):
        self.r_tol = r_tol
        self.kappa_tol = kappa_tol
        self.kappa_accept_ratio = kappa_accept_ratio
        self.ls_scale = ls_scale
        self.max_iter = max_iter
        self.max_ls = max_ls
        self.max_time = max_time
        self.diff_sol = diff_sol
        self.reg = reg
        self.epsilon_min = epsilon_min
        self.kappa_reg = kappa_reg
        self.gamma_reg = gamma_reg
        self.solver_name = solver_name
        self.undercut = undercut
        self.verbose = verbose
        self.warn = warn
        self.solver_name = solver_name
        self.log_path = log_path
        self.log_t = log_t


def _open_ipm_log(log_path):
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    log_f = open(log_path, "a", encoding="utf-8")
    if write_header:
        log_f.write(
            "t,iter,r_vol,k_vol,alpha_aff,alpha,alpha_ort,alpha_soc,ls_iters,"
            "reg_val,kappa,mu,sigma,alpha_bound,ort_limit_group,ort_limit_z,"
            "ort_limit_delta,ort_limit_z_val,ort_limit_dz_val,lin_res_aff,lin_res_cor\n"
        )
    return log_f


def _alpha_bound_label(alpha, alpha_ort, alpha_soc, tol=1.0e-12):
    is_ort = abs(alpha - alpha_ort) <= tol
    is_soc = abs(alpha - alpha_soc) <= tol
    if is_ort and is_soc:
        return "tie"
    if is_ort:
        return "ort"
    if is_soc:
        return "soc"
    return "other"


def _ort_step_limit_info(z, dz, ort_z, ort_dz, tau):
    alpha = 1.0
    limit = None
    for i in range(len(ort_z)):
        for j in range(len(ort_z[i])):
            k = ort_z[i][j]
            ks = ort_dz[i][j]
            if dz[ks] > 0:
                cand = tau * z[k] / dz[ks]
                if cand < alpha:
                    alpha = cand
                    limit = (i, k, ks, float(z[k]), float(dz[ks]))
    return alpha, limit


class InteriorPoint:
    def __init__(
        self,
        z,
        theta,
        space,
        idx,
        residual_func,
        jacobian_z_func,
        jacobian_theta_func,
        r,
        jacobian_z,
        jacobian_theta,
        options,
    ):

        self.options = options
        self.r_tol = options.r_tol
        self.kappa_tol = options.kappa_tol
        self.ls_scale = options.ls_scale
        self.max_iter = options.max_iter
        self.max_time = options.max_time
        self.max_ls = options.max_ls
        self.diff_sol = options.diff_sol
        self.epsilon_min = options.epsilon_min
        self.kappa_reg = options.kappa_reg
        self.gamma_reg = options.gamma_reg
        self.reg = options.reg

        self.space = space
        self.idx = idx
        self.residual_methods = {
            "r": residual_func,
            "rz": jacobian_z_func,
            "rtheta": jacobian_theta_func,
        }

        # 1D vectors: order is irrelevant
        self.z = z  # current point
        self.z_candidate = np.zeros_like(z)  # candidate point
        self.r = r
        self.delta = np.zeros(idx.n_delta)
        self.theta = theta

        # 2D matrices: support either numpy arrays or helper objects (e.g., RZLin).
        self.rz = jacobian_z  # Jacobian wrt z
        if hasattr(jacobian_theta, "shape"):
            self.rtheta = np.asfortranarray(jacobian_theta)
        else:
            self.rtheta = jacobian_theta  # object-style container
        rz_matrix = (
            jacobian_z.M if hasattr(jacobian_z, "M") else np.asfortranarray(jacobian_z)
        )

        # Sensitivity matrices: Fortran-order (to match Julia column-major)
        self.delta_z = np.zeros((idx.nz, len(theta)), order="F")  # ∂z/∂θ
        self.delta_z_space = np.zeros(
            (idx.n_delta, len(theta)), order="F"
        )  # mapped gradients

        # Solver should see a Fortran-contiguous matrix
        self.solver = LUSolver(rz_matrix)

        self.z_regularized = np.zeros_like(z)
        self.regularization_value = np.zeros(1)
        self.iterations = 0
        self.reg_val = [0.0]
        self.kappa = np.zeros(1)

        # Positive orthant cone variables (1D -> layout irrelevant)
        self.z_orthant = [
            np.zeros(len(idx.ortz[0])),
            np.zeros(len(idx.ortz[1])),
        ]
        self.delta_orthant = [
            np.zeros(len(idx.ort_delta[0])),
            np.zeros(len(idx.ort_delta[1])),
        ]

        # Second-order cone variables (also 1D blocks)
        self.z_soc = [
            [np.zeros(len(group)), np.zeros(len(group))] for group in idx.socri
        ]
        self.delta_soc = [
            [np.zeros(len(group)), np.zeros(len(group))] for group in idx.socri
        ]
        self.rho_vector = [
            [np.zeros(max(0, len(group) - 1)), np.zeros(max(0, len(group) - 1))]
            for group in idx.socri
        ]
        self.soc_element_indices = [
            [list(range(1, len(group))), list(range(1, len(group)))]
            for group in idx.socri
        ]


def initialize_interior_point(
    z, theta, space, idx, r_func, rz_func, rtheta_func, options
):
    r = np.zeros(idx.n_delta)

    # FIXME : Fortran might not necessary
    jacobian_z = np.zeros((idx.n_delta, idx.n_delta), order="F")
    jacobian_theta = np.zeros((idx.n_delta, len(theta)), order="F")
    jac = rz_func(z, theta)
    np.copyto(jacobian_z, np.asarray(jac))

    return InteriorPoint(
        z=z,
        theta=theta,
        space=space,
        idx=idx,
        residual_func=r_func,
        jacobian_z_func=rz_func,
        jacobian_theta_func=rtheta_func,
        r=r,
        jacobian_z=jacobian_z,
        jacobian_theta=jacobian_theta,
        options=options,
    )


def initialize_interior_point_mpc(
    z, theta, space, idx, r_func, rz_func, rtheta_func, r, rz, rtheta, options
):
    rz_func(rz, z, theta)  # compute jacobian for pre-factorization
    return InteriorPoint(
        z=z,
        theta=theta,
        space=space,
        idx=idx,
        residual_func=r_func,
        jacobian_z_func=rz_func,
        jacobian_theta_func=rtheta_func,
        r=r,
        jacobian_z=rz,
        jacobian_theta=rtheta,
        options=options,
    )


def interior_point_solve(ip):
    np.set_printoptions(precision=18, suppress=False)

    # unpack
    space = ip.space
    opts = ip.options
    z = ip.z  # Optimization variables
    r = ip.r
    jacobian = ip.rz  # jacobian of r w.r.t z
    mpc_solver = _mpc_linearized_solver(jacobian)
    delta = ip.delta  # newton step direction
    theta = ip.theta  # data , parameter vector
    solver = ip.solver  #

    # tolerances and parameters
    r_tol = opts.r_tol
    kappa_tol = opts.kappa_tol
    kappa_accept_ratio = max(1.0, float(getattr(opts, "kappa_accept_ratio", 1.0)))
    kappa_accept_tol = float(kappa_tol) * kappa_accept_ratio
    ls_scale = opts.ls_scale  # beta
    max_iter = opts.max_iter
    max_ls = opts.max_ls
    diff_sol = opts.diff_sol
    kappa_reg = opts.kappa_reg
    gamma_reg = opts.gamma_reg
    undercut = opts.undercut
    verbose = opts.verbose

    # reset iteration count and regularization
    ip.iterations = 0
    ip.reg_val[0] = 0.0
    ip.last_alpha_aff = None
    ip.last_alpha = None
    ip.last_alpha_ort = None
    ip.last_alpha_soc = None
    ip.last_ls_iters = None

    zort = ip.z_orthant
    delta_ort = ip.delta_orthant
    zsoc = ip.z_soc
    delta_soc = ip.delta_soc
    rho_vec = ip.rho_vector
    # index sets
    idx = ip.idx
    ortz = idx.ortz  # orthant related vars
    ort_delta = idx.ort_delta
    socz = idx.socz  # SOC related vars
    soc_delta = idx.soc_delta

    ortr = idx.ortr
    socr = idx.socr
    socri = idx.socri
    soc_element_indices = ip.soc_element_indices

    try:
        r = ip.residual_methods["r"](z, theta, 0.0)
    except TypeError:
        ip.residual_methods["r"](ip.r, z, theta, 0.0)
        r = ip.r
    if r is None:
        r = ip.r

    k_vol = bilinear_violation(ip, r)  # mostly contact/friction complmentarity
    r_vol = residual_violation(ip, r)
    ip.last_k_vol = k_vol
    ip.last_r_vol = r_vol

    log_f = None
    log_active = opts.log_path is not None
    current_t = getattr(ip, "current_t", None)
    if log_active and (opts.log_t is None or current_t == opts.log_t):
        log_f = _open_ipm_log(opts.log_path)
        log_f.write(f"# ipm t={current_t} max_iter={int(max_iter)}\n")

    prev_improved = True
    for iteration in range(0, int(max_iter)):
        if r_vol < r_tol and k_vol < kappa_accept_tol:
            break

        ort_limit_info = None
        alpha_bound = "unknown"

        ip.iterations += 1
        ip.reg_val[0] = regularization(ip, k_vol, kappa_reg, gamma_reg)
        if ip.reg_val[0] is not None:
            ip.last_reg_val = float(ip.reg_val[0])

        jacobian = rz_run(ip, jacobian, z, theta, reg=ip.reg_val[0])
        if (
            mpc_solver is not None
            and isinstance(jacobian, mpc_solver.RZLin)
            and isinstance(r, mpc_solver.RLin)
        ):
            mpc_solver.linear_solve(delta, jacobian, r, reg=ip.reg_val[0])
        else:
            b_vec = _residual_vector(r, idx)
            delta = linear_solve(
                solver, x=delta, A=jacobian, b=b_vec, reg=ip.reg_val[0]
            )
        ip.last_delta_aff = delta.copy()
        if log_f is not None:
            # Log linear solve residual for the affine step.
            A_mat = jacobian.M if hasattr(jacobian, "M") else jacobian
            try:
                res_aff = A_mat @ delta - b_vec
                ip.last_lin_res_aff = float(np.linalg.norm(res_aff))
            except Exception:
                ip.last_lin_res_aff = None

        alpha_ort, _ = _ort_step_limit_info(z, delta, ortz, ort_delta, tau=1.0)
        alpha_soc = soc_step_length(
            z,
            delta,
            socz,
            soc_delta,
            zsoc,
            delta_soc,
            rho_vec,
            soc_element_indices,
            tau=1.0,
        )
        alpha = min(alpha_ort, alpha_soc)  # alpha keep getting smaller
        ip.last_alpha_aff = alpha
        # sigma is a parameter that decide whether current step should to near boundary or stay in central
        mu, sigma = centering(
            z,
            delta,
            ortz,
            ort_delta,
            socz,
            soc_delta,
            zort,
            delta_ort,
            zsoc,
            delta_soc,
            alpha,
        )
        ip.last_mu = float(mu)
        ip.last_sigma = float(sigma)
        # If residuals were not improving in the previous iteration, cap sigma to avoid reinforcing bad centering.
        # if not prev_improved:
        #     sigma = min(sigma, 0.3)
        # If SOC is the active step limiter, enforce a minimum centering strength.
        # if alpha_soc < alpha_ort:
        #     sigma = max(sigma, 0.1)
        ip.last_sigma_eff = float(sigma)
        ip.last_prev_improved = bool(prev_improved)
        # corrector residual
        ip.kappa[0] = max(sigma * mu, kappa_tol / undercut)
        ip.last_kappa = float(ip.kappa[0])

        try:
            r = ip.residual_methods["r"](z, theta, ip.kappa[0])
        except TypeError:
            ip.residual_methods["r"](ip.r, z, theta, ip.kappa[0])
            r = ip.r

        if (
            mpc_solver is not None
            and isinstance(jacobian, mpc_solver.RZLin)
            and isinstance(r, mpc_solver.RLin)
        ):
            mpc_solver.general_correction_term_lin(
                r, delta, ortr, socr, socri, ort_delta, soc_delta
            )
        else:
            r = general_correction_term(
                r, delta, ortr, socr, socri, ort_delta, soc_delta
            )

        if (
            mpc_solver is not None
            and isinstance(jacobian, mpc_solver.RZLin)
            and isinstance(r, mpc_solver.RLin)
        ):
            mpc_solver.linear_solve(delta, jacobian, r, reg=ip.reg_val[0])
        else:
            b_vec = _residual_vector(r, idx)
            delta = linear_solve(
                solver, x=delta, A=jacobian, b=b_vec, reg=ip.reg_val[0]
            )
        ip.last_delta_cor = delta.copy()
        if log_f is not None:
            # Log linear solve residual for the corrector step.
            A_mat = jacobian.M if hasattr(jacobian, "M") else jacobian
            try:
                res_cor = A_mat @ delta - b_vec
                ip.last_lin_res_cor = float(np.linalg.norm(res_cor))
            except Exception:
                ip.last_lin_res_cor = None

        tau = max(
            0.95, 1 - max(r_vol, k_vol) ** 2
        )  # fraction to the boundary safety factor

        alpha_ort, ort_limit_info = _ort_step_limit_info(
            z, delta, ortz, ort_delta, tau=tau
        )
        alpha_soc, soc_limit_info = soc_step_length(
            z,
            delta,
            socz,
            soc_delta,
            zsoc,
            delta_soc,
            rho_vec,
            soc_element_indices,
            tau=min(tau, 0.99),
            return_limit=True,
        )
        alpha = min(alpha_ort, alpha_soc)
        ip.last_alpha_ort = alpha_ort
        ip.last_alpha_soc = alpha_soc
        ip.last_alpha = alpha
        ip.last_soc_limit = soc_limit_info
        alpha_bound = _alpha_bound_label(alpha, alpha_ort, alpha_soc)
        # Heuristic safeguard: if we're near the SOC boundary and the direction is
        # strongly outward, avoid collapsing alpha_soc to ~0.
        # if soc_limit_info is not None:
        #     z_vec = np.array(soc_limit_info.get("z", []), dtype=float)
        #     dz0 = soc_limit_info.get("dz0")
        #     dz_norm = soc_limit_info.get("dz_norm")
        #     if z_vec.size > 1 and dz0 is not None and dz_norm is not None:
        #         slack = float(z_vec[0] - np.linalg.norm(z_vec[1:]))
        #         dir_margin = float(dz0 - dz_norm)
        #         if slack < 1.0e-8 and dir_margin < -1.0e-3:
        #             alpha_soc = max(alpha_soc, 1.0e-1)
        #             alpha = min(alpha_ort, alpha_soc)
        #             ip.last_alpha_soc = alpha_soc
        #             ip.last_alpha = alpha
        #             alpha_bound = _alpha_bound_label(alpha, alpha_ort, alpha_soc)

        z_prev = None
        z_pre_step = None
        if hasattr(ip, "iter_callback") and callable(ip.iter_callback):
            z_prev = z.copy()
            z_pre_step = z.copy()
        candidate_point(z, space, z, delta, alpha)

        new_k_vol = 0.0
        new_r_vol = 0.0

        for i in range(max_ls):
            try:
                r = ip.residual_methods["r"](z, theta, 0)
            except TypeError:
                ip.residual_methods["r"](ip.r, z, theta, 0)
                r = ip.r

            new_k_vol = bilinear_violation(ip, r)
            new_r_vol = residual_violation(ip, r)
            if new_r_vol <= r_vol or new_k_vol <= k_vol:
                break
            candidate_point(z, space, z, delta, -alpha * ls_scale ** (i + 1))
        ip.last_ls_iters = i

        prev_improved = (new_r_vol <= r_vol) and (new_k_vol <= k_vol)
        k_vol = new_k_vol
        r_vol = new_r_vol
        ip.last_k_vol = k_vol
        ip.last_r_vol = r_vol
        if z_prev is not None:
            ip.last_step = z - z_prev
            ip.last_z_prev = z_prev
            ip.last_z_pre_step = z_pre_step
        if log_f is not None:
            reg_val = getattr(ip, "last_reg_val", None)
            kappa = getattr(ip, "last_kappa", None)
            mu = getattr(ip, "last_mu", None)
            sigma = getattr(ip, "last_sigma", None)
            lin_res_aff = getattr(ip, "last_lin_res_aff", None)
            lin_res_cor = getattr(ip, "last_lin_res_cor", None)
            if ort_limit_info is None:
                ort_group = -1
                ort_z_idx = -1
                ort_delta_idx = -1
                ort_z_val = float("nan")
                ort_dz_val = float("nan")
            else:
                ort_group, ort_z_idx, ort_delta_idx, ort_z_val, ort_dz_val = (
                    ort_limit_info
                )
            log_f.write(
                f"{current_t},{iteration},{r_vol:.6e},{k_vol:.6e},"
                f"{ip.last_alpha_aff:.6e},{ip.last_alpha:.6e},"
                f"{ip.last_alpha_ort:.6e},{ip.last_alpha_soc:.6e},"
                f"{ip.last_ls_iters},{reg_val},{kappa},{mu},{sigma},"
                f"{alpha_bound},{ort_group},{ort_z_idx},{ort_delta_idx},"
                f"{ort_z_val},{ort_dz_val},{lin_res_aff},{lin_res_cor}\n"
            )
            log_f.flush()
        if hasattr(ip, "iter_callback") and callable(ip.iter_callback):
            ip.iter_callback(
                ip,
                {
                    "iter": iteration,
                    "r_vol": r_vol,
                    "k_vol": k_vol,
                    "alpha_aff": ip.last_alpha_aff,
                    "alpha": ip.last_alpha,
                    "alpha_ort": ip.last_alpha_ort,
                    "alpha_soc": ip.last_alpha_soc,
                    "ls_iters": ip.last_ls_iters,
                    "reg_val": getattr(ip, "last_reg_val", None),
                    "kappa": getattr(ip, "last_kappa", None),
                    "mu": getattr(ip, "last_mu", None),
                    "sigma": getattr(ip, "last_sigma", None),
                    "soc_limit": getattr(ip, "last_soc_limit", None),
                },
            )

    if log_f is not None:
        log_f.close()

    if r_vol < r_tol and k_vol < kappa_accept_tol:
        regularization_max(ip, k_vol, opts.gamma_reg)
        if diff_sol:
            differentiate_solution(ip, reg=ip.reg_val[0])
        ip.last_converged_near = bool(k_vol >= kappa_tol)
        return True
    else:
        ip.last_k_vol = k_vol
        ip.last_r_vol = r_vol
        return False


def regularization(ip, kappa_violation, kappa_reg, gamma_reg):
    ip.reg_val[0] = kappa_violation * gamma_reg if kappa_violation < kappa_reg else 0.0
    return ip.reg_val[0]


def regularization_max(ip, kappa_violation, gamma_reg):
    reg = kappa_violation * gamma_reg
    if reg > ip.reg_val[0]:
        ip.reg_val[0] = reg


def candidate_point(z_bar, space, z, delta, alpha):
    np.copyto(z_bar, z - alpha * delta)


def differentiate_solution(ip, reg=0.0):
    z = ip.z
    theta = ip.theta
    rz = ip.rz  # Fortran-order
    rtheta = ip.rtheta  # Fortran-order
    delta_z = ip.delta_z  # Fortran-order
    delta_z_space = ip.delta_z_space  # Fortran-order
    space = ip.space

    # update rz and rθ
    rz_run(ip, rz, z, theta, reg=reg)
    rtheta_run(ip, rtheta, z, theta)

    mpc_solver = _mpc_linearized_solver(rz)
    if (
        mpc_solver is not None
        and isinstance(rz, mpc_solver.RZLin)
        and hasattr(rtheta, "r_theta_dyn0")
    ):
        delta_z_space.fill(0.0)
        mpc_solver.linear_solve_theta(delta_z_space, rz, rtheta, reg=reg)
        delta_z_space *= -1.0
    else:
        linear_solve(ip.solver, x=delta_z_space, A=rz, b=rtheta, reg=reg)
        delta_z_space *= -1.0

    # map from δzs → δz
    mapping(delta_z, space, delta_z_space, z)


def mapping(delta_z, space, delta_z_space, z):
    np.copyto(delta_z, delta_z_space)


def residual_violation(ip, r, nquat: int = 0):
    if hasattr(r, "rdyn") and hasattr(r, "rrst"):
        rdyn = np.asarray(r.rdyn)
        rrst = np.asarray(r.rrst)
        e1 = np.linalg.norm(rdyn, ord=np.inf)
        e2 = np.linalg.norm(rrst, ord=np.inf)
        if getattr(ip.options, "verbose", False):
            if e1 >= e2:
                idx = int(np.argmax(np.abs(rdyn)))
                val = float(rdyn[idx])
                print(f"[residual_violation] block=rdyn idx={idx} val={val}")
            else:
                idx = int(np.argmax(np.abs(rrst)))
                val = float(rrst[idx])
                print(f"[residual_violation] block=rrst idx={idx} val={val}")
        return max(e1, e2)
    r_arr = np.asarray(r)
    r_eq = r_arr[ip.idx.equr]
    if getattr(ip.options, "verbose", False) and not getattr(
        ip, "suppress_violation_logs", False
    ):
        local_idx = int(np.argmax(np.abs(r_eq)))
        global_idx = int(ip.idx.equr[local_idx])
        val = float(r_eq[local_idx])
        detail = "unknown"
        if hasattr(ip, "model") and hasattr(ip.model, "nq") and hasattr(ip.model, "nc"):
            nq = int(ip.model.nq)
            nc = int(ip.model.nc)
            nb = int(getattr(ip.model, "nb", 0))
            if global_idx < nq:
                detail = f"dyn[{global_idx}]"
            elif global_idx < nq + nc:
                detail = f"res_sd[{global_idx - nq}]"
            elif global_idx < nq + nc + nb:
                detail = f"res_vT[{global_idx - nq - nc}]"
            elif global_idx < nq + nc + nb + nc:
                detail = f"res_fric_ineq[{global_idx - nq - nc - nb}]"
        print(
            f"[residual_violation] block=eqr idx={global_idx} val={val} detail={detail}"
        )
    return np.linalg.norm(r_eq, ord=np.inf)


def bilinear_violation(ip, r):
    if hasattr(r, "rbil"):
        r_bil = r.rbil
    else:
        r_bil = r[ip.idx.bil]
    if getattr(ip.options, "verbose", False) and not getattr(
        ip, "suppress_violation_logs", False
    ):
        r_bil_arr = np.asarray(r_bil)
        local_idx = int(np.argmax(np.abs(r_bil_arr)))
        if hasattr(ip.idx, "bil"):
            global_idx = int(ip.idx.bil[local_idx])
            idx_msg = f"{global_idx}"
        else:
            idx_msg = f"{local_idx}"
        val = float(r_bil_arr[local_idx])
        print(f"[bilinear_violation] idx={idx_msg} val={val}")
    return np.linalg.norm(r_bil, ord=np.inf)


def rz_run(ip, rz, z, theta, reg=0.0):
    z_reg = ip.z_regularized
    np.copyto(z_reg, z)

    # positive orthant regularization
    for group in ip.idx.ortz:
        if len(group) == 0:
            continue
        # clamp z_reg[group] >= reg
        z_reg[group] = np.maximum(z_reg[group], reg)

    # call underlying jacobian wrt z
    try:
        # out-of-place version: returns a new array
        try:
            jac = ip.residual_methods["rz"](z_reg, theta, reg=reg)
        except TypeError:
            jac = ip.residual_methods["rz"](z_reg, theta)
        if jac is not None:
            # ensure we only copy values, rz remains Fortran-order
            np.copyto(rz, np.asarray(jac))
    except TypeError:
        # in-place version: fills rz directly
        try:
            ip.residual_methods["rz"](rz, z_reg, theta, reg=reg)
        except TypeError:
            ip.residual_methods["rz"](rz, z_reg, theta)

    return rz


def rtheta_run(ip, rtheta, z, theta):
    try:
        jac_theta = ip.residual_methods["rtheta"](z, theta)
        if jac_theta is not None:
            np.copyto(rtheta, np.asarray(jac_theta))
    except TypeError:
        ip.residual_methods["rtheta"](rtheta, z, theta)

    return rtheta
