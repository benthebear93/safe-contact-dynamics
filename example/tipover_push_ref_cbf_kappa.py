import copy
import json
import time
from pathlib import Path

import numpy as np
from example.cbf_qp import minimize_qp

from src.residual_models.tipover_push_residual import make_tipover_push_residual
from src.robots.tipover.model_linear import TipOverPusher
from src.simulator.simulator import Simulator


def load_ref_traj(path: Path) -> dict:
    payload = json.loads(path.read_text())
    for key in ("q", "u", "h"):
        if key not in payload:
            raise KeyError(f"Missing key in ref trajectory: {key}")
    return payload


def build_tipover_model(q0: np.ndarray, ref: dict) -> TipOverPusher:
    return TipOverPusher(
        nq=int(ref.get("nq", 7)),
        nu=int(ref.get("nu", 2)),
        nw=int(ref.get("nw", 0)),
        nc=int(ref.get("nc", 5)),
        box_half_width=0.09,
        box_half_height=0.11,
        pusher_radius=0.02,
        mb=1.2,
        mp=10.0,
        I=None,
        mu_floor=0.6,
        mu_pusher=0.5,
        gravity=9.81,
        pusher_gap=0.004,
        pusher_height=float(q0[6]),
    )


def build_sim(model, residuals, dt, num_steps, kappa_tol, diff_sol=False):
    r, rz, rtheta = residuals
    sim = Simulator(
        model,
        num_steps,
        h=dt,
        diff_sol=diff_sol,
        residual=r,
        jacobian_z=rz,
        jacobian_theta=rtheta,
        temp_cmd=np.zeros(model.nu),
        kappa_tol=kappa_tol,
    )
    sim.ip.options.verbose = False
    return sim


def step_with_u(sim, t, u_cmd, diff_sol=False):
    sim.traj.u[t][:] = u_cmd
    sim.traj.w[t][:] = 0.0
    status = sim.step(t, diff_sol=diff_sol)
    return status, sim


def _solve_qp_casadi(H, g, A, b, lbx, ubx):
    # Solve with SciPy SLSQP only.
    try:
        H = np.asarray(H, dtype=float)
        g = np.asarray(g, dtype=float).reshape(-1)
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        lbx = np.asarray(lbx, dtype=float).reshape(-1)
        ubx = np.asarray(ubx, dtype=float).reshape(-1)
        n = g.size
        if H.shape == (n, n):
            try:
                x0 = np.linalg.solve(H + 1e-12 * np.eye(n), -g)
            except Exception:
                x0 = np.clip(-g, lbx, ubx)
        else:
            x0 = np.clip(-g, lbx, ubx)
        x0 = np.clip(x0, lbx, ubx)

        res = minimize_qp(H, g, A, b, lbx, ubx, x0, maxiter=200)
        x = np.asarray(res.x, dtype=float).reshape(-1)
        feas = bool(np.all(A @ x - b <= 1e-6))
        if bool(res.success) and feas:
            info = {
                "solver": "scipy_slsqp",
                "success": True,
                "return_status": str(res.message),
            }
            return x, True, info
    except Exception as e:
        return (
            np.zeros(np.asarray(g, dtype=float).reshape(-1).size),
            False,
            {"solver": "scipy_slsqp", "success": False, "error": str(e)},
        )
    return (
        np.zeros(np.asarray(g, dtype=float).reshape(-1).size),
        False,
        {"solver": "scipy_slsqp", "success": False, "return_status": "infeasible_or_failed"},
    )


def _floor_corner_gamma(gamma, n_floor_contacts=4):
    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    n = min(n_floor_contacts, gamma.size)
    if n <= 0:
        return np.zeros(0, dtype=float)
    return gamma[:n].copy()


def _predict_terms(sim, t, u_nom, n_floor_contacts=4, theta_idx=3):
    """Predict contact-force and progress linearization around u_nom.

    Returns:
      gamma_nom (n_floor,), dgamma_du (n_floor, nu),
      theta_next_nom (scalar), dtheta_du (nu,)
    """
    sim_nom = getattr(sim, "_diff_sim", None)
    if sim_nom is None:
        sim_nom = copy.deepcopy(sim)
        sim_nom.ip.options.diff_sol = True
    else:
        sim_nom.ip.options.diff_sol = True
        sim_nom.prev_z = sim.prev_z
        sim_nom.traj.q[t][:] = sim.traj.q[t]
        sim_nom.traj.q[t + 1][:] = sim.traj.q[t + 1]
        sim_nom.traj.v[t][:] = sim.traj.v[t]

    status, sim_nom = step_with_u(sim_nom, t, u_nom, diff_sol=True)
    if not status:
        return (
            False,
            sim_nom,
            np.zeros(0, dtype=float),
            np.zeros((0, u_nom.size), dtype=float),
            0.0,
            np.zeros(u_nom.size, dtype=float),
        )

    gamma_nom = _floor_corner_gamma(
        sim_nom.traj.gamma[t], n_floor_contacts=n_floor_contacts
    )
    dgamma_du_full = np.asarray(sim_nom.grad.dgamma1_du1[t], dtype=float)
    n = gamma_nom.shape[0]
    dgamma_du = (
        np.zeros((0, u_nom.size), dtype=float)
        if n <= 0
        else dgamma_du_full[:n, :].copy()
    )

    theta_next_nom = float(sim_nom.traj.q[t + 2][theta_idx])
    try:
        dtheta_du = np.asarray(sim_nom.grad.dq3_du1[t][theta_idx, :], dtype=float)
    except Exception:
        dtheta_du = np.zeros(u_nom.size, dtype=float)

    return True, sim_nom, gamma_nom, dgamma_du, theta_next_nom, dtheta_du


def _build_qp_objective(u_nom, Q):
    """Build baseline QP objective: ||u - u_nom||_Q^2."""
    H = Q.copy()
    g = -Q @ u_nom
    return H, g


def _robust_cbf_delta(params, h_curr, h_nom, dgamma_row):
    robust_delta = 0.0
    robust_delta_base = 0.0
    robust_delta_ema = 0.0
    robust_delta_hk = 0.0
    robust_delta_hnom = 0.0
    robust_delta_jac = 0.0
    robust_delta_under_gate = 0.0
    robust_delta_under_gate_nom = 0.0

    if not params.get("use_robust_cbf", False):
        return {
            "robust_delta": robust_delta,
            "robust_delta_base": robust_delta_base,
            "robust_delta_ema": robust_delta_ema,
            "robust_delta_hk": robust_delta_hk,
            "robust_delta_hnom": robust_delta_hnom,
            "robust_delta_jac": robust_delta_jac,
            "robust_delta_under_gate": robust_delta_under_gate,
            "robust_delta_under_gate_nom": robust_delta_under_gate_nom,
        }

    robust_margin = float(params.get("eps_robust", 0.0))
    robust_delta_base = float(params.get("robust_base", 0.0))
    robust_mode = str(params.get("robust_mode", "ema_max"))
    prev_err = float(params.get("_robust_err_ema", 0.0))
    prev_under = float(params.get("_robust_under_ema", prev_err))
    robust_delta_ema = max(robust_margin, prev_err)

    if robust_mode == "state_dependent":
        h_scale = max(float(params.get("robust_h_scale", 1e-3)), 1e-12)
        near_gate = 1.0 / (1.0 + abs(float(h_curr)) / h_scale)
        robust_delta_hk = float(params.get("robust_hk_gain", 0.0)) * near_gate
        robust_delta_hnom = float(params.get("robust_hnom_gain", 0.0)) * max(
            0.0, -float(h_nom)
        )
        robust_delta_jac = float(params.get("robust_jac_gain", 0.0)) * float(
            np.linalg.norm(np.asarray(dgamma_row, dtype=float).reshape(-1), ord=2)
        )
        robust_delta = (
            robust_delta_base
            + robust_delta_ema
            + robust_delta_hk
            + robust_delta_hnom
            + robust_delta_jac
        )
    elif robust_mode == "under_gate":
        h_scale = max(float(params.get("robust_h_scale", 1e-3)), 1e-12)
        near_gate = 1.0 / (1.0 + abs(float(h_curr)) / h_scale)
        robust_delta_under_gate = near_gate * prev_under
        robust_delta = robust_margin + robust_delta_under_gate
    elif robust_mode == "under_gate_plus":
        h_scale = max(float(params.get("robust_h_scale", 1e-3)), 1e-12)
        near_gate = 1.0 / (1.0 + abs(float(h_curr)) / h_scale)
        h_nom_deficit = max(0.0, -float(h_nom))
        robust_delta_under_gate = near_gate * prev_under
        robust_delta_under_gate_nom = (
            near_gate * float(params.get("robust_hnom_gain", 0.02)) * h_nom_deficit
        )
        robust_delta = (
            robust_margin + robust_delta_under_gate + robust_delta_under_gate_nom
        )
    else:
        robust_delta = robust_delta_ema

    return {
        "robust_delta": float(max(0.0, robust_delta)),
        "robust_delta_base": float(robust_delta_base),
        "robust_delta_ema": float(robust_delta_ema),
        "robust_delta_hk": float(robust_delta_hk),
        "robust_delta_hnom": float(robust_delta_hnom),
        "robust_delta_jac": float(robust_delta_jac),
        "robust_delta_under_gate": float(robust_delta_under_gate),
        "robust_delta_under_gate_nom": float(robust_delta_under_gate_nom),
    }


def safety_filter(sim, t, u_nom, params):
    u_nom = np.array(u_nom, dtype=float).copy()
    u_min = np.array(params["u_min"], dtype=float)
    u_max = np.array(params["u_max"], dtype=float)
    u_nom = np.clip(u_nom, u_min, u_max)

    ok_pred, sim_nom, gamma_nom, dgamma_du, theta_next_nom, dtheta_du = _predict_terms(
        sim,
        t,
        u_nom,
        n_floor_contacts=params["n_floor_contacts"],
        theta_idx=params["theta_idx"],
    )
    if not ok_pred:
        return False, sim_nom, u_nom, None

    n_floor = gamma_nom.shape[0]
    corner_idx = int(params.get("constrained_corner_idx", 0))
    if corner_idx < 0 or corner_idx >= n_floor:
        corner_idx = 0

    h_nom = float(params["F_max"] - gamma_nom[corner_idx])

    gamma_prev = (
        _floor_corner_gamma(
            sim.traj.gamma[t - 1], n_floor_contacts=params["n_floor_contacts"]
        )
        if t > 0
        else np.zeros(n_floor, dtype=float)
    )
    h_curr = float(params["F_max"] - gamma_prev[corner_idx])

    # Keep CBF inequality form unchanged.
    a_cbf = -dgamma_du[corner_idx, :].reshape(1, -1)
    rhs = float((1.0 - params["cbf_alpha"]) * h_curr)
    robust_terms = _robust_cbf_delta(
        params=params,
        h_curr=h_curr,
        h_nom=h_nom,
        dgamma_row=dgamma_du[corner_idx, :],
    )
    rhs += float(robust_terms["robust_delta"])
    if params.get("use_rhs_smoothing", False):
        rhs_prev = float(params.get("_rhs_cbf_prev", rhs))
        smooth_alpha = float(params.get("rhs_smooth_alpha", 0.2))
        rhs = smooth_alpha * rhs + (1.0 - smooth_alpha) * rhs_prev
        params["_rhs_cbf_prev"] = rhs
    rhs_aff = rhs - h_nom + float(a_cbf.reshape(-1) @ u_nom)
    A = -a_cbf
    b = np.array([-rhs_aff], dtype=float)

    Q = np.array(params["qp_Q"], dtype=float)
    if Q.ndim == 1:
        Q = np.diag(Q)

    theta_ref_arr = params["theta_ref"]
    theta_ref_next = float(theta_ref_arr[min(t + 1, len(theta_ref_arr) - 1)])

    H, g = _build_qp_objective(u_nom=u_nom, Q=Q)

    t_qp0 = time.perf_counter()
    u_safe, ok_qp, qp_info = _solve_qp_casadi(H, g, A, b, u_min, u_max)
    t_qp1 = time.perf_counter()
    qp_time_s = float(t_qp1 - t_qp0)

    if not ok_qp:
        # Infeasibility diagnostics as requested.
        res_nom = (A @ u_nom - b).reshape(-1)
        print(
            f"[QP infeasible] t={t}, mode={params.get('controller_name','unknown')}, "
            f"max_res_nom={float(np.max(res_nom)):.3e}, min_slack_nom={float(np.min(b - A @ u_nom)):.3e}, "
            f"active_like={int(np.sum(np.abs(b - A @ u_nom) < 1e-8))}, info={qp_info}"
        )
        u_safe = u_nom

    cbf_residual = float((A @ u_safe - b).reshape(-1)[0])
    if cbf_residual > 1e-6:
        print(
            f"[Constraint residual] t={t}, mode={params.get('controller_name','unknown')}, "
            f"res={cbf_residual:.3e}, A={A.reshape(-1)}, b={b.reshape(-1)}"
        )

    status, sim = step_with_u(sim, t, u_safe, diff_sol=False)
    if not status:
        return False, sim, u_safe, None

    gamma_safe = _floor_corner_gamma(
        sim.traj.gamma[t], n_floor_contacts=params["n_floor_contacts"]
    )
    gamma_safe_corner = float(gamma_safe[corner_idx]) if gamma_safe.size else 0.0
    h_actual = float(params["F_max"] - gamma_safe_corner)
    h_pred = float(h_nom + float(a_cbf.reshape(-1) @ (u_safe - u_nom)))
    if params.get("use_robust_cbf", False):
        pred_err = h_actual - h_pred
        err_under = max(0.0, -pred_err)
        err_over = max(0.0, pred_err)
        beta = float(params.get("robust_beta", 0.2))
        params["_robust_err_ema"] = (1.0 - beta) * float(
            params.get("_robust_err_ema", 0.0)
        ) + beta * abs(pred_err)
        params["_robust_under_ema"] = (1.0 - beta) * float(
            params.get("_robust_under_ema", 0.0)
        ) + beta * err_under
        params["_robust_over_ema"] = (1.0 - beta) * float(
            params.get("_robust_over_ema", 0.0)
        ) + beta * err_over

    return (
        True,
        sim,
        u_safe,
        {
            "gamma_nom": gamma_nom.copy(),
            "gamma_safe": gamma_safe.copy(),
            "h_nom": h_nom,
            "h_pred": h_pred,
            "h_actual": h_actual,
            "u_delta_norm": float(np.linalg.norm(u_safe - u_nom)),
            "qp_solved": True,
            "qp_time_s": qp_time_s,
            "cbf_residual": cbf_residual,
            "theta_next_nom": float(theta_next_nom),
            "theta_ref_next": float(theta_ref_next),
            "robust_delta": float(robust_terms["robust_delta"]),
        },
    )


def run_sim(params, mode):
    # mode in {"nominal", "cbf", "robust_cbf"}
    model = params["model"]
    residuals = params["residuals"]
    dt = params["dt"]
    num_steps = params["num_steps"]
    kappa_tol = params["kappa_tol"]
    u_ref = params["u_ref"]
    q0 = params["q0"]
    v0 = params["v0"]

    sim = build_sim(model, residuals, dt, num_steps, kappa_tol, diff_sol=False)
    sim.traj.reset()
    sim.grad.reset()
    sim.prev_z = None
    sim.set_state(q0, v0, 0)

    sim_diff = build_sim(model, residuals, dt, num_steps, kappa_tol, diff_sol=True)
    sim_diff.traj.reset()
    sim_diff.grad.reset()
    sim_diff.prev_z = None
    sim._diff_sim = sim_diff

    cidx = int(params.get("constrained_corner_idx", 0))

    t_log = []
    theta_log = []
    gamma_corner0_log = []
    h_pred_log = []
    h_actual_log = []
    u_delta_norm_log = []
    qp_time_log = []
    cbf_residual_log = []
    violation_log = []
    u_nom_log = []
    u_cmd_log = []
    q_log = []

    violation_count = 0
    status = True
    ctrl_params = params.copy()
    ctrl_params["controller_name"] = mode
    ctrl_params["use_robust_cbf"] = mode == "robust_cbf"
    ctrl_params["use_rhs_smoothing"] = bool(
        params.get("use_rhs_smoothing", False) and mode == "robust_cbf"
    )
    ctrl_params["rhs_smooth_alpha"] = float(params.get("rhs_smooth_alpha", 0.2))
    ctrl_params["_rhs_cbf_prev"] = float(params.get("_rhs_cbf_prev", 0.0))

    for t in range(num_steps):
        u_nom = np.array(u_ref[t], dtype=float)
        u_nom_log.append(u_nom.copy())

        if mode == "nominal":
            sim.traj.u[t][:] = u_nom
            sim.traj.w[t][:] = 0.0
            status = sim.step(t, diff_sol=False)
            u_cmd = u_nom.copy()
            qp_time = 0.0
            cbf_residual = np.nan
            h_pred_now = np.nan
            h_actual_now = np.nan
        else:
            status, sim_next, u_cmd, info = safety_filter(sim, t, u_nom, ctrl_params)
            sim = sim_next
            if info is None:
                info = {}
            qp_time = float(info.get("qp_time_s", 0.0))
            cbf_residual = float(info.get("cbf_residual", np.nan))
            h_pred_now = float(info.get("h_pred", np.nan))
            h_actual_now = float(info.get("h_actual", np.nan))

        if not status:
            break

        gamma_vec = _floor_corner_gamma(
            sim.traj.gamma[t], n_floor_contacts=params["n_floor_contacts"]
        )
        gamma0 = float(gamma_vec[cidx]) if gamma_vec.size else 0.0
        theta_now = float(sim.traj.q[t + 1][params["theta_idx"]])

        violation = int(gamma0 > float(params["F_max"]))
        violation_count += violation

        t_log.append(float(t * dt))
        theta_log.append(theta_now)
        gamma_corner0_log.append(gamma0)
        h_pred_log.append(h_pred_now)
        h_actual_log.append(
            h_actual_now if np.isfinite(h_actual_now) else float(params["F_max"] - gamma0)
        )
        u_delta_norm_log.append(float(np.linalg.norm(u_cmd - u_nom)))
        qp_time_log.append(qp_time)
        cbf_residual_log.append(cbf_residual)
        violation_log.append(violation)
        u_cmd_log.append(np.array(u_cmd, dtype=float))
        q_log.append(np.array(sim.traj.q[t + 1], dtype=float))

    t_arr = np.array(t_log, dtype=float)
    theta_arr = np.array(theta_log, dtype=float)
    gamma_arr = np.array(gamma_corner0_log, dtype=float)
    u_delta_arr = np.array(u_delta_norm_log, dtype=float)
    violation_arr = np.array(violation_log, dtype=int)

    theta_target = float(params["theta_target"])
    success_tipover = bool(
        theta_arr.size > 0 and float(np.max(theta_arr)) >= theta_target
    )
    success_safe = bool(np.sum(violation_arr) == 0)

    return {
        "mode": mode,
        "status": bool(status),
        "t": t_arr,
        "theta": theta_arr,
        "gamma_corner0": gamma_arr,
        "h_pred": np.array(h_pred_log, dtype=float),
        "h_actual": np.array(h_actual_log, dtype=float),
        "u_delta_norm": u_delta_arr,
        "violation": violation_arr,
        "violation_cumsum": np.cumsum(violation_arr),
        "u_nom": np.array(u_nom_log[: len(t_arr)], dtype=float),
        "u_cmd": np.array(u_cmd_log[: len(t_arr)], dtype=float),
        "traj_q": np.array(q_log, dtype=float),
        "cbf_residual": np.array(cbf_residual_log[: len(t_arr)], dtype=float),
        "qp_time_s": np.array(qp_time_log[: len(t_arr)], dtype=float),
        "max_force": float(np.max(gamma_arr)) if gamma_arr.size else 0.0,
        "n_viol": int(np.sum(violation_arr)),
        "mean_qp_solve_time": float(np.mean(qp_time_log)) if len(qp_time_log) else 0.0,
        "success_tipover": success_tipover,
        "success_safe": success_safe,
        "success": bool(success_tipover and success_safe),
    }




def print_summary_table(results):
    print("\nSummary")
    print("mode            success  tipover  max_force   n_viol  mean_qp_s")
    print("---------------------------------------------------------------")
    for res in results:
        mode_name = {
            "nominal": "iLQR_only",
            "cbf": "CBF",
            "robust_cbf": "rCBF",
        }[res["mode"]]
        print(
            f"{mode_name:14s} {str(res['success']):7s} {str(res['success_tipover']):7s} "
            f"{res['max_force']:10.4f} {res['n_viol']:7d} {res['mean_qp_solve_time']:10.6f}"
        )


def print_violation_status(results):
    print("\n[violation summary]")
    for mode in ("cbf", "robust_cbf"):
        row = next((r for r in results if r.get("mode") == mode), None)
        if row is None:
            continue
        vio = np.asarray(row.get("violation", np.zeros(0, dtype=int)), dtype=int)
        n = int(vio.size)
        n_viol = int(np.sum(vio)) if n > 0 else 0
        any_viol = n_viol > 0
        label = "CBF" if mode == "cbf" else "rCBF"
        print(
            f"  {label}: violation={'YES' if any_viol else 'NO'} "
            f"(n_viol={n_viol}/{n}, max_force={float(row.get('max_force', np.nan)):.6f})"
        )





