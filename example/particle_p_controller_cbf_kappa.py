import copy
import numpy as np
import time
from example.cbf_qp import minimize_qp


from src.robots.particle.model_linear import Particle
from src.cpp.particle_residual import make_particle_residual
from src.simulator.simulator import Simulator












def particle_pd_controller(force_error, force_error_prev, dt, p_gain, d_gain, u_ff):
    """Force-error PD controller with feedforward for gravity impulse."""
    d_force = (force_error - force_error_prev) / dt if dt > 0.0 else 0.0
    u = np.zeros(3)
    u[2] = u_ff - (p_gain * force_error + d_gain * d_force)
    return u


def step_with_u(sim, t, u_cmd, diff_sol=False):
    sim.traj.u[t][:] = u_cmd
    sim.traj.w[t][:] = 0.0
    status = sim.step(t, diff_sol=diff_sol)
    return status, sim


def _solve_qp_casadi(H, g, A, b, lbx, ubx, timing=None, qp_stats=None):
    """Solve a convex QP with SciPy SLSQP. Returns (x, success, info)."""
    attempts = []
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

        res = minimize_qp(
            H, g, A, b, lbx, ubx, x0, maxiter=200, timing=timing
        )
        x_opt = np.asarray(res.x, dtype=float).reshape(-1)
        feasible = bool(np.all(A @ x_opt - b <= 1e-6))
        success = bool(res.success) and feasible
        return_status = str(getattr(res, "message", "unknown"))
        attempts.append(
            {
                "solver": "scipy_slsqp",
                "success": success,
                "return_status": return_status,
            }
        )
        if success:
            if qp_stats is not None:
                qp_stats["success"] += 1
            return (
                x_opt,
                True,
                {
                    "solver": "scipy_slsqp",
                    "success": True,
                    "return_status": return_status,
                    "attempts": attempts,
                },
            )
    except Exception as e:
        attempts.append({"solver": "scipy_slsqp", "success": False, "error": str(e)})
    if qp_stats is not None:
        qp_stats["fail"] += 1
    return np.zeros(H.shape[0]), False, {"solver": None, "success": False, "attempts": attempts}


def _contact_force_and_jacobians(sim, t, u_cmd, contact_index=0):
    """Run a one-step diff solve to get F_normal and its jacobians.

    This uses Dojo's implicit contact solve and AutoDiff Jacobians exposed via sim.grad.
    """
    t0 = time.perf_counter()
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
    t1 = time.perf_counter()
    status, sim_nom = step_with_u(sim_nom, t, u_cmd, diff_sol=True)
    t2 = time.perf_counter()
    if not status:
        return False, sim_nom, 0.0, None

    gamma = (
        float(sim_nom.traj.gamma[t][contact_index])
        if len(sim_nom.traj.gamma[t])
        else 0.0
    )
    dgamma_dq = np.array(sim_nom.grad.dgamma1_dq1[t][contact_index, :], dtype=float)
    dgamma_dv = np.array(sim_nom.grad.dgamma1_dv1[t][contact_index, :], dtype=float)
    dgamma_du = np.array(sim_nom.grad.dgamma1_du1[t][contact_index, :], dtype=float)
    v_nom = np.array(sim_nom.traj.v[t], dtype=float)
    v_next = np.array(sim_nom.traj.v[t + 1], dtype=float)
    a_nom = (v_next - v_nom) / sim_nom.h
    timing = getattr(sim, "_diff_timing", None)
    if timing is not None:
        timing["total_s"] += t2 - t0
        timing["sync_s"] += t1 - t0
        timing["step_s"] += t2 - t1
        timing["calls"] += 1

    return True, sim_nom, gamma, (dgamma_dq, dgamma_dv, dgamma_du, v_nom, a_nom)


def safety_filter(sim, t, u_pd, params):
    """QP-based CBF safety filter enforcing F_normal <= F_max.

    Notes:
    - Uses a discrete-time CBF on h_{k+1} (one-step lookahead) to improve
      forward-invariance behavior with implicit contact solves.
    - Optionally adds a pre-impact normal-velocity constraint to reduce
      impact spikes (impulse-like events).
    """
    u_pd = np.array(u_pd, dtype=float).copy()
    u_min = np.array(params["u_min"], dtype=float)
    u_max = np.array(params["u_max"], dtype=float)
    u_pd = np.clip(u_pd, u_min, u_max)

    status, sim_nom, gamma_nom, jac = _contact_force_and_jacobians(
        sim, t, u_pd, contact_index=params["contact_index"]
    )
    if not status or jac is None:
        return False, sim_nom, u_pd, None

    dgamma_dq, dgamma_dv, dgamma_du, v_nom, a_nom = jac
    stats = params.get("_cbf_stats")
    if stats is not None:
        stats["gamma_nom"].append(float(gamma_nom))
        stats["dgamma_du_norm"].append(float(np.linalg.norm(dgamma_du)))
        stats["dgamma_du_vec"].append(np.array(dgamma_du, dtype=float))
        stats["u_pd"].append(np.array(u_pd, dtype=float))
    # Safety function: h = F_max - F_normal.
    h_next_nom = params["F_max"] - gamma_nom
    gamma_prev = (
        float(sim.traj.gamma[t - 1][params["contact_index"]])
        if t > 0 and len(sim.traj.gamma[t - 1])
        else 0.0
    )
    if params.get("use_predicted_h_curr", False):
        h_curr = float(params.get("_h_prev_pred", params["F_max"] - gamma_prev))
    else:
        h_curr = params["F_max"] - gamma_prev

    # Discrete-time CBF: h_{k+1} >= (1 - alpha) * h_k
    # Linearize h_{k+1}(u) around u_pd:
    #   h_{k+1}(u) ≈ h_next_nom + (-dgamma_du) @ (u - u_pd)
    a_cbf = -dgamma_du.reshape(1, -1)
    rhs_cbf = (
        (1.0 - params["cbf_alpha"]) * h_curr
        - h_next_nom
        + float(a_cbf.reshape(-1) @ u_pd)
    )
    # Robust CBF tightening term Delta_k:
    #   h_pred_next >= (1-alpha) h_k + Delta_k
    # Delta_k can be configured by robust_mode.
    robust_margin = float(params.get("robust_margin", 0.0))
    robust_delta_base = float(params.get("robust_base", 0.0))
    robust_delta_ema = 0.0
    robust_delta_hk = 0.0
    robust_delta_hnom = 0.0
    robust_delta_jac = 0.0
    robust_delta_adapt = 0.0
    robust_delta_under_gate = 0.0
    robust_delta_under_gate_nom = 0.0
    near_gate = 0.0
    prev_err_under = 0.0
    robust_delta = 0.0
    if params.get("use_robust_cbf", False):
        prev_err = float(params.get("_robust_err_ema", 0.0))
        prev_under = float(params.get("_robust_under_ema", prev_err))
        prev_err_under = prev_under
        prev_over = float(params.get("_robust_over_ema", 0.0))
        robust_mode = str(params.get("robust_mode", "ema_max"))
        robust_delta_ema = max(robust_margin, prev_err)
        if robust_mode == "state_dependent":
            # Increase tightening near boundary and when nominal one-step safety is weak.
            h_scale = max(float(params.get("robust_h_scale", 1e-3)), 1e-12)
            near_gate = 1.0 / (1.0 + abs(float(h_curr)) / h_scale)
            robust_delta_hk = float(params.get("robust_hk_gain", 0.0)) * near_gate
            robust_delta_hnom = float(params.get("robust_hnom_gain", 0.0)) * max(
                0.0, -float(h_next_nom)
            )
            robust_delta_jac = float(params.get("robust_jac_gain", 0.0)) * float(
                np.linalg.norm(dgamma_du)
            )
            robust_delta = (
                robust_delta_base
                + robust_delta_ema
                + robust_delta_hk
                + robust_delta_hnom
                + robust_delta_jac
            )
        elif robust_mode == "under_gate":
            # Simplified robust mode:
            #   Delta_k = robust_margin + near_gate * prev_err_under
            # where near_gate = 1 / (1 + |h_curr| / h_scale).
            # Sanity check:
            # - Large |h_curr| => near_gate -> 0  => Delta_k ~ robust_margin.
            # - Near boundary (h_curr ~ 0) => near_gate -> 1 => Delta_k ~ robust_margin + prev_err_under.
            h_scale = max(float(params.get("robust_h_scale", 1e-3)), 1e-12)
            near_gate = 1.0 / (1.0 + abs(float(h_curr)) / h_scale)
            robust_delta_under_gate = near_gate * prev_under
            robust_delta = robust_margin + robust_delta_under_gate
        elif robust_mode == "under_gate_plus":
            # Slightly richer simplified robust mode:
            #   Delta_k = robust_margin + near_gate * (prev_err_under + c_nom * max(0, -h_next_nom))
            # where c_nom = robust_hnom_gain.
            # Sanity check:
            # - Large |h_curr| => near_gate -> 0 => Delta_k ~ robust_margin.
            # - Near boundary => Delta_k grows with under-memory and nominal deficit.
            h_scale = max(float(params.get("robust_h_scale", 1e-3)), 1e-12)
            near_gate = 1.0 / (1.0 + abs(float(h_curr)) / h_scale)
            h_nom_deficit = max(0.0, -float(h_next_nom))
            under_gate_core = (
                prev_under + float(params.get("robust_hnom_gain", 0.02)) * h_nom_deficit
            )
            robust_delta_under_gate = near_gate * prev_under
            robust_delta_under_gate_nom = (
                near_gate * float(params.get("robust_hnom_gain", 0.02)) * h_nom_deficit
            )
            robust_delta = robust_margin + near_gate * under_gate_core
        elif robust_mode == "robust_adaptive":
            # Use one-sided error estimate (optimistic prediction only) and adapt
            # tightening strength mainly near boundary.
            h_scale = max(float(params.get("robust_h_scale", 1e-3)), 1e-12)
            # near_gate in (0,1]: close to boundary (|h_curr| ~ 0) => near_gate ~ 1,
            # far from boundary => near_gate -> 0.
            near_gate = 1.0 / (1.0 + abs(float(h_curr)) / h_scale)
            # Three additive signals used for adaptive tightening:
            # 1) boundary proximity term (more care near h=0)
            robust_delta_hk = float(params.get("robust_hk_gain", 0.0)) * near_gate
            # 2) nominal safety deficit term (if nominal prediction is already unsafe)
            robust_delta_hnom = float(params.get("robust_hnom_gain", 0.0)) * max(
                0.0, -float(h_next_nom)
            )
            # 3) local sensitivity term (large |dgamma/du| => prediction more fragile)
            robust_delta_jac = float(params.get("robust_jac_gain", 0.0)) * float(
                np.linalg.norm(dgamma_du)
            )
            # eta is an adaptive gain on the state/sensitivity terms.
            # - If underestimation stays above target, eta increases.
            # - If underestimation is small, eta decreases.
            eta_prev = float(params.get("_robust_eta", 1.0))
            eta_lr = float(params.get("robust_eta_lr", 0.2))
            eta_min = float(params.get("robust_eta_min", 0.5))
            eta_max = float(params.get("robust_eta_max", 2.0))
            under_target = max(float(params.get("robust_under_target", 5e-4)), 1e-12)
            # Learning step is normalized by target under-error and gated by near_gate,
            # so adaptation is strongest near the active safety boundary.
            eta_step = eta_lr * ((prev_under - under_target) / under_target) * near_gate
            eta = float(np.clip(eta_prev + eta_step, eta_min, eta_max))
            params["_robust_eta"] = eta
            # over_comp discounts tightening when model is already conservative
            # (h_actual > h_pred), preventing unnecessary over-tightening.
            over_comp = float(params.get("robust_over_comp_gain", 0.0)) * prev_over
            # Adaptive component on top of one-sided under-estimation memory.
            robust_delta_adapt = eta * (
                robust_delta_hk + robust_delta_hnom + robust_delta_jac
            )
            # Final robust tightening used in RHS:
            # Delta_k = base + under_memory + adaptive_part - conservative_comp + fixed_margin
            # Clamp at 0 to avoid loosening the baseline CBF inequality.
            robust_delta = max(
                0.0,
                robust_delta_base
                + prev_under
                + robust_delta_adapt
                - over_comp
                + robust_margin,
            )
        else:
            robust_delta = robust_delta_ema
        rhs_cbf += robust_delta

    # Optional RHS smoothing to reduce sensitivity to kappa / linearization jitter.
    if params.get("use_rhs_smoothing", False):
        rhs_prev = float(params.get("_rhs_cbf_prev", rhs_cbf))
        smooth_alpha = float(params.get("rhs_smooth_alpha", 0.2))
        rhs_cbf = smooth_alpha * rhs_cbf + (1.0 - smooth_alpha) * rhs_prev
        params["_rhs_cbf_prev"] = rhs_cbf
    A_list = [(-a_cbf).reshape(1, -1)]
    b_list = [np.array([-rhs_cbf], dtype=float)]

    # Optional impact-velocity constraint (pre-impact normal velocity bound).
    if params.get("use_impact_constraint", True):
        phi = sim.model.signed_distance(sim.traj.q[t + 1])
        if float(phi[-1]) <= params.get("phi_thresh", 1e-4):
            xdot0, _, B, _, u0 = sim_nom.local_affine_xdot(t)
            a_v = B[2, :].reshape(1, -1)
            rhs_v = -params["v_n_max"] - float(xdot0[2]) + float(a_v.reshape(-1) @ u0)
            A_list.append((-a_v).reshape(1, -1))
            b_list.append(np.array([-rhs_v], dtype=float))

    A = np.vstack(A_list)
    b = np.concatenate(b_list)
    rhs_h = float((1.0 - params["cbf_alpha"]) * h_curr)
    if stats is not None:
        cons_residual_pd = A @ u_pd - b
        stats.setdefault("u_pd_max_violation", []).append(
            float(np.max(cons_residual_pd))
        )
        stats.setdefault("u_pd_min_margin", []).append(float(np.min(b - A @ u_pd)))
        stats.setdefault("u_pd_active_count_tol1e8", []).append(
            int(np.sum(np.abs(b - A @ u_pd) <= 1e-8))
        )

    use_slack = params.get("use_slack", False)
    slack_weight = float(params.get("slack_weight", 1e4))
    Q = np.array(params["qp_Q"], dtype=float)
    if Q.ndim == 1:
        Q = np.diag(Q)

    cbf_row_residual = np.nan
    cbf_soft_row_residual = np.nan
    cbf_row_active = False
    cbf_row_violation = np.nan
    slack_cbf = 0.0
    qp_return_status = "not_run"
    qp_solver = "none"

    if use_slack:
        # Decision variables: [u; s], s >= 0 (one slack per constraint).
        m = A.shape[0]
        H = np.zeros((u_pd.size + m, u_pd.size + m), dtype=float)
        H[: u_pd.size, : u_pd.size] = Q
        H[u_pd.size :, u_pd.size :] = slack_weight * np.eye(m)
        g = np.zeros(u_pd.size + m, dtype=float)
        g[: u_pd.size] = -Q @ u_pd
        A_aug = np.zeros((A.shape[0], u_pd.size + m), dtype=float)
        A_aug[:, : u_pd.size] = A
        A_aug[:, u_pd.size :] = -np.eye(m)  # -s on the left (A u - s <= b)
        lbx = np.concatenate([u_min, np.zeros(m)])
        ubx = np.concatenate([u_max, np.full(m, np.inf)])
        x_opt, ok, solve_info = _solve_qp_casadi(
            H,
            g,
            A_aug,
            b,
            lbx,
            ubx,
            timing=params.get("_qp_timing"),
            qp_stats=params.get("_qp_stats"),
        )
        u_safe = x_opt[: u_pd.size] if ok else u_pd
        if ok:
            slack = x_opt[u_pd.size :]
            if slack.size:
                slack_cbf = float(slack[0])
            # Hard residual (without slack), useful for diagnosing relaxation.
            cbf_row_residual = float((A @ u_safe - b)[0])
            # Soft residual (with slack), should be <= 0 at optimum.
            cbf_soft_row_residual = float((A @ u_safe - slack - b)[0])
            cbf_row_violation = max(cbf_soft_row_residual, 0.0)
            cbf_row_active = abs(cbf_soft_row_residual) <= 1e-8
        qp_return_status = str(solve_info.get("return_status", "unknown"))
        qp_solver = str(solve_info.get("solver", "none"))
        if stats is not None and not ok:
            # Distinguish likely infeasibility from solver failure by checking
            # whether the fallback nominal input satisfies linearized constraints.
            cons_residual = A @ u_pd - b
            min_margin = float(np.min(b - A @ u_pd))
            active_tol = 1e-8
            active_count = int(np.sum(np.abs(b - A @ u_pd) <= active_tol))
            stats["qp_fail_details"].append(
                {
                    "t": int(t),
                    "mode": "slack",
                    "max_violation_u_pd": float(np.max(cons_residual)),
                    "u_pd_feasible_tol1e9": bool(np.max(cons_residual) <= 1e-9),
                    "min_margin_u_pd": min_margin,
                    "active_count_u_pd_tol1e8": active_count,
                    "A_row_inf_max": float(
                        np.max(np.linalg.norm(A, ord=np.inf, axis=1))
                    ),
                    "b_abs_max": float(np.max(np.abs(b))),
                    "solve_info": solve_info,
                }
            )
        if stats is not None:
            slack = x_opt[u_pd.size :] if ok else np.zeros(m)
            stats["slack_max"].append(float(np.max(slack)) if slack.size else 0.0)
            stats["qp_ok"].append(bool(ok))
    else:
        # Decision variables: u only
        H = Q
        g = -Q @ u_pd
        lbx = u_min
        ubx = u_max
        x_opt, ok, solve_info = _solve_qp_casadi(
            H,
            g,
            A,
            b,
            lbx,
            ubx,
            timing=params.get("_qp_timing"),
            qp_stats=params.get("_qp_stats"),
        )
        u_safe = x_opt if ok else u_pd
        if ok:
            cbf_row_residual = float((A @ u_safe - b)[0])
            cbf_soft_row_residual = cbf_row_residual
            cbf_row_violation = max(cbf_row_residual, 0.0)
            cbf_row_active = abs(cbf_row_residual) <= 1e-8
        qp_return_status = str(solve_info.get("return_status", "unknown"))
        qp_solver = str(solve_info.get("solver", "none"))
        if stats is not None:
            stats["slack_max"].append(0.0)
            stats["qp_ok"].append(bool(ok))
            if not ok:
                cons_residual = A @ u_pd - b
                min_margin = float(np.min(b - A @ u_pd))
                active_tol = 1e-8
                active_count = int(np.sum(np.abs(b - A @ u_pd) <= active_tol))
                stats["qp_fail_details"].append(
                    {
                        "t": int(t),
                        "mode": "hard",
                        "max_violation_u_pd": float(np.max(cons_residual)),
                        "u_pd_feasible_tol1e9": bool(np.max(cons_residual) <= 1e-9),
                        "min_margin_u_pd": min_margin,
                        "active_count_u_pd_tol1e8": active_count,
                        "A_row_inf_max": float(
                            np.max(np.linalg.norm(A, ord=np.inf, axis=1))
                        ),
                        "b_abs_max": float(np.max(np.abs(b))),
                        "solve_info": solve_info,
                    }
                )

    status, sim = step_with_u(sim, t, u_safe)
    if not status:
        return False, sim, u_safe, None
    gamma_safe = float(sim.traj.gamma[t][0]) if len(sim.traj.gamma[t]) else 0.0

    if stats is not None:
        stats["u_safe"].append(np.array(u_safe, dtype=float))
        stats["delta_u_norm"].append(float(np.linalg.norm(u_safe - u_pd)))
        delta_gamma_pred = float(a_cbf.reshape(-1) @ (u_safe - u_pd))
        delta_gamma_actual = float(gamma_safe - gamma_nom)
        stats["gamma_safe"].append(float(gamma_safe))
        stats["gamma_safe_minus_nom"].append(delta_gamma_actual)
        stats["delta_gamma_pred"].append(delta_gamma_pred)
        stats["delta_gamma_actual"].append(delta_gamma_actual)
        if abs(delta_gamma_pred) > 1e-12:
            stats["delta_gamma_ratio"].append(delta_gamma_actual / delta_gamma_pred)

    # Linearization error: actual vs predicted h_{k+1}.
    h_next_actual = params["F_max"] - gamma_safe
    h_next_pred = h_next_nom + float(a_cbf.reshape(-1) @ (u_safe - u_pd))
    delta_h_linear = float(h_next_pred - h_next_nom)
    err_pred = float(h_next_actual - h_next_pred)
    margin = float(h_next_pred - rhs_h)
    gamma_pred = float(params["F_max"] - h_next_pred)
    if params.get("use_predicted_h_curr", False):
        # Feed predicted h into next-step CBF constraint (experimental mode).
        params["_h_prev_pred"] = float(h_next_pred)
    lin_err = abs(h_next_actual - h_next_pred)
    if "_lin_err_log" in params:
        params["_lin_err_log"].append((t, lin_err))
    if stats is not None:
        stats["rhs_cbf"].append(float(rhs_cbf))
        stats["h_next_nom"].append(float(h_next_nom))
        stats["h_next_pred"].append(float(h_next_pred))
        stats["h_next_actual"].append(float(h_next_actual))
        if gamma_safe > params["F_max"]:
            stats["violations"].append(
                {
                    "t": int(t),
                    "gamma_nom": float(gamma_nom),
                    "gamma_safe": float(gamma_safe),
                    "dgamma_du_norm": float(np.linalg.norm(dgamma_du)),
                    "delta_gamma_pred": float(a_cbf.reshape(-1) @ (u_safe - u_pd)),
                    "delta_gamma_actual": float(gamma_safe - gamma_nom),
                    "h_next_pred": float(h_next_pred),
                    "h_next_actual": float(h_next_actual),
                }
            )

    step_log = params.get("_step_log")
    if step_log is not None:
        step_log.append(
            {
                "k": int(t),
                "time_s": float(t * sim.h),
                "kappa": float(params["kappa_tol"]),
                "F_max": float(params["F_max"]),
                "alpha": float(params["cbf_alpha"]),
                "h_k": float(h_curr),
                "rhs": rhs_h,
                "h_next_nom": float(h_next_nom),
                "delta_h_linear": delta_h_linear,
                "h_next_pred": float(h_next_pred),
                "h_next_actual": float(h_next_actual),
                "delta_h_err_pred_actual": err_pred,
                "margin": margin,
                "gamma_nom": float(gamma_nom),
                "gamma_pred": gamma_pred,
                "gamma_actual": float(gamma_safe),
                "u_nom_x": float(u_pd[0]),
                "u_nom_y": float(u_pd[1]),
                "u_nom_z": float(u_pd[2]),
                "u_star_x": float(u_safe[0]),
                "u_star_y": float(u_safe[1]),
                "u_star_z": float(u_safe[2]),
                "u_delta_norm": float(np.linalg.norm(u_safe - u_pd)),
                "qp_ok": int(ok),
                "qp_solver": qp_solver,
                "qp_return_status": qp_return_status,
                "use_slack": int(use_slack),
                "slack_cbf": float(slack_cbf),
                "cbf_row_residual_hard": float(cbf_row_residual),
                "cbf_row_residual_soft": float(cbf_soft_row_residual),
                "cbf_row_violation": float(cbf_row_violation),
                "cbf_constraint_active": int(cbf_row_active),
                "robust_delta": float(robust_delta),
                "robust_delta_base": float(robust_delta_base),
                "robust_delta_ema": float(robust_delta_ema),
                "robust_delta_hk": float(robust_delta_hk),
                "robust_delta_hnom": float(robust_delta_hnom),
                "robust_delta_jac": float(robust_delta_jac),
                "robust_delta_adapt": float(robust_delta_adapt),
                "robust_delta_under_gate": float(robust_delta_under_gate),
                "robust_delta_under_gate_nom": float(robust_delta_under_gate_nom),
                "near_gate": float(near_gate),
                "prev_err_under": float(prev_err_under),
                "robust_margin": float(robust_margin),
                "robust_eta": float(params.get("_robust_eta", 1.0)),
            }
        )

    # Update robust error estimate for next step using actual vs predicted h_{k+1}.
    if params.get("use_robust_cbf", False):
        # err_signed = h_actual - h_pred:
        # < 0 means optimistic prediction (unsafe direction, underestimated force),
        # > 0 means conservative prediction.
        err_signed = float(h_next_actual - h_next_pred)
        err = abs(err_signed)
        under = max(0.0, -err_signed)  # optimistic prediction on h (unsafe side)
        over = max(0.0, err_signed)  # conservative prediction on h
        # EMA keeps a low-pass memory of prediction mismatch:
        # - _robust_under_ema drives safety tightening in robust_adaptive mode.
        # - _robust_over_ema can relax excessive tightening via over_comp.
        # - _robust_err_ema is a symmetric magnitude summary.
        beta = float(params.get("robust_beta", 0.2))
        params["_robust_err_ema"] = (1.0 - beta) * float(
            params.get("_robust_err_ema", 0.0)
        ) + beta * err
        params["_robust_under_ema"] = (1.0 - beta) * float(
            params.get("_robust_under_ema", 0.0)
        ) + beta * under
        params["_robust_over_ema"] = (1.0 - beta) * float(
            params.get("_robust_over_ema", 0.0)
        ) + beta * over
    params["_u_pd_prev"] = np.array(u_pd, dtype=float)
    return True, sim, u_safe, gamma_safe


def build_sim(particle, residuals, dt, num_steps, kappa_tol, diff_sol=False):
    r, rz, rtheta = residuals
    sim = Simulator(
        particle,
        num_steps,
        h=dt,
        diff_sol=diff_sol,
        residual=r,
        jacobian_z=rz,
        jacobian_theta=rtheta,
        temp_cmd=np.zeros(particle.nu),
        kappa_tol=kappa_tol,
    )
    return sim


def run_sim(params, use_cbf):
    particle = params["particle"]
    dt = params["dt"]
    num_steps = params["num_steps"]
    gamma_ref = params["gamma_ref"]
    p_gain = params["p_gain"]
    d_gain = params["d_gain"]
    kappa_tol = params["kappa_tol"]
    residuals = params["residuals"]
    u_min = np.array(params["u_min"], dtype=float)
    u_max = np.array(params["u_max"], dtype=float)

    sim = build_sim(particle, residuals, dt, num_steps, kappa_tol, diff_sol=False)
    sim.traj.reset()
    sim.grad.reset()
    sim.prev_z = None
    sim.set_state(params["q0"], params["v0"], 0)
    sim._diff_timing = params.get("_diff_timing")
    sim_diff = build_sim(particle, residuals, dt, num_steps, kappa_tol, diff_sol=True)
    sim_diff.traj.reset()
    sim_diff.grad.reset()
    sim_diff.prev_z = None
    sim_diff._diff_timing = sim._diff_timing
    sim._diff_sim = sim_diff
    if params.get("use_predicted_h_curr", False):
        params["_h_prev_pred"] = float(params["F_max"])

    gamma_log = []
    phi_log = []
    z_log = []
    u_log = []
    status = True

    gamma_curr = 0.0
    force_error_prev = gamma_ref[0] - gamma_curr
    for t in range(num_steps):
        # One-step lookahead to offset solver latency.
        ref_idx = min(t + 1, num_steps - 1)
        force_error = gamma_ref[ref_idx] - gamma_curr
        # Feedforward keeps contact force near gamma_ref in impulse units.
        u_ff = dt * particle.mb * particle.gravity - gamma_ref[ref_idx]
        u_nom = particle_pd_controller(
            force_error, force_error_prev, dt, p_gain, d_gain, u_ff
        )
        if not np.isfinite(u_nom[2]):
            u_nom[2] = 0.0
        u_nom = np.clip(u_nom, u_min, u_max)

        # Use PD controller output as the nominal/reference input.
        u_ref = u_nom.copy()

        if use_cbf:
            status, sim_next, u_cmd, gamma_val = safety_filter(sim, t, u_ref, params)
            sim = sim_next
            if gamma_val is None:
                gamma_val = 0.0
        else:
            u_cmd = u_ref
            sim.traj.u[t][:] = u_cmd
            sim.traj.w[t][:] = 0.0
            status = sim.step(t, diff_sol=False)
            gamma_val = float(sim.traj.gamma[t][0]) if len(sim.traj.gamma[t]) else 0.0

        if not status:
            break

        gamma_log.append(gamma_val)
        gamma_curr = gamma_val
        force_error_prev = force_error
        phi = sim.model.signed_distance(sim.traj.q[t + 1])
        phi_log.append(float(phi[-1]))
        z_log.append(float(sim.traj.q[t + 1][2]))
        u_log.append(float(u_cmd[2]))

    steps_done = len(gamma_log)
    return {
        "status": status,
        "t": np.arange(0.0, dt * steps_done, dt),
        "z": np.array(z_log),
        "gamma": np.array(gamma_log),
        "phi": np.array(phi_log),
        "u_z": np.array(u_log),
        "gamma_ref": gamma_ref[:steps_done],
        "traj_q": np.array(sim.traj.q[: steps_done + 1]),
        "traj_v": np.array(sim.traj.v[: steps_done + 1]),
        "sim_model": sim.model,
        "qp_timing": params.get("_qp_timing"),
        "lin_err_log": params.get("_lin_err_log"),
        "F_max": float(params["F_max"]),
    }




