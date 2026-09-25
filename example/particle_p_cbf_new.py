#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from example.cbf_trust_region import (
    box_trust_region_bounds,
    box_trust_region_metrics,
)
from example.particle_p_controller_cbf_kappa import (
    _contact_force_and_jacobians,
    _solve_qp_casadi,
    build_sim,
    particle_pd_controller,
    step_with_u,
)
from src.cpp.particle_residual import make_particle_residual
from src.robots.particle.model_linear import Particle


def _build_base_params(kappa: float, alpha: float, f_max: float, seed: int) -> dict:
    np.random.seed(seed)
    particle = Particle(nq=3, nc=1, nu=3, nw=0, nb=4, mb=1, mu=0.5, gravity=9.81)
    residuals = make_particle_residual(particle)

    dt = 0.01
    num_steps = 400
    t = np.arange(0.0, dt * num_steps, dt)
    gamma_ref = 0.225 + 0.075 * np.sin(2.0 * np.pi * 0.5 * t)

    q0 = particle.nominal_configuration()
    q0[2] = particle.r
    v0 = np.zeros(particle.nq)

    params = {
        "particle": particle,
        "dt": dt,
        "num_steps": num_steps,
        "gamma_ref": gamma_ref,
        "p_gain": 0.1,
        "d_gain": 0.0,
        "kappa_tol": float(kappa),
        "residuals": residuals,
        "q0": q0,
        "v0": v0,
        "u_min": np.array([-10.0, -10.0, -10.0]),
        "u_max": np.array([10.0, 10.0, 10.0]),
        "F_max": float(f_max),
        "cbf_alpha": float(alpha),
        "qp_Q": np.array([1.0, 1.0, 1.0]),
        "use_slack": False,
        "slack_weight": 1e4,
        "contact_index": 0,
        "use_impact_constraint": False,
        "phi_thresh": 1e-4,
        "v_n_max": 0.2,
        "use_robust_cbf": False,
        "delta_kappa": 0.0,
        "trust_region_radius": None,
        "_qp_timing": {"total_s": 0.0, "by_solver_s": {}, "calls": 0},
        "_diff_timing": {"total_s": 0.0, "sync_s": 0.0, "step_s": 0.0, "calls": 0},
        "_lin_err_log": [],
        "_qp_stats": {"success": 0, "fail": 0},
        "_cbf_stats": {
            "gamma_nom": [],
            "gamma_safe": [],
            "gamma_safe_minus_nom": [],
            "delta_gamma_pred": [],
            "delta_gamma_actual": [],
            "delta_gamma_ratio": [],
            "delta_u_norm": [],
            "u_delta_linf": [],
            "trust_region_ratio": [],
            "trust_region_active": [],
            "dgamma_du_norm": [],
            "dgamma_du_vec": [],
            "u_pd": [],
            "u_safe": [],
            "rhs_cbf": [],
            "h_next_nom": [],
            "h_next_pred": [],
            "h_next_actual": [],
            "slack_max": [],
            "qp_ok": [],
            "qp_fail_details": [],
            "violations": [],
            "delta_kappa_used": [],
        },
        "_step_log": [],
    }
    return params


def _estimate_delta_kappa(
    rows: list[dict],
    *,
    boundary_h: float,
    quantile: float,
    statistic: str,
) -> tuple[float, dict]:
    relevant = []
    all_under = []
    for row in rows:
        under = max(0.0, float(row["h_next_pred"]) - float(row["h_next_actual"]))
        all_under.append(under)
        h_k = float(row["h_k"])
        h_next_nom = float(row["h_next_nom"])
        if (h_k <= boundary_h) or (h_next_nom <= boundary_h):
            relevant.append(under)

    use = relevant if relevant else all_under
    if use:
        if statistic == "max":
            delta = float(np.max(use))
        else:
            delta = float(np.quantile(np.asarray(use, dtype=float), quantile))
    else:
        delta = 0.0

    info = {
        "boundary_h": float(boundary_h),
        "quantile": float(quantile),
        "statistic": statistic,
        "num_rows": int(len(rows)),
        "num_boundary_rows": int(len(relevant)),
        "delta_kappa_hat": float(delta),
        "max_under_all": float(np.max(all_under)) if all_under else 0.0,
        "max_under_boundary": float(np.max(relevant)) if relevant else 0.0,
    }
    return delta, info


def safety_filter_fixed_delta(sim, t: int, u_pd: np.ndarray, params: dict):
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
    del dgamma_dq, dgamma_dv, v_nom, a_nom
    stats = params.get("_cbf_stats")
    if stats is not None:
        stats["gamma_nom"].append(float(gamma_nom))
        stats["dgamma_du_norm"].append(float(np.linalg.norm(dgamma_du)))
        stats["dgamma_du_vec"].append(np.array(dgamma_du, dtype=float))
        stats["u_pd"].append(np.array(u_pd, dtype=float))

    h_next_nom = float(params["F_max"] - gamma_nom)
    gamma_prev = (
        float(sim.traj.gamma[t - 1][params["contact_index"]])
        if t > 0 and len(sim.traj.gamma[t - 1])
        else 0.0
    )
    h_curr = float(params["F_max"] - gamma_prev)

    a_cbf = -dgamma_du.reshape(1, -1)
    rhs_h = float((1.0 - params["cbf_alpha"]) * h_curr)
    rhs_cbf = rhs_h - h_next_nom + float(a_cbf.reshape(-1) @ u_pd)

    robust_delta = 0.0
    if params.get("use_robust_cbf", False):
        robust_delta = max(0.0, float(params.get("delta_kappa", 0.0)))
        rhs_cbf += robust_delta

    A = (-a_cbf).reshape(1, -1)
    b = np.array([-rhs_cbf], dtype=float)

    use_slack = bool(params.get("use_slack", False))
    slack_weight = float(params.get("slack_weight", 1e4))
    Q = np.array(params["qp_Q"], dtype=float)
    if Q.ndim == 1:
        Q = np.diag(Q)
    trust_region_radius = params.get("trust_region_radius")
    qp_u_min, qp_u_max = box_trust_region_bounds(
        u_pd, u_min, u_max, trust_region_radius
    )

    cbf_row_residual = np.nan
    cbf_soft_row_residual = np.nan
    cbf_row_active = False
    cbf_row_violation = np.nan
    slack_cbf = 0.0
    qp_return_status = "not_run"
    qp_solver = "none"

    if use_slack:
        m = A.shape[0]
        H = np.zeros((u_pd.size + m, u_pd.size + m), dtype=float)
        H[: u_pd.size, : u_pd.size] = Q
        H[u_pd.size :, u_pd.size :] = slack_weight * np.eye(m)
        g = np.zeros(u_pd.size + m, dtype=float)
        g[: u_pd.size] = -Q @ u_pd
        A_aug = np.zeros((A.shape[0], u_pd.size + m), dtype=float)
        A_aug[:, : u_pd.size] = A
        A_aug[:, u_pd.size :] = -np.eye(m)
        lbx = np.concatenate([qp_u_min, np.zeros(m)])
        ubx = np.concatenate([qp_u_max, np.full(m, np.inf)])
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
            cbf_row_residual = float((A @ u_safe - b)[0])
            cbf_soft_row_residual = float((A @ u_safe - slack - b)[0])
            cbf_row_violation = max(cbf_soft_row_residual, 0.0)
            cbf_row_active = abs(cbf_soft_row_residual) <= 1e-8
        qp_return_status = str(solve_info.get("return_status", "unknown"))
        qp_solver = str(solve_info.get("solver", "none"))
        if stats is not None:
            slack = x_opt[u_pd.size :] if ok else np.zeros(m)
            stats["slack_max"].append(float(np.max(slack)) if slack.size else 0.0)
            stats["qp_ok"].append(bool(ok))
    else:
        H = Q
        g = -Q @ u_pd
        lbx = qp_u_min
        ubx = qp_u_max
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

    status, sim = step_with_u(sim, t, u_safe)
    if not status:
        return False, sim, u_safe, None
    gamma_safe = float(sim.traj.gamma[t][0]) if len(sim.traj.gamma[t]) else 0.0
    tr_metrics = box_trust_region_metrics(u_safe, u_pd, trust_region_radius)

    if stats is not None:
        stats["u_safe"].append(np.array(u_safe, dtype=float))
        stats["delta_u_norm"].append(float(np.linalg.norm(u_safe - u_pd)))
        stats["u_delta_linf"].append(float(tr_metrics["u_delta_linf"]))
        stats["trust_region_ratio"].append(float(tr_metrics["trust_region_ratio"]))
        stats["trust_region_active"].append(bool(tr_metrics["trust_region_active"]))
        delta_gamma_pred = float(a_cbf.reshape(-1) @ (u_safe - u_pd))
        delta_gamma_actual = float(gamma_safe - gamma_nom)
        stats["gamma_safe"].append(float(gamma_safe))
        stats["gamma_safe_minus_nom"].append(delta_gamma_actual)
        stats["delta_gamma_pred"].append(delta_gamma_pred)
        stats["delta_gamma_actual"].append(delta_gamma_actual)
        if abs(delta_gamma_pred) > 1e-12:
            stats["delta_gamma_ratio"].append(delta_gamma_actual / delta_gamma_pred)

    h_next_actual = float(params["F_max"] - gamma_safe)
    h_next_pred = float(h_next_nom + a_cbf.reshape(-1) @ (u_safe - u_pd))
    delta_h_linear = float(h_next_pred - h_next_nom)
    mismatch = float(h_next_actual - h_next_pred)
    under_prediction = max(0.0, -mismatch)
    margin = float(h_next_pred - rhs_h)
    gamma_pred = float(params["F_max"] - h_next_pred)
    lin_err = abs(mismatch)
    if "_lin_err_log" in params:
        params["_lin_err_log"].append((t, lin_err))

    if stats is not None:
        stats["rhs_cbf"].append(float(rhs_cbf))
        stats["h_next_nom"].append(float(h_next_nom))
        stats["h_next_pred"].append(float(h_next_pred))
        stats["h_next_actual"].append(float(h_next_actual))
        stats["delta_kappa_used"].append(float(robust_delta))
        if gamma_safe > params["F_max"]:
            stats["violations"].append(
                {
                    "t": int(t),
                    "gamma_nom": float(gamma_nom),
                    "gamma_safe": float(gamma_safe),
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
                "delta_h_linear": float(delta_h_linear),
                "h_next_pred": float(h_next_pred),
                "h_next_actual": float(h_next_actual),
                "delta_h_err_pred_actual": float(mismatch),
                "under_prediction": float(under_prediction),
                "margin": float(margin),
                "gamma_nom": float(gamma_nom),
                "gamma_pred": float(gamma_pred),
                "gamma_actual": float(gamma_safe),
                "u_nom_x": float(u_pd[0]),
                "u_nom_y": float(u_pd[1]),
                "u_nom_z": float(u_pd[2]),
                "u_star_x": float(u_safe[0]),
                "u_star_y": float(u_safe[1]),
                "u_star_z": float(u_safe[2]),
                "u_delta_norm": float(np.linalg.norm(u_safe - u_pd)),
                "u_delta_linf": float(tr_metrics["u_delta_linf"]),
                "trust_region_radius": (
                    float(trust_region_radius)
                    if trust_region_radius is not None
                    else float("nan")
                ),
                "trust_region_ratio": float(tr_metrics["trust_region_ratio"]),
                "trust_region_active": int(tr_metrics["trust_region_active"]),
                "qp_ok": int(ok),
                "qp_solver": qp_solver,
                "qp_return_status": qp_return_status,
                "use_slack": int(use_slack),
                "slack_cbf": float(slack_cbf),
                "cbf_row_residual_hard": float(cbf_row_residual),
                "cbf_row_residual_soft": float(cbf_soft_row_residual),
                "cbf_row_violation": float(cbf_row_violation),
                "cbf_constraint_active": int(cbf_row_active),
                "delta_kappa": float(robust_delta),
            }
        )
    return True, sim, u_safe, gamma_safe


def run_sim(params: dict, use_cbf: bool) -> dict:
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

    gamma_log = []
    phi_log = []
    z_log = []
    u_log = []
    status = True

    gamma_curr = 0.0
    force_error_prev = gamma_ref[0] - gamma_curr
    for t in range(num_steps):
        ref_idx = min(t + 1, num_steps - 1)
        force_error = gamma_ref[ref_idx] - gamma_curr
        u_ff = dt * particle.mb * particle.gravity - gamma_ref[ref_idx]
        u_nom = particle_pd_controller(
            force_error, force_error_prev, dt, p_gain, d_gain, u_ff
        )
        if not np.isfinite(u_nom[2]):
            u_nom[2] = 0.0
        u_nom = np.clip(u_nom, u_min, u_max)
        u_ref = u_nom.copy()

        if use_cbf:
            status, sim_next, u_cmd, gamma_val = safety_filter_fixed_delta(
                sim, t, u_ref, params
            )
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


def _pack_mode_output(sim_res: dict, step_log: list[dict]) -> dict:
    t = np.array([float(r["time_s"]) for r in step_log], dtype=float)
    gamma_actual = np.array([float(r["gamma_actual"]) for r in step_log], dtype=float)
    gamma_nom = np.array([float(r["gamma_nom"]) for r in step_log], dtype=float)
    u_nom_z = np.array([float(r["u_nom_z"]) for r in step_log], dtype=float)
    u_star_z = np.array([float(r["u_star_z"]) for r in step_log], dtype=float)
    delta_kappa = np.array([float(r["delta_kappa"]) for r in step_log], dtype=float)
    mismatch = np.array(
        [float(r["delta_h_err_pred_actual"]) for r in step_log], dtype=float
    )
    u_delta_linf = np.array(
        [float(r["u_delta_linf"]) for r in step_log], dtype=float
    )
    trust_region_active = np.array(
        [int(r["trust_region_active"]) for r in step_log], dtype=int
    )
    qp_ok = np.array([int(r["qp_ok"]) for r in step_log], dtype=int)
    return {
        "t": t,
        "gamma_actual": gamma_actual,
        "gamma_nom": gamma_nom,
        "u_nom_z": u_nom_z,
        "u_star_z": u_star_z,
        "delta_kappa": delta_kappa,
        "delta_h_mismatch": mismatch,
        "u_delta_linf": u_delta_linf,
        "trust_region_active": trust_region_active,
        "qp_ok": qp_ok,
        "traj_q": np.asarray(sim_res["traj_q"], dtype=float),
    }


def _print_violation_summary(data: dict, f_max: float) -> None:
    print("\n[violation summary]")
    for mode in ("cbf", "rcbf"):
        g = np.asarray(data[mode].get("gamma_actual", np.zeros(0)), dtype=float)
        n = int(g.size)
        n_viol = int(np.sum(g > float(f_max))) if n > 0 else 0
        label = "CBF" if mode == "cbf" else "rCBF-fixed-delta"
        print(
            f"  {label}: violation={'YES' if n_viol > 0 else 'NO'} "
            f"(n_viol={n_viol}/{n}, max_gamma={float(np.max(g)) if n > 0 else float('nan'):.6f}, "
            f"F_max={float(f_max):.6f})"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Particle CBF/rCBF runner with fixed offline delta_kappa tightening."
    )
    ap.add_argument("--kappa", type=float, default=5e-5)
    ap.add_argument("--fmax", type=float, default=0.25)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--trust-region-radius",
        type=float,
        default=None,
        help="Optional box radius enforcing |u-u_nom|_inf <= radius in each CBF QP.",
    )
    ap.add_argument(
        "--delta-kappa",
        type=float,
        default=None,
        help="If set, use this fixed tightening directly for rCBF.",
    )
    ap.add_argument(
        "--delta-statistic",
        choices=("quantile", "max"),
        default="quantile",
        help="Statistic used to estimate fixed delta_kappa from the CBF rollout.",
    )
    ap.add_argument(
        "--delta-quantile",
        type=float,
        default=0.99,
        help="Quantile used when --delta-statistic=quantile.",
    )
    ap.add_argument(
        "--boundary-h",
        type=float,
        default=0.02,
        help="Boundary-relevant filter: use rows with h_k <= boundary_h or h_next_nom <= boundary_h.",
    )
    ap.add_argument(
        "--delta-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the estimated or manual delta_kappa before running rCBF.",
    )
    ap.add_argument(
        "--out-data",
        type=Path,
        default=Path("paper/cbf_compare/data_particle_ref_cbf_kappa_fixed_delta.npz"),
    )
    ap.add_argument(
        "--out-summary",
        type=Path,
        default=Path("paper/cbf_compare/data_particle_ref_cbf_kappa_fixed_delta_summary.json"),
    )
    args = ap.parse_args()

    cbf_params = _build_base_params(
        kappa=float(args.kappa),
        alpha=float(args.alpha),
        f_max=float(args.fmax),
        seed=int(args.seed),
    )
    cbf_params["trust_region_radius"] = args.trust_region_radius
    cbf_res = run_sim(cbf_params, use_cbf=True)
    cbf_data = _pack_mode_output(cbf_res, cbf_params["_step_log"])

    if args.delta_kappa is None:
        delta_kappa, delta_info = _estimate_delta_kappa(
            cbf_params["_step_log"],
            boundary_h=float(args.boundary_h),
            quantile=float(args.delta_quantile),
            statistic=str(args.delta_statistic),
        )
    else:
        delta_kappa = max(0.0, float(args.delta_kappa))
        delta_info = {
            "boundary_h": float(args.boundary_h),
            "quantile": float(args.delta_quantile),
            "statistic": "manual",
            "num_rows": int(len(cbf_params["_step_log"])),
            "num_boundary_rows": int(0),
            "delta_kappa_hat": float(delta_kappa),
            "max_under_all": float(
                np.max(
                    [
                        max(0.0, float(r["h_next_pred"]) - float(r["h_next_actual"]))
                        for r in cbf_params["_step_log"]
                    ]
                )
            )
            if cbf_params["_step_log"]
            else 0.0,
            "max_under_boundary": 0.0,
        }

    delta_kappa = max(0.0, float(delta_kappa) * float(args.delta_scale))
    delta_info["delta_scale"] = float(args.delta_scale)
    delta_info["delta_kappa_used"] = float(delta_kappa)

    rcbf_params = _build_base_params(
        kappa=float(args.kappa),
        alpha=float(args.alpha),
        f_max=float(args.fmax),
        seed=int(args.seed),
    )
    rcbf_params["trust_region_radius"] = args.trust_region_radius
    rcbf_params["use_robust_cbf"] = True
    rcbf_params["delta_kappa"] = float(delta_kappa)
    rcbf_res = run_sim(rcbf_params, use_cbf=True)
    rcbf_data = _pack_mode_output(rcbf_res, rcbf_params["_step_log"])

    data = {"cbf": cbf_data, "rcbf": rcbf_data}
    _print_violation_summary(data, float(args.fmax))
    print(f"[info] fixed delta_kappa = {float(delta_kappa):.8f}")

    meta = {
        "modes": ["cbf", "rcbf"],
        "kappa": float(args.kappa),
        "fmax": float(args.fmax),
        "alpha": float(args.alpha),
        "seed": int(args.seed),
        "trust_region_radius": (
            float(args.trust_region_radius)
            if args.trust_region_radius is not None
            else None
        ),
        "source": "example/particle_p_cbf_new.py",
        "rcbf_type": "fixed_delta_kappa",
        "delta_info": delta_info,
    }

    payload = {"metadata_json": np.array(json.dumps(meta))}
    for mode in ("cbf", "rcbf"):
        for key, val in data[mode].items():
            payload[f"{mode}/{key}"] = np.asarray(val, dtype=float)

    args.out_data.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_data, **payload)
    args.out_summary.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[write] {args.out_data}")
    print(f"[write] {args.out_summary}")


if __name__ == "__main__":
    main()
