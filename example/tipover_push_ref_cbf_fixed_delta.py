#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from example import tipover_push_ref_cbf_kappa as base
from example.cbf_trust_region import (
    box_trust_region_bounds,
    box_trust_region_metrics,
)


def estimate_delta_kappa(rows, boundary_h: float, statistic: str, quantile: float):
    under_all = []
    under_boundary = []
    for row in rows:
        h_pred = float(row["h_pred"])
        h_actual = float(row["h_actual"])
        under = max(0.0, h_pred - h_actual)
        under_all.append(under)
        h_curr = float(row["h_curr"])
        h_nom = float(row["h_nom"])
        if h_curr <= boundary_h or h_nom <= boundary_h:
            under_boundary.append(under)

    sample = under_boundary if under_boundary else under_all
    if sample:
        if statistic == "max":
            delta = float(np.max(sample))
        else:
            delta = float(np.quantile(np.asarray(sample, dtype=float), quantile))
    else:
        delta = 0.0
    return delta, {
        "boundary_h": float(boundary_h),
        "statistic": statistic,
        "quantile": float(quantile),
        "num_rows": int(len(rows)),
        "num_boundary_rows": int(len(under_boundary)),
        "delta_kappa_hat": float(delta),
    }


def safety_filter_fixed_delta(sim, t, u_nom, params):
    u_nom = np.array(u_nom, dtype=float).copy()
    u_min = np.array(params["u_min"], dtype=float)
    u_max = np.array(params["u_max"], dtype=float)
    u_nom = np.clip(u_nom, u_min, u_max)

    ok_pred, sim_nom, gamma_nom, dgamma_du, theta_next_nom, dtheta_du = base._predict_terms(
        sim,
        t,
        u_nom,
        n_floor_contacts=params["n_floor_contacts"],
        theta_idx=params["theta_idx"],
    )
    if not ok_pred:
        return False, sim_nom, u_nom, None
    del theta_next_nom, dtheta_du

    n_floor = gamma_nom.shape[0]
    corner_idx = int(params.get("constrained_corner_idx", 0))
    if corner_idx < 0 or corner_idx >= n_floor:
        corner_idx = 0

    h_nom = float(params["F_max"] - gamma_nom[corner_idx])
    gamma_prev = (
        base._floor_corner_gamma(
            sim.traj.gamma[t - 1], n_floor_contacts=params["n_floor_contacts"]
        )
        if t > 0
        else np.zeros(n_floor, dtype=float)
    )
    h_curr = float(params["F_max"] - gamma_prev[corner_idx])

    a_cbf = -dgamma_du[corner_idx, :].reshape(1, -1)
    rhs = float((1.0 - params["cbf_alpha"]) * h_curr)
    if params.get("use_robust_cbf", False):
        rhs += max(0.0, float(params.get("delta_kappa", 0.0)))
    rhs_aff = rhs - h_nom + float(a_cbf.reshape(-1) @ u_nom)
    A = -a_cbf
    b = np.array([-rhs_aff], dtype=float)

    Q = np.array(params["qp_Q"], dtype=float)
    if Q.ndim == 1:
        Q = np.diag(Q)
    H, g = base._build_qp_objective(u_nom=u_nom, Q=Q)
    trust_region_radius = params.get("trust_region_radius")
    qp_u_min, qp_u_max = box_trust_region_bounds(
        u_nom, u_min, u_max, trust_region_radius
    )
    u_safe, ok_qp, qp_info = base._solve_qp_casadi(
        H, g, A, b, qp_u_min, qp_u_max
    )
    if not ok_qp:
        u_safe = u_nom
    tr_metrics = box_trust_region_metrics(u_safe, u_nom, trust_region_radius)
    cbf_residual = float((A @ u_safe - b).reshape(-1)[0])

    status, sim = base.step_with_u(sim, t, u_safe, diff_sol=False)
    if not status:
        return False, sim, u_safe, None

    gamma_safe = base._floor_corner_gamma(
        sim.traj.gamma[t], n_floor_contacts=params["n_floor_contacts"]
    )
    gamma_safe_corner = float(gamma_safe[corner_idx]) if gamma_safe.size else 0.0
    h_actual = float(params["F_max"] - gamma_safe_corner)
    h_pred = float(h_nom + float(a_cbf.reshape(-1) @ (u_safe - u_nom)))
    return (
        True,
        sim,
        u_safe,
        {
            "gamma_nom": gamma_nom.copy(),
            "gamma_safe": gamma_safe.copy(),
            "h_curr": float(h_curr),
            "h_nom": float(h_nom),
            "h_pred": float(h_pred),
            "h_actual": float(h_actual),
            "u_delta_norm": float(np.linalg.norm(u_safe - u_nom)),
            **tr_metrics,
            "qp_solved": bool(ok_qp),
            "cbf_residual": float(cbf_residual),
            "robust_delta": float(max(0.0, params.get("delta_kappa", 0.0) if params.get("use_robust_cbf", False) else 0.0)),
            "qp_info": qp_info,
        },
    )


def run_mode(params, mode):
    model = params["model"]
    residuals = params["residuals"]
    dt = params["dt"]
    num_steps = params["num_steps"]
    kappa_tol = params["kappa_tol"]
    u_ref = params["u_ref"]
    q0 = params["q0"]
    v0 = params["v0"]

    sim = base.build_sim(model, residuals, dt, num_steps, kappa_tol, diff_sol=False)
    sim.traj.reset()
    sim.grad.reset()
    sim.prev_z = None
    sim.set_state(q0, v0, 0)

    sim_diff = base.build_sim(model, residuals, dt, num_steps, kappa_tol, diff_sol=True)
    sim_diff.traj.reset()
    sim_diff.grad.reset()
    sim_diff.prev_z = None
    sim._diff_sim = sim_diff

    cidx = int(params.get("constrained_corner_idx", 0))
    ctrl_params = params.copy()
    ctrl_params["use_robust_cbf"] = mode == "robust_cbf"

    rows = []
    t_log = []
    theta_log = []
    gamma_corner0_log = []
    h_pred_log = []
    h_actual_log = []
    u_delta_norm_log = []
    u_delta_linf_log = []
    trust_region_active_log = []
    qp_solved_log = []
    cbf_residual_log = []
    violation_log = []
    u_nom_log = []
    u_cmd_log = []
    q_log = []
    status = True

    for t in range(num_steps):
        u_nom = np.array(u_ref[t], dtype=float)
        u_nom_log.append(u_nom.copy())
        if mode == "nominal":
            sim.traj.u[t][:] = u_nom
            sim.traj.w[t][:] = 0.0
            status = sim.step(t, diff_sol=False)
            u_cmd = u_nom.copy()
            info = {
                "h_curr": np.nan,
                "h_nom": np.nan,
                "h_pred": np.nan,
                "h_actual": np.nan,
                "u_delta_norm": 0.0,
                "u_delta_linf": 0.0,
                "trust_region_active": False,
                "qp_solved": True,
                "cbf_residual": np.nan,
                "robust_delta": 0.0,
            }
        else:
            status, sim_next, u_cmd, info = safety_filter_fixed_delta(sim, t, u_nom, ctrl_params)
            sim = sim_next
            if info is None:
                info = {}
        if not status:
            break

        gamma_vec = base._floor_corner_gamma(
            sim.traj.gamma[t], n_floor_contacts=params["n_floor_contacts"]
        )
        gamma0 = float(gamma_vec[cidx]) if gamma_vec.size else 0.0
        theta_now = float(sim.traj.q[t + 1][params["theta_idx"]])
        violation = int(gamma0 > float(params["F_max"]))

        t_log.append(float(t * dt))
        theta_log.append(theta_now)
        gamma_corner0_log.append(gamma0)
        h_pred_log.append(float(info.get("h_pred", np.nan)))
        h_actual_log.append(float(info.get("h_actual", float(params["F_max"] - gamma0))))
        u_delta_norm_log.append(float(info.get("u_delta_norm", 0.0)))
        u_delta_linf_log.append(float(info.get("u_delta_linf", 0.0)))
        trust_region_active_log.append(bool(info.get("trust_region_active", False)))
        qp_solved_log.append(bool(info.get("qp_solved", False)))
        cbf_residual_log.append(float(info.get("cbf_residual", np.nan)))
        violation_log.append(violation)
        u_cmd_log.append(np.array(u_cmd, dtype=float))
        q_log.append(np.array(sim.traj.q[t + 1], dtype=float))
        rows.append(
            {
                "t": float(t * dt),
                "h_curr": float(info.get("h_curr", np.nan)),
                "h_nom": float(info.get("h_nom", np.nan)),
                "h_pred": float(info.get("h_pred", np.nan)),
                "h_actual": float(info.get("h_actual", np.nan)),
                "robust_delta": float(info.get("robust_delta", 0.0)),
                "u_delta_linf": float(info.get("u_delta_linf", 0.0)),
                "trust_region_active": bool(info.get("trust_region_active", False)),
                "qp_solved": bool(info.get("qp_solved", False)),
            }
        )

    t_arr = np.array(t_log, dtype=float)
    theta_arr = np.array(theta_log, dtype=float)
    gamma_arr = np.array(gamma_corner0_log, dtype=float)
    violation_arr = np.array(violation_log, dtype=int)
    theta_target = float(params["theta_target"])
    success_tipover = bool(theta_arr.size > 0 and float(np.max(theta_arr)) >= theta_target)
    success_safe = bool(np.sum(violation_arr) == 0)
    return {
        "mode": mode,
        "status": bool(status),
        "t": t_arr,
        "theta": theta_arr,
        "gamma_corner0": gamma_arr,
        "h_pred": np.array(h_pred_log, dtype=float),
        "h_actual": np.array(h_actual_log, dtype=float),
        "u_delta_norm": np.array(u_delta_norm_log, dtype=float),
        "u_delta_linf": np.array(u_delta_linf_log, dtype=float),
        "trust_region_active": np.array(trust_region_active_log, dtype=np.int64),
        "qp_solved": np.array(qp_solved_log, dtype=np.int64),
        "violation": violation_arr,
        "violation_cumsum": np.cumsum(violation_arr),
        "u_nom": np.array(u_nom_log[: len(t_arr)], dtype=float),
        "u_cmd": np.array(u_cmd_log[: len(t_arr)], dtype=float),
        "traj_q": np.array(q_log, dtype=float),
        "cbf_residual": np.array(cbf_residual_log[: len(t_arr)], dtype=float),
        "max_force": float(np.max(gamma_arr)) if gamma_arr.size else 0.0,
        "n_viol": int(np.sum(violation_arr)),
        "mean_qp_solve_time": 0.0,
        "success_tipover": success_tipover,
        "success_safe": success_safe,
        "success": bool(success_tipover and success_safe),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser(description="Tipover CBF/rCBF runner with fixed offline delta_kappa tightening.")
    ap.add_argument("--ref", type=Path, default=Path("reference_trajectory/tipover_ref_traj.json"))
    ap.add_argument("--f-max", type=float, default=0.9)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--kappa", type=float, default=5e-4)
    ap.add_argument("--theta-target", type=float, default=None)
    ap.add_argument("--use-cpp-jac", action="store_true", default=True, help="Compatibility flag; C++ Jacobians are always enabled.")
    ap.add_argument("--delta-kappa", type=float, default=None)
    ap.add_argument("--delta-statistic", choices=("quantile", "max"), default="quantile")
    ap.add_argument("--delta-quantile", type=float, default=0.99)
    ap.add_argument("--boundary-h", type=float, default=0.05)
    ap.add_argument("--delta-scale", type=float, default=1.0)
    ap.add_argument("--trust-region-radius", type=float, default=None, help="Optional box radius enforcing |u-u_nom|_inf <= radius in each CBF QP.")
    ap.add_argument("--out-data", type=Path, default=Path("paper/cbf_compare/data_tipover_ref_cbf_fixed_delta.npz"))
    ap.add_argument("--out-summary", type=Path, default=Path("paper/cbf_compare/data_tipover_ref_cbf_fixed_delta_summary.json"))
    args = ap.parse_args()

    ref = base.load_ref_traj(args.ref)
    q_ref = np.asarray(ref["q"], dtype=float)
    u_ref = np.asarray(ref["u"], dtype=float)
    h = float(ref["h"])
    q0 = q_ref[0].copy()
    v0 = (q_ref[1] - q_ref[0]) / h if q_ref.shape[0] >= 2 else np.zeros(7)
    model = base.build_tipover_model(q0, ref)
    model.set_nominal_configuration(q0)
    residuals = base.make_tipover_push_residual(model, use_cpp_jac=args.use_cpp_jac)
    theta_ref = q_ref[: int(u_ref.shape[0]), 3].copy()
    theta_target = float(args.theta_target) if args.theta_target is not None else float(theta_ref[-1])

    base_params = {
        "model": model,
        "residuals": residuals,
        "dt": h,
        "num_steps": int(u_ref.shape[0]),
        "kappa_tol": float(args.kappa),
        "u_ref": u_ref,
        "theta_ref": theta_ref,
        "q0": q0,
        "v0": v0,
        "u_min": np.array([-6.0, -6.0], dtype=float),
        "u_max": np.array([6.0, 6.0], dtype=float),
        "F_max": float(args.f_max),
        "cbf_alpha": float(args.alpha),
        "qp_Q": np.array([1.0, 1.0], dtype=float),
        "n_floor_contacts": 4,
        "constrained_corner_idx": 0,
        "theta_idx": 3,
        "theta_target": theta_target,
        "use_robust_cbf": False,
        "delta_kappa": 0.0,
        "trust_region_radius": args.trust_region_radius,
    }

    res_nominal = run_mode(base_params, "nominal")
    res_cbf = run_mode(base_params, "cbf")
    if args.delta_kappa is None:
        delta_kappa, delta_info = estimate_delta_kappa(
            res_cbf["rows"],
            boundary_h=float(args.boundary_h),
            statistic=str(args.delta_statistic),
            quantile=float(args.delta_quantile),
        )
    else:
        delta_kappa = max(0.0, float(args.delta_kappa))
        delta_info = {
            "boundary_h": float(args.boundary_h),
            "statistic": "manual",
            "quantile": float(args.delta_quantile),
            "num_rows": int(len(res_cbf["rows"])),
            "num_boundary_rows": 0,
            "delta_kappa_hat": float(delta_kappa),
        }
    delta_kappa = max(0.0, float(delta_kappa) * float(args.delta_scale))
    delta_info["delta_scale"] = float(args.delta_scale)
    delta_info["delta_kappa_used"] = float(delta_kappa)
    robust_params = base_params.copy()
    robust_params["use_robust_cbf"] = True
    robust_params["delta_kappa"] = float(delta_kappa)
    res_rcbf = run_mode(robust_params, "robust_cbf")
    results = [res_nominal, res_cbf, res_rcbf]

    meta = {
        "f_max": float(args.f_max),
        "theta_target": theta_target,
        "modes": [res["mode"] for res in results],
        "source": "example/tipover_push_ref_cbf_fixed_delta.py",
        "rcbf_type": "fixed_delta_kappa",
        "trust_region_radius": (
            float(args.trust_region_radius)
            if args.trust_region_radius is not None
            else None
        ),
        "delta_info": delta_info,
    }
    arrays = {"metadata_json": np.array(json.dumps(meta))}
    for res in results:
        mode = str(res["mode"])
        for key, value in res.items():
            if key == "rows":
                continue
            if isinstance(value, np.ndarray):
                arrays[f"{mode}/{key}"] = value
            elif isinstance(value, (bool, np.bool_)):
                arrays[f"{mode}/{key}"] = np.array(int(value), dtype=np.int64)
            elif isinstance(value, (int, np.integer)):
                arrays[f"{mode}/{key}"] = np.array(int(value), dtype=np.int64)
            elif isinstance(value, (float, np.floating)):
                arrays[f"{mode}/{key}"] = np.array(float(value), dtype=np.float64)
    args.out_data.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_data, **arrays)
    args.out_summary.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[info] fixed delta_kappa = {float(delta_kappa):.8f}")
    base.print_summary_table(results)
    base.print_violation_status(results)
    print(f"saved rollout data: {args.out_data}")
    print(f"[write] {args.out_summary}")


if __name__ == "__main__":
    main()
