#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from example import planar_push_linear_cimpc_cbf_kappa as base
from example.cbf_trust_region import (
    box_trust_region_bounds,
    box_trust_region_metrics,
)
from src.cpp.pusher_residual import make_pusher_residual
from src.simulator.simulator import Simulator


def estimate_delta_kappa(rows, boundary_h: float, statistic: str, quantile: float):
    under_all = []
    under_boundary = []
    for row in rows:
        under = max(0.0, float(row["h_pred"]) - float(row["h_actual"]))
        under_all.append(under)
        if float(row["h_curr"]) <= boundary_h or float(row["h_nom"]) <= boundary_h:
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


def cbf_filter_fixed_delta(sim, t, u_nom, params):
    u_nom = np.asarray(u_nom, dtype=float).copy()
    u_min = np.asarray(params["u_min"], dtype=float)
    u_max = np.asarray(params["u_max"], dtype=float)
    u_nom = np.clip(u_nom, u_min, u_max)
    status, _, gamma_nom, dgamma_du = base._contact_force_and_jacobians(sim, t, u_nom, int(params["contact_index"]))
    if not status or dgamma_du is None:
        return False, u_nom, {"reason": "diff_solve_fail"}
    f_max = float(params["f_max"])
    alpha = float(params["cbf_alpha"])
    gamma_prev = float(sim.traj.gamma[t - 1][params["contact_index"]]) if t > 0 and len(sim.traj.gamma[t - 1]) > params["contact_index"] else 0.0
    h_curr = float(f_max - gamma_prev)
    h_next_nom = float(f_max - gamma_nom)
    a = -dgamma_du.reshape(1, -1)
    rhs = float((1.0 - alpha) * h_curr)
    if bool(params.get("use_rcbf", False)):
        rhs += max(0.0, float(params.get("delta_kappa", 0.0)))
    rhs_aff = rhs - h_next_nom + float(a.reshape(-1) @ u_nom)
    A = (-a).reshape(1, -1)
    b = np.array([-rhs_aff], dtype=float)
    Q = np.diag(np.asarray(params["qp_weight"], dtype=float))
    H = Q
    g = -Q @ u_nom
    trust_region_radius = params.get("trust_region_radius")
    qp_u_min, qp_u_max = box_trust_region_bounds(
        u_nom, u_min, u_max, trust_region_radius
    )
    u_star, ok, msg = base._solve_qp_slsqp(H, g, A, b, qp_u_min, qp_u_max)
    u_used = u_star if ok else u_nom
    tr_metrics = box_trust_region_metrics(u_used, u_nom, trust_region_radius)
    h_next_pred = float(h_next_nom + float(a.reshape(-1) @ (u_used - u_nom)))
    gamma_pred = float(f_max - h_next_pred)
    return True, u_used, {"reason": "ok" if ok else "qp_fail", "msg": msg, "qp_solved": bool(ok), "gamma_nom": float(gamma_nom), "gamma_pred": float(gamma_pred), "robust_delta": float(max(0.0, params.get("delta_kappa", 0.0) if params.get("use_rcbf", False) else 0.0)), "h_curr": float(h_curr), "h_nom": float(h_next_nom), "h_pred": float(h_next_pred), **tr_metrics}


def run_rollout(u_nom_seq, q0, h, kappa_tol, kappa_accept_ratio, cbf_params=None, *, v0=None):
    """Replay nominal controls; optionally supply a screening initial velocity."""
    model = base.build_pusher(mu_body=0.5)
    model.set_nominal_configuration(np.asarray(q0, dtype=float))
    residuals = make_pusher_residual(model)
    r, rz, rtheta = residuals
    sim = Simulator(model, int(len(u_nom_seq)), h=float(h), diff_sol=True, residual=r, jacobian_z=rz, jacobian_theta=rtheta, policy=None, temp_cmd=np.zeros(2), kappa_tol=float(kappa_tol))
    sim.ip.options.kappa_accept_ratio = float(kappa_accept_ratio)
    sim.ip.options.max_iter = 100
    sim.ip.options.max_ls = 10
    initial_velocity = np.zeros(model.nq) if v0 is None else np.asarray(v0, dtype=float)
    if initial_velocity.shape != (model.nq,):
        raise ValueError(f"Expected v0 shape ({model.nq},), got {initial_velocity.shape}")
    sim.set_state(np.asarray(q0, dtype=float), initial_velocity, 0)
    rows = []
    u_log = []
    gamma_log = []
    gamma_nom_log = []
    gamma_pred_log = []
    robust_delta_log = []
    u_delta_linf_log = []
    trust_region_active_log = []
    qp_solved_log = []
    for t, u_nom in enumerate(u_nom_seq):
        u_cmd = np.asarray(u_nom, dtype=float).copy()
        info = {"gamma_nom": np.nan, "gamma_pred": np.nan, "robust_delta": 0.0, "h_curr": np.nan, "h_nom": np.nan, "h_pred": np.nan}
        if cbf_params is not None:
            ok_cbf, u_cmd, info = cbf_filter_fixed_delta(sim, t, u_cmd, cbf_params)
            if not ok_cbf:
                break
        sim.traj.u[t][:] = u_cmd
        sim.traj.w[t][:] = 0.0
        ok = sim.step(t, diff_sol=True)
        if not ok:
            break
        gamma_t = float(sim.traj.gamma[t][int(cbf_params["contact_index"])]) if cbf_params is not None and len(sim.traj.gamma[t]) > int(cbf_params["contact_index"]) else np.nan
        h_actual = float(cbf_params["f_max"] - gamma_t) if cbf_params is not None and np.isfinite(gamma_t) else np.nan
        rows.append({"t": float(t * h), "h_curr": float(info.get("h_curr", np.nan)), "h_nom": float(info.get("h_nom", np.nan)), "h_pred": float(info.get("h_pred", np.nan)), "h_actual": float(h_actual), "robust_delta": float(info.get("robust_delta", 0.0)), "u_delta_linf": float(info.get("u_delta_linf", 0.0)), "trust_region_active": bool(info.get("trust_region_active", False)), "qp_solved": bool(info.get("qp_solved", False))})
        u_log.append(np.asarray(u_cmd, dtype=float))
        gamma_log.append(float(gamma_t))
        gamma_nom_log.append(float(info.get("gamma_nom", np.nan)))
        gamma_pred_log.append(float(info.get("gamma_pred", np.nan)))
        robust_delta_log.append(float(info.get("robust_delta", 0.0)))
        u_delta_linf_log.append(float(info.get("u_delta_linf", 0.0)))
        trust_region_active_log.append(bool(info.get("trust_region_active", False)))
        qp_solved_log.append(bool(info.get("qp_solved", False)))
    return {"u": np.asarray(u_log, dtype=float), "gamma_safe": np.asarray(gamma_log, dtype=float), "gamma_nom": np.asarray(gamma_nom_log, dtype=float), "gamma_pred": np.asarray(gamma_pred_log, dtype=float), "robust_delta": np.asarray(robust_delta_log, dtype=float), "u_delta_linf": np.asarray(u_delta_linf_log, dtype=float), "trust_region_active": np.asarray(trust_region_active_log, dtype=bool), "qp_solved": np.asarray(qp_solved_log, dtype=bool), "rows": rows}


def main():
    ap = argparse.ArgumentParser(description="Planar push CBF/rCBF runner with fixed offline delta_kappa tightening from a saved u_nom rollout.")
    ap.add_argument("--in-json", type=Path, default=Path("reference_trajectory/planar_push_linear_cimpc_u_seq.json"))
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--delta-kappa", type=float, default=None)
    ap.add_argument("--delta-statistic", choices=("quantile", "max"), default="quantile")
    ap.add_argument("--delta-quantile", type=float, default=0.99)
    ap.add_argument("--boundary-h", type=float, default=0.03)
    ap.add_argument("--delta-scale", type=float, default=1.0)
    ap.add_argument("--trust-region-radius", type=float, default=None, help="Optional box radius enforcing |u-u_nom|_inf <= radius in each CBF QP.")
    ap.add_argument("--out-json", type=Path, default=Path("reference_trajectory/planar_push_linear_cimpc_u_seq_fixed_delta.json"))
    args = ap.parse_args()

    payload = json.loads(args.in_json.read_text())
    h = float(payload["h"])
    q0 = np.asarray(payload["q0"], dtype=float)
    u_nom = np.asarray(payload["u_nom"], dtype=float)
    if u_nom.ndim != 2 or u_nom.shape[1] != 2:
        raise ValueError(f"Expected u_nom shape (N,2), got {u_nom.shape}")
    f_max = float(payload.get("f_max", base.CFG["f_max"]))
    kappa = float(base.CFG["kappa"] if args.kappa is None else args.kappa)
    kappa_accept_ratio = float(base.CFG["kappa_accept_ratio"])
    cbf_params = {
        "f_max": float(f_max),
        "cbf_alpha": float(base.CFG["cbf_alpha"] if args.alpha is None else args.alpha),
        "use_rcbf": False,
        "delta_kappa": 0.0,
        "contact_index": int(base.CFG["cbf_contact_index"]),
        "qp_weight": np.array([float(base.CFG["cbf_qx"]), float(base.CFG["cbf_qy"])], dtype=float),
        "u_min": np.array([float(base.CFG["cbf_u_min_x"]), float(base.CFG["cbf_u_min_y"])], dtype=float),
        "u_max": np.array([float(base.CFG["cbf_u_max_x"]), float(base.CFG["cbf_u_max_y"])], dtype=float),
        "trust_region_radius": args.trust_region_radius,
    }
    cbf_run = run_rollout(u_nom, q0, h, kappa, kappa_accept_ratio, cbf_params=cbf_params)
    if args.delta_kappa is None:
        delta_kappa, delta_info = estimate_delta_kappa(cbf_run["rows"], boundary_h=float(args.boundary_h), statistic=str(args.delta_statistic), quantile=float(args.delta_quantile))
    else:
        delta_kappa = max(0.0, float(args.delta_kappa))
        delta_info = {"boundary_h": float(args.boundary_h), "statistic": "manual", "quantile": float(args.delta_quantile), "num_rows": int(len(cbf_run["rows"])), "num_boundary_rows": 0, "delta_kappa_hat": float(delta_kappa)}
    delta_kappa = max(0.0, float(delta_kappa) * float(args.delta_scale))
    delta_info["delta_scale"] = float(args.delta_scale)
    delta_info["delta_kappa_used"] = float(delta_kappa)
    rcbf_params = dict(cbf_params)
    rcbf_params["use_rcbf"] = True
    rcbf_params["delta_kappa"] = float(delta_kappa)
    rcbf_run = run_rollout(u_nom, q0, h, kappa, kappa_accept_ratio, cbf_params=rcbf_params)
    out = {
        "h": float(h),
        "q0": q0.tolist(),
        "u_nom": u_nom.tolist(),
        "cbf_u": cbf_run["u"].tolist(),
        "cbf_gamma_nom": cbf_run["gamma_nom"].tolist(),
        "cbf_gamma_pred": cbf_run["gamma_pred"].tolist(),
        "cbf_gamma_safe": cbf_run["gamma_safe"].tolist(),
        "cbf_u_delta_linf": cbf_run["u_delta_linf"].tolist(),
        "cbf_trust_region_active": cbf_run["trust_region_active"].tolist(),
        "cbf_qp_solved": cbf_run["qp_solved"].tolist(),
        "rcbf_u": rcbf_run["u"].tolist(),
        "rcbf_gamma_nom": rcbf_run["gamma_nom"].tolist(),
        "rcbf_gamma_pred": rcbf_run["gamma_pred"].tolist(),
        "rcbf_gamma_safe": rcbf_run["gamma_safe"].tolist(),
        "rcbf_u_delta_linf": rcbf_run["u_delta_linf"].tolist(),
        "rcbf_trust_region_active": rcbf_run["trust_region_active"].tolist(),
        "rcbf_qp_solved": rcbf_run["qp_solved"].tolist(),
        "rcbf_delta_kappa": float(delta_kappa),
        "rcbf_delta_seq": rcbf_run["robust_delta"].tolist(),
        "f_max": float(f_max),
        "kappa": float(kappa),
        "alpha": float(cbf_params["cbf_alpha"]),
        "trust_region_radius": (
            float(args.trust_region_radius)
            if args.trust_region_radius is not None
            else None
        ),
        "source": "example/planar_push_linear_cimpc_cbf_fixed_delta.py",
        "delta_info": delta_info,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[info] fixed delta_kappa = {float(delta_kappa):.8f}")
    print(f"[write] {args.out_json}")


if __name__ == "__main__":
    main()
