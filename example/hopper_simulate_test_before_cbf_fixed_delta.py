#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from example import hopper_simulate_test_before_cbf as base
from example.cbf_trust_region import (
    box_trust_region_bounds,
    box_trust_region_metrics,
)


def estimate_delta_kappa(cbf_log, boundary_h: float, statistic: str, quantile: float):
    under_all = []
    under_boundary = []
    for item in cbf_log:
        h_pred = float(item.get("h_next_pred", 0.0))
        h_actual = float(item.get("h_next_actual", 0.0))
        under = max(0.0, h_pred - h_actual)
        under_all.append(under)
        h_curr = float(item.get("h_curr", np.nan))
        h_nom = float(item.get("h_nom", np.nan))
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
        "num_rows": int(len(cbf_log)),
        "num_boundary_rows": int(len(under_boundary)),
        "delta_kappa_hat": float(delta),
    }


def cbf_filter_fixed_delta(
    sim,
    t,
    u_nom,
    delta_kappa: float = 0.0,
    use_robust_cbf: bool = False,
    alpha: float | None = None,
    trust_region_radius: float | None = None,
):
    u_nom = np.asarray(u_nom, dtype=float).copy()
    u_min = np.array([-100.0, -100.0], dtype=float)
    u_max = np.array([100.0, 100.0], dtype=float)
    u_nom = np.clip(u_nom, u_min, u_max)

    ok_pred, sim_diff, gamma_nom, dgamma_du = base._predict_gamma_linearization(sim, t, u_nom)
    sim._cbf_diff_sim = sim_diff
    if not ok_pred:
        return u_nom, {"qp_solved": False, "reason": "linearization_failed"}

    gamma_prev = float(np.asarray(sim.traj.gamma[t - 1], dtype=float).reshape(-1)[0]) if t > 0 else 0.0
    h_curr = float(base.F_MAX - gamma_prev)
    h_nom = float(base.F_MAX - gamma_nom)
    cbf_alpha = float(base.CBF_ALPHA if alpha is None else alpha)
    rhs = float((1.0 - cbf_alpha) * h_curr)
    robust_delta = float(max(0.0, delta_kappa)) if bool(use_robust_cbf) else 0.0
    rhs_robust = float(rhs + robust_delta)

    A = dgamma_du.reshape(1, -1)
    b = np.array([h_nom + float(dgamma_du @ u_nom) - rhs_robust], dtype=float)
    H = np.diag(np.asarray(base.CBF_Q, dtype=float).reshape(-1))
    g = -H @ u_nom
    qp_u_min, qp_u_max = box_trust_region_bounds(
        u_nom, u_min, u_max, trust_region_radius
    )
    u_safe, ok_qp, qp_msg = base._solve_qp(
        H, g, A, b, qp_u_min, qp_u_max
    )
    if not ok_qp:
        u_safe = u_nom.copy()
    tr_metrics = box_trust_region_metrics(u_safe, u_nom, trust_region_radius)

    h_next_pred = float(h_nom - float(dgamma_du @ (u_safe - u_nom)))
    info = {
        "qp_solved": bool(ok_qp),
        "qp_msg": qp_msg,
        "gamma_nom": float(gamma_nom),
        "h_curr": float(h_curr),
        "h_nom": float(h_nom),
        "h_next_pred": float(h_next_pred),
        "h_next_actual": None,
        "robust_delta": float(robust_delta),
        "cbf_residual": float((A @ u_safe - b).reshape(-1)[0]),
        "u_nom": u_nom.tolist(),
        "u_safe": u_safe.tolist(),
        **tr_metrics,
    }
    return u_safe, info


def simulate_from_trajectory_fixed_delta(
    model,
    residual,
    jacobian_z,
    jacobian_theta,
    traj_path: Path,
    tag: str,
    enable_cbf: bool = True,
    use_robust_cbf: bool = False,
    delta_kappa: float = 0.0,
    alpha: float | None = None,
    kappa: float | None = None,
    trust_region_radius: float | None = None,
):
    u_seq, h, q_ref, raw = base.load_u_from_json(traj_path, model.nu)
    T = int(len(u_seq))
    sim = base.Simulator(
        model=model,
        T=T,
        h=h,
        diff_sol=True,
        residual=residual,
        jacobian_z=jacobian_z,
        jacobian_theta=jacobian_theta,
        kappa_tol=float(base.KAPPA_TOL if kappa is None else kappa),
    )
    sim.ip.options.verbose = False
    q0 = model.nominal_configuration().astype(float)
    q0[:] = base.Q1
    v0 = base.V1.copy()
    sim.traj.reset()
    sim.grad.reset()
    sim.prev_z = None
    sim.set_state(q0, v0, 0)

    solved_steps = 0
    solver_status = True
    fall = False
    cbf_qp_fail_count = 0
    cbf_log = []
    for t in range(T):
        u_nom = u_seq[t].copy()
        q_now = sim.traj.q[t + 1]
        q_prev = sim.traj.q[t]
        v_now = (q_now - q_prev) / h
        in_contact = float(model.signed_distance(sim.traj.q[t + 1])[0]) <= float(base.CONTACT_THRESH)
        if base.USE_Q_TRACKING and q_ref is not None and t + 1 < len(q_ref):
            q_ref_now = q_ref[t + 1]
            q_ref_prev = q_ref[t]
            v_ref = (q_ref_now - q_ref_prev) / h
            du0 = float(base.KP_THETA) * (float(q_ref_now[2]) - float(q_now[2])) + float(base.KD_THETA) * (float(v_ref[2]) - float(v_now[2]))
            du1 = float(base.KP_R) * (float(q_ref_now[3]) - float(q_now[3])) + float(base.KD_R) * (float(v_ref[3]) - float(v_now[3]))
            du0 = float(np.clip(du0, -float(base.DU0_TRACK_LIMIT), float(base.DU0_TRACK_LIMIT)))
            du1 = float(np.clip(du1, -float(base.DU1_TRACK_LIMIT), float(base.DU1_TRACK_LIMIT)))
            u_nom[0] = float(u_seq[t][0]) + du0
            u_nom[1] = float(u_seq[t][1]) + du1
        if in_contact and u_nom[1] > 0.0:
            u_nom[1] *= float(base.U2_CONTACT_SCALE)
        if bool(base.THETA_FLIGHT_BIAS_ENABLE) and (not in_contact):
            u_nom[0] += float(base.THETA_FLIGHT_BIAS)
        u_nom[0] = float(np.clip(u_nom[0], -100.0, 100.0))
        u_nom[1] = float(np.clip(u_nom[1], -100.0, 100.0))

        if bool(enable_cbf and base.USE_CBF):
            u_cmd, cbf_info = cbf_filter_fixed_delta(
                sim,
                t,
                u_nom,
                delta_kappa=delta_kappa,
                use_robust_cbf=bool(use_robust_cbf),
                alpha=alpha,
                trust_region_radius=trust_region_radius,
            )
            if not bool(cbf_info.get("qp_solved", False)):
                cbf_qp_fail_count += 1
            cbf_log.append(cbf_info)
        else:
            u_cmd = u_nom.copy()
            cbf_log.append({"qp_solved": False, "reason": "disabled"})

        sim.traj.u[t][:] = u_cmd
        sim.traj.w[t][:] = np.zeros(model.nw, dtype=float)
        solver_status = sim.step(t, sim.diff_sol)
        if not solver_status:
            break
        if bool(enable_cbf and base.USE_CBF):
            gamma_actual = float(
                np.asarray(sim.traj.gamma[t], dtype=float).reshape(-1)[0]
            )
            cbf_info["h_next_actual"] = float(base.F_MAX - gamma_actual)
        solved_steps += 1
        if float(sim.traj.q[t + 2][1]) <= 0.0:
            fall = True
            break

    status = bool(solver_status and (not fall))
    q_sim = np.array([sim.traj.q[i + 1] for i in range(solved_steps + 1)], dtype=float)
    u_sim = np.array([sim.traj.u[i] for i in range(solved_steps)], dtype=float)
    u_raw = np.asarray(u_seq, dtype=float)
    gamma_sim = np.array([sim.traj.gamma[i] for i in range(solved_steps)], dtype=float)
    b_sim = np.array([sim.traj.b[i] for i in range(solved_steps)], dtype=float)
    contact_tiles_x = base.extract_contact_tiles_from_rollout(model, q_sim, phi_thresh=5e-5)
    out_dir = base.OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"hopper_model_linear_sim_{tag}.json"
    payload = {
        "status": bool(status),
        "status_text": "Pass" if bool(status) else "Fail",
        "solver_status": bool(solver_status),
        "fall": bool(fall),
        "solved_steps": int(solved_steps),
        "T": int(T),
        "h": float(h),
        "input_json": str(traj_path),
        "use_cbf": bool(enable_cbf and base.USE_CBF),
        "use_robust_cbf": bool(use_robust_cbf),
        "delta_kappa": float(delta_kappa),
        "cbf_alpha": float(base.CBF_ALPHA if alpha is None else alpha),
        "f_max": float(base.F_MAX),
        "kappa_tol": float(base.KAPPA_TOL if kappa is None else kappa),
        "trust_region_radius": (
            float(trust_region_radius)
            if trust_region_radius is not None
            else None
        ),
        "q": q_sim.tolist(),
        "u": u_sim.tolist(),
        "gamma": gamma_sim.tolist(),
        "b": b_sim.tolist(),
        "contact_tiles_x": contact_tiles_x.tolist(),
        "cbf_log": cbf_log,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    out_rollout_json = out_dir / f"hopper_rollout_q_u_contact_{tag}.json"
    rollout_payload = {"h": float(h), "t_state": (np.arange(q_sim.shape[0], dtype=float) * h).tolist(), "t_control": (np.arange(u_sim.shape[0], dtype=float) * h).tolist(), "q": q_sim.tolist(), "u": u_sim.tolist(), "gamma": gamma_sim.tolist(), "b": b_sim.tolist(), "contact_tiles_x": contact_tiles_x.tolist()}
    out_rollout_json.write_text(json.dumps(rollout_payload, indent=2))
    print(f"[write] {out_json}")
    print(f"[write] {out_rollout_json}")
    return {"status": bool(status), "solver_status": bool(solver_status), "fall": bool(fall), "solved_steps": int(solved_steps), "T": int(T), "h": float(h), "q": q_sim, "u": u_sim, "u_ref": u_raw[:, : model.nu].copy(), "gamma": gamma_sim, "b": b_sim, "contact_tiles_x": contact_tiles_x, "cbf_log": cbf_log}


def main():
    ap = argparse.ArgumentParser(description="Hopper CBF/rCBF runner with fixed offline delta_kappa tightening.")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--delta-kappa", type=float, default=None)
    ap.add_argument("--delta-statistic", choices=("quantile", "max"), default="quantile")
    ap.add_argument("--delta-quantile", type=float, default=0.99)
    ap.add_argument("--boundary-h", type=float, default=0.05)
    ap.add_argument("--delta-scale", type=float, default=1.0)
    ap.add_argument("--trust-region-radius", type=float, default=None, help="Optional box radius enforcing |u-u_nom|_inf <= radius in each CBF QP.")
    ap.add_argument("--out-dir", type=Path, default=base.OUT_DIR)
    args = ap.parse_args()
    base.OUT_DIR = args.out_dir

    model = base.Hopper(nq=4, nu=2, nw=0, nc=1, mb=1.0, ml=0.1, jb=0.25, jl=0.025, mu=1.0, gravity=9.81, r_min=0.1, r_max=0.5)
    residual, jacobian_z, jacobian_theta = base.make_hopper_residual(model)
    result_after = simulate_from_trajectory_fixed_delta(model, residual, jacobian_z, jacobian_theta, base.TRAJ_PATH_AFTER, "after_cbf_fixed", enable_cbf=True, use_robust_cbf=False, delta_kappa=0.0, alpha=args.alpha, kappa=args.kappa, trust_region_radius=args.trust_region_radius)
    result_after_nominal = simulate_from_trajectory_fixed_delta(model, residual, jacobian_z, jacobian_theta, base.TRAJ_PATH_AFTER, "after_nominal_fixed", enable_cbf=False, use_robust_cbf=False, delta_kappa=0.0, alpha=args.alpha, kappa=args.kappa)
    if args.delta_kappa is None:
        delta_kappa, delta_info = estimate_delta_kappa(result_after["cbf_log"], boundary_h=float(args.boundary_h), statistic=str(args.delta_statistic), quantile=float(args.delta_quantile))
    else:
        delta_kappa = max(0.0, float(args.delta_kappa))
        delta_info = {"boundary_h": float(args.boundary_h), "statistic": "manual", "quantile": float(args.delta_quantile), "num_rows": int(len(result_after["cbf_log"])), "num_boundary_rows": 0, "delta_kappa_hat": float(delta_kappa)}
    delta_kappa = max(0.0, float(delta_kappa) * float(args.delta_scale))
    delta_info["delta_scale"] = float(args.delta_scale)
    delta_info["delta_kappa_used"] = float(delta_kappa)
    result_after_rcbf = simulate_from_trajectory_fixed_delta(model, residual, jacobian_z, jacobian_theta, base.TRAJ_PATH_AFTER, "after_rcbf_fixed", enable_cbf=True, use_robust_cbf=True, delta_kappa=float(delta_kappa), alpha=args.alpha, kappa=args.kappa, trust_region_radius=args.trust_region_radius)
    gamma_after = np.asarray(result_after["gamma"], dtype=float).reshape(-1)
    gamma_after_rcbf = np.asarray(result_after_rcbf["gamma"], dtype=float).reshape(-1)
    print(f"[info] max contact force (after_cbf_fixed): {float(np.max(gamma_after)) if gamma_after.size > 0 else 0.0:.6f}")
    print(f"[info] max contact force (after_rcbf_fixed): {float(np.max(gamma_after_rcbf)) if gamma_after_rcbf.size > 0 else 0.0:.6f}")
    summary_path = base.OUT_DIR / "hopper_fixed_delta_summary.json"
    summary_path.write_text(json.dumps({"source": "example/hopper_simulate_test_before_cbf_fixed_delta.py", "rcbf_type": "fixed_delta_kappa", "alpha": float(base.CBF_ALPHA if args.alpha is None else args.alpha), "kappa": float(base.KAPPA_TOL if args.kappa is None else args.kappa), "trust_region_radius": float(args.trust_region_radius) if args.trust_region_radius is not None else None, "delta_info": delta_info}, indent=2), encoding="utf-8")
    print(f"[info] fixed delta_kappa = {float(delta_kappa):.8f}")
    print(f"[write] {summary_path}")


if __name__ == "__main__":
    main()
