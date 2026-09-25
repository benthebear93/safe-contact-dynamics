import json
import copy
import os
from pathlib import Path

import numpy as np
from example.cbf_qp import minimize_qp

from src.cpp.hopper_residual import make_hopper_residual
from src.robots.hopper.model_linear import Hopper
from src.simulator.simulator import Simulator

ROOT = Path(__file__).resolve().parents[1]
TRAJ_PATH_BEFORE = ROOT / "reference_trajectory" / "hopper_trajectory_before.json"
TRAJ_PATH_AFTER = ROOT / "reference_trajectory" / "hopper_trajectory_after.json"
OUT_DIR = ROOT / "example" / "output"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None else float(default)

H_DEFAULT = 0.01
KAPPA_TOL = 5e-5
Q1 = np.array([0.0, 0.5, 0.0, 0.5], dtype=float)
V1 = np.zeros(4, dtype=float)
USE_Q_TRACKING = True
KP_THETA = 1.8
KD_THETA = 0.25
KP_R = 1.0
KD_R = 0.1
DU0_TRACK_LIMIT = 2.0
DU1_TRACK_LIMIT = 2.0
U2_CONTACT_SCALE = 1.0
CONTACT_THRESH = 1e-5
THETA_FLIGHT_BIAS_ENABLE = False
THETA_FLIGHT_BIAS = 0.7
USE_CBF = True
CBF_ALPHA = 0.95
F_MAX = 1.3
CBF_Q = np.array([1.0, 1.0], dtype=float)
USE_ROBUST_CBF = _env_bool("HOPPER_USE_ROBUST_CBF", True)
ROBUST_BETA = _env_float("HOPPER_ROBUST_BETA", 0.2)
ROBUST_MARGIN = _env_float("HOPPER_ROBUST_MARGIN", 0.03)
ROBUST_BASE = _env_float("HOPPER_ROBUST_BASE", 0.01)
ROBUST_H_SCALE = _env_float("HOPPER_ROBUST_H_SCALE", 1.0)
ROBUST_C_NOM = _env_float("HOPPER_ROBUST_C_NOM", 0.6)


def load_u_from_json(path: Path, nu: int):
    """Load control sequence from JSON and return first nu channels."""
    data = json.loads(path.read_text())
    if "u" not in data:
        raise KeyError(f"'u' not found in {path}")
    u_raw = np.asarray(data["u"], dtype=float)
    if u_raw.ndim != 2:
        raise ValueError(f"Expected 2D u array, got shape {u_raw.shape}")
    if u_raw.shape[1] < nu:
        raise ValueError(f"Need at least {nu} control dims, got {u_raw.shape[1]}")
    u = u_raw[:, :nu].copy()
    h = float(data.get("dt", H_DEFAULT))
    x_raw = np.asarray(data.get("x", []), dtype=float)
    q_ref = None
    if x_raw.ndim == 2 and x_raw.shape[1] >= 4:
        q_ref = x_raw[:, :4].copy()
    return u, h, q_ref, data


def extract_contact_tiles_from_rollout(model, q, phi_thresh=5e-5, merge_dist=0.08):
    """Return merged x-locations where phi indicates near-contact."""
    q_arr = np.asarray(q, dtype=float)
    n = max(0, len(q_arr) - 1)
    if n <= 0:
        return np.array([], dtype=float)

    contact_x = []
    for i in range(n):
        qk = q_arr[i + 1]
        phi = float(np.asarray(model.signed_distance(qk), dtype=float).reshape(-1)[0])
        if phi <= float(phi_thresh):
            qk = q_arr[i + 1]
            x_foot = float(qk[0] + qk[3] * np.sin(qk[2]))
            contact_x.append(x_foot)
    if not contact_x:
        return np.array([], dtype=float)

    xs = np.sort(np.asarray(contact_x, dtype=float))
    merged = [float(xs[0])]
    for x in xs[1:]:
        if abs(x - merged[-1]) > float(merge_dist):
            merged.append(float(x))
        else:
            merged[-1] = 0.5 * (merged[-1] + float(x))
    return np.asarray(merged, dtype=float)


def _solve_qp(H, g, A, b, lbx, ubx):
    H = np.asarray(H, dtype=float)
    g = np.asarray(g, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    lbx = np.asarray(lbx, dtype=float).reshape(-1)
    ubx = np.asarray(ubx, dtype=float).reshape(-1)

    n = g.size
    try:
        x0 = np.linalg.solve(H + 1e-12 * np.eye(n), -g)
    except Exception:
        x0 = -g
    x0 = np.clip(x0, lbx, ubx)

    res = minimize_qp(H, g, A, b, lbx, ubx, x0, maxiter=100)
    x = np.asarray(res.x, dtype=float).reshape(-1)
    ok = bool(res.success) and bool(np.all(A @ x - b <= 1e-6))
    return x, ok, str(res.message)


def _predict_gamma_linearization(sim, t, u_nom):
    sim_diff = getattr(sim, "_cbf_diff_sim", None)
    if sim_diff is None:
        sim_diff = copy.deepcopy(sim)
        sim_diff.ip.options.diff_sol = True
    else:
        sim_diff.ip.options.diff_sol = True
        sim_diff.prev_z = sim.prev_z
        sim_diff.traj.q[t][:] = sim.traj.q[t]
        sim_diff.traj.q[t + 1][:] = sim.traj.q[t + 1]
        sim_diff.traj.v[t][:] = sim.traj.v[t]

    sim_diff.traj.u[t][:] = np.asarray(u_nom, dtype=float)
    sim_diff.traj.w[t][:] = 0.0
    ok = sim_diff.step(t, diff_sol=True)
    if not ok:
        return False, sim_diff, 0.0, np.zeros_like(u_nom)

    gamma_nom = float(np.asarray(sim_diff.traj.gamma[t], dtype=float).reshape(-1)[0])
    dgamma_du = np.asarray(sim_diff.grad.dgamma1_du1[t], dtype=float)[0, :].copy()
    return True, sim_diff, gamma_nom, dgamma_du


def _update_robust_error_ema(robust_state, h_next_pred, h_next_actual):
    err_signed = float(h_next_actual - h_next_pred)
    err = abs(err_signed)
    under = max(0.0, -err_signed)
    beta = float(ROBUST_BETA)
    robust_state["err_ema"] = (1.0 - beta) * float(
        robust_state.get("err_ema", 0.0)
    ) + beta * err
    robust_state["under_ema"] = (1.0 - beta) * float(
        robust_state.get("under_ema", 0.0)
    ) + beta * under


def cbf_filter(sim, t, u_nom, use_robust_cbf=False, robust_state=None):
    u_nom = np.asarray(u_nom, dtype=float).copy()
    u_min = np.array([-100.0, -100.0], dtype=float)
    u_max = np.array([100.0, 100.0], dtype=float)
    u_nom = np.clip(u_nom, u_min, u_max)

    ok_pred, sim_diff, gamma_nom, dgamma_du = _predict_gamma_linearization(
        sim, t, u_nom
    )
    sim._cbf_diff_sim = sim_diff
    if not ok_pred:
        return u_nom, {"qp_solved": False, "reason": "linearization_failed"}

    gamma_prev = (
        float(np.asarray(sim.traj.gamma[t - 1], dtype=float).reshape(-1)[0])
        if t > 0
        else 0.0
    )
    h_curr = float(F_MAX - gamma_prev)
    h_nom = float(F_MAX - gamma_nom)
    rhs = float((1.0 - CBF_ALPHA) * h_curr)
    robust_delta = 0.0
    if bool(use_robust_cbf):
        prev_err = (
            float(robust_state.get("err_ema", 0.0)) if robust_state is not None else 0.0
        )
        prev_under = (
            float(robust_state.get("under_ema", prev_err))
            if robust_state is not None
            else prev_err
        )
        h_scale = max(float(ROBUST_H_SCALE), 1e-12)
        near_gate = 1.0 / (1.0 + abs(float(h_curr)) / h_scale)
        robust_delta = float(
            near_gate * (prev_under + float(ROBUST_C_NOM) * max(0.0, -float(h_nom)))
        )
    rhs_robust = float(rhs + robust_delta)

    # h_next(u) ~= h_nom - dgamma_du (u-u_nom) >= rhs
    # <=> dgamma_du * u <= h_nom + dgamma_du*u_nom - rhs
    A = dgamma_du.reshape(1, -1)
    b = np.array([h_nom + float(dgamma_du @ u_nom) - rhs_robust], dtype=float)

    H = np.diag(np.asarray(CBF_Q, dtype=float).reshape(-1))
    g = -H @ u_nom

    u_safe, ok_qp, qp_msg = _solve_qp(H, g, A, b, u_min, u_max)
    if not ok_qp:
        u_safe = u_nom.copy()

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
    }
    return u_safe, info


def simulate_from_trajectory(
    model: Hopper,
    residual,
    jacobian_z,
    jacobian_theta,
    traj_path: Path,
    tag: str,
    enable_cbf: bool = True,
    use_robust_cbf: bool = False,
):
    u_seq, h, q_ref, raw = load_u_from_json(traj_path, model.nu)
    T = int(len(u_seq))

    sim = Simulator(
        model=model,
        T=T,
        h=h,
        diff_sol=True,
        residual=residual,
        jacobian_z=jacobian_z,
        jacobian_theta=jacobian_theta,
        kappa_tol=float(KAPPA_TOL),
    )
    sim.ip.options.verbose = False

    q0 = model.nominal_configuration().astype(float)
    q0[:] = Q1
    v0 = V1.copy()
    sim.traj.reset()
    sim.grad.reset()
    sim.prev_z = None
    sim.set_state(q0, v0, 0)

    solved_steps = 0
    solver_status = True
    fall = False
    cbf_qp_fail_count = 0
    cbf_log = []
    robust_state = {"err_ema": 0.0, "under_ema": 0.0}
    for t in range(T):
        u_nom = u_seq[t].copy()
        q_now = sim.traj.q[t + 1]
        q_prev = sim.traj.q[t]
        v_now = (q_now - q_prev) / h
        in_contact = float(model.signed_distance(sim.traj.q[t + 1])[0]) <= float(
            CONTACT_THRESH
        )
        if USE_Q_TRACKING and q_ref is not None and t + 1 < len(q_ref):
            q_ref_now = q_ref[t + 1]
            q_ref_prev = q_ref[t]
            v_ref = (q_ref_now - q_ref_prev) / h

            du0 = float(KP_THETA) * (float(q_ref_now[2]) - float(q_now[2])) + float(
                KD_THETA
            ) * (float(v_ref[2]) - float(v_now[2]))
            du1 = float(KP_R) * (float(q_ref_now[3]) - float(q_now[3])) + float(
                KD_R
            ) * (float(v_ref[3]) - float(v_now[3]))
            du0 = float(np.clip(du0, -float(DU0_TRACK_LIMIT), float(DU0_TRACK_LIMIT)))
            du1 = float(np.clip(du1, -float(DU1_TRACK_LIMIT), float(DU1_TRACK_LIMIT)))

            u_nom[0] = float(u_seq[t][0]) + du0
            u_nom[1] = float(u_seq[t][1]) + du1

        if in_contact and u_nom[1] > 0.0:
            u_nom[1] *= float(U2_CONTACT_SCALE)
        if bool(THETA_FLIGHT_BIAS_ENABLE) and (not in_contact):
            u_nom[0] += float(THETA_FLIGHT_BIAS)

        u_nom[0] = float(np.clip(u_nom[0], -100.0, 100.0))
        u_nom[1] = float(np.clip(u_nom[1], -100.0, 100.0))

        if bool(enable_cbf and USE_CBF):
            u_cmd, cbf_info = cbf_filter(
                sim,
                t,
                u_nom,
                use_robust_cbf=bool(use_robust_cbf),
                robust_state=robust_state,
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
            print(f"[warn] solver failed at t={t} ({tag})")
            break

        if bool(enable_cbf and USE_CBF):
            gamma_actual = float(
                np.asarray(sim.traj.gamma[t], dtype=float).reshape(-1)[0]
            )
            h_next_actual = float(F_MAX - gamma_actual)
            cbf_info["h_next_actual"] = h_next_actual
            if bool(use_robust_cbf):
                _update_robust_error_ema(
                    robust_state,
                    float(cbf_info["h_next_pred"]),
                    h_next_actual,
                )

        solved_steps += 1
        # Fall condition: body height z <= 0
        if float(sim.traj.q[t + 2][1]) <= 0.0:
            fall = True
            print(
                f"[warn] fall detected at t={t} ({tag}), body_z={float(sim.traj.q[t + 2][1]):.6f}"
            )
            break

    status = bool(solver_status and (not fall))

    q_sim = np.array([sim.traj.q[i + 1] for i in range(solved_steps + 1)], dtype=float)
    u_sim = np.array([sim.traj.u[i] for i in range(solved_steps)], dtype=float)
    u_raw = np.asarray(u_seq, dtype=float)
    gamma_sim = np.array([sim.traj.gamma[i] for i in range(solved_steps)], dtype=float)
    b_sim = np.array([sim.traj.b[i] for i in range(solved_steps)], dtype=float)
    contact_tiles_x = extract_contact_tiles_from_rollout(model, q_sim, phi_thresh=5e-5)

    out_dir = OUT_DIR
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
        "input_json_keys": list(raw.keys()),
        "u_raw_shape": list(np.asarray(raw["u"]).shape),
        "u_used_shape": list(u_seq.shape),
        "use_q_tracking": bool(USE_Q_TRACKING),
        "kp_theta": float(KP_THETA),
        "kd_theta": float(KD_THETA),
        "kp_r": float(KP_R),
        "kd_r": float(KD_R),
        "du0_track_limit": float(DU0_TRACK_LIMIT),
        "du1_track_limit": float(DU1_TRACK_LIMIT),
        "u2_contact_scale": float(U2_CONTACT_SCALE),
        "theta_flight_bias_enable": bool(THETA_FLIGHT_BIAS_ENABLE),
        "theta_flight_bias": float(THETA_FLIGHT_BIAS),
        "use_cbf": bool(enable_cbf and USE_CBF),
        "use_robust_cbf": bool(use_robust_cbf),
        "cbf_alpha": float(CBF_ALPHA),
        "f_max": float(F_MAX),
        "cbf_q": np.asarray(CBF_Q, dtype=float).tolist(),
        "robust_beta": float(ROBUST_BETA),
        "robust_margin": float(ROBUST_MARGIN),
        "robust_base": float(ROBUST_BASE),
        "robust_h_scale": float(ROBUST_H_SCALE),
        "robust_c_nom": float(ROBUST_C_NOM),
        "cbf_qp_fail_count": int(cbf_qp_fail_count),
        "q1": Q1.tolist(),
        "v1": V1.tolist(),
        "q": q_sim.tolist(),
        "u": u_sim.tolist(),
        "gamma": gamma_sim.tolist(),
        "b": b_sim.tolist(),
        "contact_tiles_x": contact_tiles_x.tolist(),
        "cbf_log": cbf_log,
    }
    out_json.write_text(json.dumps(payload, indent=2))

    t_q = np.arange(q_sim.shape[0], dtype=float) * h
    t_u = np.arange(u_sim.shape[0], dtype=float) * h
    out_rollout_json = out_dir / f"hopper_rollout_q_u_contact_{tag}.json"
    rollout_payload = {
        "h": float(h),
        "t_state": t_q.tolist(),
        "t_control": t_u.tolist(),
        "q": q_sim.tolist(),
        "u": u_sim.tolist(),
        "gamma": gamma_sim.tolist(),
        "b": b_sim.tolist(),
        "contact_tiles_x": contact_tiles_x.tolist(),
    }
    out_rollout_json.write_text(json.dumps(rollout_payload, indent=2))


    print(f"[write] {out_json}")
    print(f"[write] {out_rollout_json}")

    if len(q_sim) >= 2:
        duration = max(float(h), (len(q_sim) - 1) * float(h))
        vx_mean = float((q_sim[-1, 0] - q_sim[0, 0]) / duration)
    else:
        vx_mean = 0.0
    print(
        "[info] "
        f"{tag}: status={status}, solver_status={solver_status}, fall={fall}, solved_steps={solved_steps}/{T}, "
        f"delta_x={q_sim[-1,0]-q_sim[0,0]:.4f}, vx_mean={vx_mean:.4f}, cbf_qp_fail_count={cbf_qp_fail_count}"
    )

    return {
        "status": bool(status),
        "solver_status": bool(solver_status),
        "fall": bool(fall),
        "solved_steps": int(solved_steps),
        "T": int(T),
        "h": float(h),
        "q": q_sim,
        "u": u_sim,
        "u_ref": u_raw[:, : model.nu].copy(),
        "gamma": gamma_sim,
        "b": b_sim,
        "contact_tiles_x": contact_tiles_x,
        "cbf_log": cbf_log,
        "mode_tag": "rcbf" if bool(use_robust_cbf) else "cbf",
    }
