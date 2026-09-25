#!/usr/bin/env python3
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from example.cbf_qp import minimize_qp


from src.residual_models.pusher_residual import make_pusher_residual
from src.robots.pusher.model_linear import Pusher
from src.simulator.simulator import Simulator
from src.simulator.trajectory import ContactTrajectory
from src.solver.interior_point_solver import InteriorPointOptions


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None else float(default)


CFG = {
    "ref_path": Path(
        "reference_trajectory/planar_push_linear_cimpc_ref.json"
    ),
    "ref_mode": "push",
    "generated_ref_path": Path(
        "reference_trajectory/planar_push_linear_cimpc_ref.json"
    ),
    "force_ref_regen": True,
    "h": 0.05,
    "kappa": 1e-4,
    "kappa_accept_ratio": 2.0,
    "h_mpc": 5,
    "max_steps": 50,
    "wq_scale": 1.0,
    "wu_scale": 1.0,
    "replay_ref_u": True,
    "run_cimpc": True,
    "goal_x": 0.4,
    "goal_y": 0.4,
    "approach_gain": 1.0,
    "push_gain": 0.25,
    "x_push_min": 0.22,
    "y_push_min": 0.01,
    "y_early_ratio": 0.1,
    "y_early_gain": 0.0,
    "x_taper_start_ratio": 0.5,
    "x_taper_end_scale": 0.2,
    "u_min_mag": 0.12,
    "u_max": 0.30,
    "contact_margin": 0.01,
    "phase1_ratio": 0.2,
    "lift_ratio": 0.01,
    "y_base": 0.0,
    "rot_offset_max": 0.03,
    "use_cbf": _env_bool("PLANAR_USE_CBF", False),
    "use_rcbf": _env_bool("PLANAR_USE_RCBF", True),
    "cbf_alpha": 0.7,
    "f_max": 0.245,
    "robust_margin": _env_float("PLANAR_ROBUST_MARGIN", 0.01),
    "robust_beta": _env_float("PLANAR_ROBUST_BETA", 0.2),
    "robust_h_scale": _env_float("PLANAR_ROBUST_H_SCALE", 5e-2),
    "robust_hnom_gain": _env_float("PLANAR_ROBUST_HNOM_GAIN", 0.2),
    "cbf_contact_index": 4,
    "cbf_qx": 1.0,
    "cbf_qy": 1.0,
    "cbf_u_min_x": -0.5,
    "cbf_u_max_x": 0.5,
    "cbf_u_min_y": -0.5,
    "cbf_u_max_y": 0.5,
    "out_u_seq": Path("reference_trajectory/planar_push_linear_cimpc_u_seq.json"),
}




def load_q7_ref(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    payload = json.loads(path.read_text())
    for key in ("q", "u", "h"):
        if key not in payload:
            raise KeyError(f"Missing key in reference file: {key}")
    q = np.asarray(payload["q"], dtype=float)
    u = np.asarray(payload["u"], dtype=float)
    h = float(payload["h"])
    if q.ndim != 2 or q.shape[1] != 7:
        raise ValueError(f"Expected q shape (N,7), got {q.shape}")
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError(f"Expected u shape (N,2), got {u.shape}")
    return q, u, h


def generate_push_reference(
    model: Pusher,
    residuals,
    h: float,
    kappa_tol: float,
    kappa_accept_ratio: float,
    steps: int,
    goal_x: float,
    goal_y: float,
    approach_gain: float,
    push_gain: float,
    x_push_min: float,
    y_push_min: float,
    y_early_ratio: float,
    y_early_gain: float,
    x_taper_start_ratio: float,
    x_taper_end_scale: float,
    u_min_mag: float,
    u_max: float,
    contact_margin: float,
    phase1_ratio: float,
    lift_ratio: float,
    y_base: float,
    rot_offset_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    r, rz, rtheta = residuals
    sim = Simulator(
        model,
        int(steps),
        h=float(h),
        diff_sol=False,
        residual=r,
        jacobian_z=rz,
        jacobian_theta=rtheta,
        temp_cmd=np.zeros(model.nu),
        kappa_tol=float(kappa_tol),
    )
    sim.ip.options.verbose = False
    sim.ip.options.warn = False
    sim.ip.options.kappa_accept_ratio = float(kappa_accept_ratio)
    sim.traj.reset()
    sim.grad.reset()
    sim.prev_z = None
    q0 = np.asarray(model.nominal_configuration(), dtype=float)
    sim.set_state(q0, np.zeros(model.nq), 0)

    u_ref = []
    # Store state sequence aligned with controls: q_ref[0] is initial state,
    # q_ref[t+1] is the state after applying u_ref[t].
    q_ref = [np.asarray(sim.traj.q[1], dtype=float).copy()]
    n1 = max(1, int(float(phase1_ratio) * int(steps)))
    lift_ratio = float(np.clip(lift_ratio, 0.0, 1.0))
    x_taper_start_ratio = float(np.clip(x_taper_start_ratio, 0.0, 1.0))
    x_taper_end_scale = float(np.clip(x_taper_end_scale, 0.0, 1.0))
    y_early_ratio = float(np.clip(y_early_ratio, 0.0, 1.0))
    y_early_gain = float(max(0.0, y_early_gain))
    y_early_steps = int(y_early_ratio * int(steps))
    for t in range(int(steps)):
        q = np.asarray(sim.traj.q[t + 1], dtype=float)
        box_xy = q[0:2]
        box_theta = float(q[3])
        p_xy = q[4:6]

        # Build box-frame axes in world coordinates.
        c = float(np.cos(box_theta))
        s_th = float(np.sin(box_theta))
        ex = np.array([c, s_th], dtype=float)  # box local +x
        ey = np.array([-s_th, c], dtype=float)  # box local +y

        # Move contact target along box-local y while staying on the side face.
        # pusher should still push toward the box center.
        if t < n1:
            y_slide = float(y_base) * model.r_box
        else:
            s = (t - n1) / max(1, int(steps) - n1 - 1)
            y_slide = (float(y_base) + float(lift_ratio) * s) * model.r_box
        y_slide = float(np.clip(y_slide, -0.8 * model.r_box, 0.8 * model.r_box))

        side_dist = float(model.r_box + model.r_pusher - float(contact_margin))
        # Left face in box frame, with y sliding offset.
        target_p = box_xy + (-side_dist) * ex + y_slide * ey
        e_p = target_p - p_xy
        d_p = float(np.linalg.norm(e_p))

        # Taper x-push floor in late horizon: keeps early motion strong but
        # reduces aggressive forcing near the end (improves feasibility).
        taper_start = int(x_taper_start_ratio * int(steps))
        if t <= taper_start:
            x_floor_t = float(x_push_min)
        else:
            tau = (t - taper_start) / max(1, int(steps) - taper_start - 1)
            x_floor_t = float(x_push_min) * (
                (1.0 - tau) + tau * float(x_taper_end_scale)
            )
        inward = box_xy - p_xy
        inward_norm = float(np.linalg.norm(inward))
        inward_hat = inward / max(inward_norm, 1e-9)

        if d_p > 0.02:
            # Approach target while keeping a minimum inward component.
            u_cmd = float(approach_gain) * e_p
        else:
            # In contact: push toward center + regulate to sliding target.
            u_cmd = float(push_gain) * inward_hat + float(approach_gain) * e_p

        # Enforce minimum inward push along center direction.
        inward_proj = float(np.dot(u_cmd, inward_hat))
        if inward_proj < x_floor_t:
            u_cmd = u_cmd + (x_floor_t - inward_proj) * inward_hat
        # Optional minimum sliding command along local +y.
        y_floor_t = (
            float(y_push_min) * y_early_gain if t < y_early_steps else float(y_push_min)
        )
        slide_proj = float(np.dot(u_cmd, ey))
        if slide_proj < y_floor_t:
            u_cmd = u_cmd + (y_floor_t - slide_proj) * ey

        # Per-axis saturation first.
        u_cmd = np.clip(u_cmd, -float(u_max), float(u_max))
        # Then keep overall command magnitude in a mild range [u_min_mag, u_max].
        u_norm = float(np.linalg.norm(u_cmd))
        u_min_mag = float(max(0.0, u_min_mag))
        u_max_mag = float(max(u_min_mag, u_max))
        if u_norm > 1e-12:
            if u_norm > u_max_mag:
                u_cmd = u_cmd * (u_max_mag / u_norm)
            elif u_norm < u_min_mag:
                u_cmd = u_cmd * (u_min_mag / u_norm)
        else:
            u_cmd = np.array([u_min_mag, 0.0], dtype=float)
        sim.traj.u[t][:] = u_cmd
        sim.traj.w[t][:] = 0.0
        ok = sim.step(t, diff_sol=False)
        if not ok:
            break
        u_ref.append(np.asarray(u_cmd, dtype=float).copy())
        q_ref.append(np.asarray(sim.traj.q[t + 2], dtype=float).copy())
    return np.asarray(q_ref, dtype=float), np.asarray(u_ref, dtype=float)


def build_pusher(mu_body: float = 0.5) -> Pusher:
    r_box = 0.1
    mb = 1.0
    model = Pusher(
        nq=7,
        nu=2,
        nw=0,
        nc=5,
        r_box=r_box,
        r_pusher=0.025,
        mb=mb,
        mp=1.0,
        I=(1.0 / 12.0) * mb * ((2.0 * r_box) ** 2 + (2.0 * r_box) ** 2),
        mu_body=mu_body,
        mu_pusher=0.5,
        gravity=9.81,
    )
    # Start with pusher near the left face of the box (x-axis), almost in contact.
    model.set_nominal_configuration(
        [
            0.0,  # box x
            0.0,  # box y
            model.r_box + 1e-8,  # box z
            1e-8,  # box yaw
            -(model.r_box + model.r_pusher) - 0.01,  # pusher x (left of box)
            0.0,  # pusher y
            model.r_pusher,  # pusher z
        ]
    )
    return model


def build_ref_contact_traj(
    model: Pusher,
    env,
    q7_ref: np.ndarray,
    u_ref: np.ndarray,
    h: float,
    kappa: float,
):
    T = int(u_ref.shape[0])
    model.indices_z()
    traj = ContactTrajectory(T, h, model, env)
    idx_theta = model.indices_theta()

    q_ext = np.vstack([q7_ref, q7_ref[-1], q7_ref[-1]])
    for i in range(T + 2):
        traj.q[i] = q_ext[i].copy()

    z_seed = max(1e-2, np.sqrt(max(kappa, 1e-12)))
    for t in range(T):
        traj.u[t] = u_ref[t].copy()
        traj.w[t] = np.zeros(model.nw)
        traj.gamma[t] = np.full(model.nc, z_seed)
        traj.b[t] = np.full(model.nb, z_seed)
        traj.kappa[t] = np.full(model.nc, kappa)

        z0 = np.zeros(model.num_var(), dtype=float)
        model.initialize_z(z0, traj.q[t + 1])
        traj.z[t] = z0

        th = np.zeros_like(traj.theta[t])
        idx_theta.initialize_theta(
            th,
            traj.q[t],
            traj.q[t + 1],
            traj.u[t],
            traj.w[t],
            model.friction_coefficients(),
            h,
        )
        traj.theta[t] = th
    return traj


def _solve_qp_slsqp(H, g, A, b, lbx, ubx):
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

    res = minimize_qp(H, g, A, b, lbx, ubx, x0, maxiter=120)
    x_opt = np.asarray(res.x, dtype=float).reshape(-1)
    feasible = bool(np.all(A @ x_opt - b <= 1e-8))
    ok = bool(res.success) and feasible
    return x_opt, ok, str(getattr(res, "message", "unknown"))


def _contact_force_and_jacobians(sim, t, u_cmd, contact_index):
    sim_nom = getattr(sim, "_cbf_diff_sim", None)
    if sim_nom is None:
        sim_nom = copy.deepcopy(sim)
    sim_nom.ip.options.diff_sol = True
    sim_nom.prev_z = sim.prev_z
    sim_nom.traj.q[t][:] = sim.traj.q[t]
    sim_nom.traj.q[t + 1][:] = sim.traj.q[t + 1]
    sim_nom.traj.v[t][:] = sim.traj.v[t]
    sim_nom.traj.u[t][:] = np.asarray(u_cmd, dtype=float)
    sim_nom.traj.w[t][:] = 0.0
    status = sim_nom.step(t, diff_sol=True)
    sim._cbf_diff_sim = sim_nom
    if not status:
        return False, sim_nom, 0.0, None

    gamma = (
        float(sim_nom.traj.gamma[t][contact_index])
        if len(sim_nom.traj.gamma[t]) > contact_index
        else 0.0
    )
    dgamma_du = np.array(sim_nom.grad.dgamma1_du1[t][contact_index, :], dtype=float)
    return True, sim_nom, gamma, dgamma_du


def cbf_filter(sim, t, u_nom, params):
    u_nom = np.asarray(u_nom, dtype=float).copy()
    u_min = np.asarray(params["u_min"], dtype=float)
    u_max = np.asarray(params["u_max"], dtype=float)
    u_nom = np.clip(u_nom, u_min, u_max)
    status, _, gamma_nom, dgamma_du = _contact_force_and_jacobians(
        sim, t, u_nom, int(params["contact_index"])
    )
    if not status or dgamma_du is None:
        return False, u_nom, {"reason": "diff_solve_fail"}

    f_max = float(params["f_max"])
    alpha = float(params["cbf_alpha"])
    gamma_prev = (
        float(sim.traj.gamma[t - 1][params["contact_index"]])
        if t > 0 and len(sim.traj.gamma[t - 1]) > params["contact_index"]
        else 0.0
    )
    h_curr = f_max - gamma_prev
    h_next_nom = f_max - gamma_nom
    # h_{k+1}(u) ~= h_next_nom - dgamma_du (u-u_nom)
    # enforce h_{k+1} >= (1-alpha) h_k
    a = -dgamma_du.reshape(1, -1)
    rhs = (1.0 - alpha) * h_curr - h_next_nom + float(a.reshape(-1) @ u_nom)
    robust_delta = 0.0
    if bool(params.get("use_rcbf", False)):
        robust_under_ema = float(params.get("_robust_under_ema", 0.0))
        h_scale = max(float(params.get("robust_h_scale", 1e-3)), 1e-12)
        c_nom = float(params.get("robust_hnom_gain", 0.02))
        near_gate = 1.0 / (1.0 + abs(float(h_curr)) / h_scale)
        robust_delta = near_gate * (
            robust_under_ema + c_nom * max(0.0, -float(h_next_nom))
        )
        rhs += robust_delta
    A = (-a).reshape(1, -1)
    b = np.array([-rhs], dtype=float)

    Q = np.diag(np.asarray(params["qp_weight"], dtype=float))
    H = Q
    g = -Q @ u_nom
    u_star, ok, msg = _solve_qp_slsqp(H, g, A, b, u_min, u_max)
    u_used = u_star if ok else u_nom
    h_next_pred = h_next_nom + float(a.reshape(-1) @ (u_used - u_nom))
    gamma_pred = float(f_max - h_next_pred)
    if not ok:
        return (
            True,
            u_nom,
            {
                "reason": "qp_fail",
                "msg": msg,
                "gamma_nom": gamma_nom,
                "gamma_pred": gamma_pred,
                "robust_delta": robust_delta,
            },
        )

    return (
        True,
        u_star,
        {
            "reason": "ok",
            "msg": msg,
            "gamma_nom": gamma_nom,
            "gamma_pred": gamma_pred,
            "robust_delta": robust_delta,
        },
    )






def main():
    from src.mpc.dynamics_context import create_dynamics_context
    from src.mpc.environment import environment_3d_flat
    from src.mpc.newton import NewtonOptions
    from src.mpc.objective import tracking_objective
    from src.mpc.policy import CIMPCOptions, ci_mpc_policy, policy

    cfg = SimpleNamespace(**CFG)
    if cfg.run_cimpc:
        cfg.replay_ref_u = False
    if cfg.use_rcbf:
        cfg.use_cbf = True

    model_nominal = build_pusher(mu_body=0.5)
    model_sim = build_pusher(mu_body=0.5)

    env = environment_3d_flat(nc_impact=0, cone="LinearizedCone")
    r, rz, rtheta = make_pusher_residual(model_nominal)
    residuals = (r, rz, rtheta)

    if cfg.ref_mode == "file":
        q7_ref, u_ref, h = load_q7_ref(cfg.ref_path)
    else:
        if cfg.generated_ref_path.exists() and (not cfg.force_ref_regen):
            q7_ref, u_ref, h = load_q7_ref(cfg.generated_ref_path)
            print(f"[ref] loaded cached push ref: {cfg.generated_ref_path}")
        else:
            h = float(cfg.h)
            q7_ref, u_ref = generate_push_reference(
                model=model_nominal,
                residuals=residuals,
                h=float(h),
                kappa_tol=float(cfg.kappa),
                kappa_accept_ratio=float(cfg.kappa_accept_ratio),
                steps=int(cfg.max_steps),
                goal_x=float(cfg.goal_x),
                goal_y=float(cfg.goal_y),
                approach_gain=float(cfg.approach_gain),
                push_gain=float(cfg.push_gain),
                x_push_min=float(cfg.x_push_min),
                y_push_min=float(cfg.y_push_min),
                y_early_ratio=float(cfg.y_early_ratio),
                y_early_gain=float(cfg.y_early_gain),
                x_taper_start_ratio=float(cfg.x_taper_start_ratio),
                x_taper_end_scale=float(cfg.x_taper_end_scale),
                u_min_mag=float(cfg.u_min_mag),
                u_max=float(cfg.u_max),
                contact_margin=float(cfg.contact_margin),
                phase1_ratio=float(cfg.phase1_ratio),
                lift_ratio=float(cfg.lift_ratio),
                y_base=float(cfg.y_base),
                rot_offset_max=float(cfg.rot_offset_max),
            )
            if q7_ref.size == 0 or u_ref.size == 0:
                raise RuntimeError(
                    "Generated push reference is empty; adjust gains/goal."
                )
            ref_payload = {
                "h": float(h),
                "q": q7_ref.tolist(),
                "u": u_ref.tolist(),
                "meta": {
                    "mode": "push",
                    "goal_x": float(cfg.goal_x),
                    "goal_y": float(cfg.goal_y),
                    "approach_gain": float(cfg.approach_gain),
                    "push_gain": float(cfg.push_gain),
                    "x_push_min": float(cfg.x_push_min),
                    "y_push_min": float(cfg.y_push_min),
                    "kappa": float(cfg.kappa),
                    "kappa_accept_ratio": float(cfg.kappa_accept_ratio),
                    "y_early_ratio": float(cfg.y_early_ratio),
                    "y_early_gain": float(cfg.y_early_gain),
                    "x_taper_start_ratio": float(cfg.x_taper_start_ratio),
                    "x_taper_end_scale": float(cfg.x_taper_end_scale),
                    "u_min_mag": float(cfg.u_min_mag),
                    "u_max": float(cfg.u_max),
                    "contact_margin": float(cfg.contact_margin),
                    "phase1_ratio": float(cfg.phase1_ratio),
                    "lift_ratio": float(cfg.lift_ratio),
                    "y_base": float(cfg.y_base),
                    "rot_offset_max": float(cfg.rot_offset_max),
                },
            }
            cfg.generated_ref_path.parent.mkdir(parents=True, exist_ok=True)
            cfg.generated_ref_path.write_text(json.dumps(ref_payload, indent=2))
            print(
                f"[generated_ref] saved {cfg.generated_ref_path} "
                f"(steps={len(u_ref)}, goal=({float(cfg.goal_x):.3f},{float(cfg.goal_y):.3f}))"
            )

    model_nominal.set_nominal_configuration(q7_ref[0])
    model_sim.set_nominal_configuration(q7_ref[0])

    s = create_dynamics_context(model_nominal, env)
    s.res.r = r
    s.res.rz = rz
    s.res.rtheta = rtheta

    T = min(int(u_ref.shape[0]), int(cfg.max_steps))
    q7_ref = q7_ref[: T + 1]
    u_ref = u_ref[:T]
    ref_traj = build_ref_contact_traj(
        model_nominal, env, q7_ref=q7_ref, u_ref=u_ref, h=h, kappa=float(cfg.kappa)
    )

    sim = Simulator(
        model_sim,
        T,
        h=h,
        diff_sol=True,
        residual=r,
        jacobian_z=rz,
        jacobian_theta=rtheta,
        policy=None,
        temp_cmd=np.zeros(2),
    )
    sim.ip.options.gamma_reg = 0.1
    sim.ip.options.max_iter = 100
    sim.ip.options.kappa_tol = float(cfg.kappa)
    sim.ip.options.kappa_accept_ratio = float(cfg.kappa_accept_ratio)
    sim.ip.options.max_ls = 10

    sim.traj.reset()
    sim.grad.reset()
    sim.prev_z = None
    sim.set_state(q7_ref[0], np.zeros(model_sim.nq), 0)

    status = True
    u_seq = []
    u_nom_seq = []
    gamma_nom_seq = []
    gamma_safe_seq = []
    gamma_pred_seq = []
    robust_delta_seq = []
    fail_t = None
    if cfg.replay_ref_u:
        print("[mode] replay-ref-u")
        for t in range(len(sim.traj.u)):
            u_cmd = np.asarray(u_ref[t], dtype=float).copy()
            sim.traj.u[t][:] = u_cmd
            sim.traj.w[t][:] = sim.disturbance(sim.traj.q[t + 1], t)
            ok = sim.step(t, sim.diff_sol)
            u_seq.append(u_cmd)
            gamma_t = (
                float(sim.traj.gamma[t][int(cfg.cbf_contact_index)])
                if len(sim.traj.gamma[t]) > int(cfg.cbf_contact_index)
                else np.nan
            )
            gamma_safe_seq.append(gamma_t)
            gamma_nom_seq.append(np.nan)
            gamma_pred_seq.append(np.nan)
            robust_delta_seq.append(0.0)
            if not ok:
                status = False
                fail_t = t
                break
    else:
        wq = float(cfg.wq_scale)
        wu = float(cfg.wu_scale)
        obj = tracking_objective(
            model_nominal,
            friction_dim=4,
            H=int(cfg.h_mpc),
            q=[
                wq * np.diag(np.array([1e-3, 1e-3, 1e-3, 1e-3, 5e-2, 5e-2, 1e-2]))
                for _ in range(int(cfg.h_mpc))
            ],
            u=[wu * np.diag(np.array([1.0, 1.0])) for _ in range(int(cfg.h_mpc))],
            gamma=[
                1e-100 * np.diag(np.ones(model_nominal.nc))
                for _ in range(int(cfg.h_mpc))
            ],
            b=[
                1e-100 * np.diag(np.ones(model_nominal.nb))
                for _ in range(int(cfg.h_mpc))
            ],
        )

        p = ci_mpc_policy(
            ref_traj,
            s,
            obj,
            H_mpc=int(cfg.h_mpc),
            N_sample=1,
            kappa_mpc=float(cfg.kappa),
            mode="configuration",
            ip_opts=InteriorPointOptions(
                gamma_reg=0.1,
                undercut=float("inf"),
                kappa_tol=float(cfg.kappa),
                r_tol=1e-5,
                kappa_reg=1e-5,
                max_iter=100,
                diff_sol=True,
                solver_name="empty_solver",
                max_time=1e5,
                max_ls=10,
            ),
            n_opts=NewtonOptions(r_tol=3e-4, max_iter=10),
            mpc_opts=CIMPCOptions(ip_max_time=1e5),
        )
        sim.policy = p
        cbf_params = {
            "f_max": float(cfg.f_max),
            "cbf_alpha": float(cfg.cbf_alpha),
            "use_rcbf": bool(cfg.use_rcbf),
            "robust_margin": float(cfg.robust_margin),
            "robust_beta": float(cfg.robust_beta),
            "robust_h_scale": float(cfg.robust_h_scale),
            "robust_hnom_gain": float(cfg.robust_hnom_gain),
            "_robust_under_ema": 0.0,
            "contact_index": int(cfg.cbf_contact_index),
            "qp_weight": np.array([float(cfg.cbf_qx), float(cfg.cbf_qy)], dtype=float),
            "u_min": np.array(
                [float(cfg.cbf_u_min_x), float(cfg.cbf_u_min_y)], dtype=float
            ),
            "u_max": np.array(
                [float(cfg.cbf_u_max_x), float(cfg.cbf_u_max_y)], dtype=float
            ),
        }
        for t in range(len(sim.traj.u)):
            try:
                u_nom = np.asarray(policy(p, sim.traj, t), dtype=float)
            except Exception as exc:
                print(f"[cimpc] policy failed at t={t}: {exc}")
                status = False
                fail_t = t
                break
            u_cmd = u_nom.copy()
            cbf_info = {"reason": "disabled"}
            if bool(cfg.use_cbf):
                ok_cbf, u_cmd, cbf_info = cbf_filter(sim, t, u_nom, cbf_params)
                if not ok_cbf:
                    print(f"[cbf] failed at t={t}: {cbf_info}")
                    status = False
                    fail_t = t
                    break
            u_nom_seq.append(u_nom.copy())
            gamma_nom_seq.append(float(cbf_info.get("gamma_nom", np.nan)))
            gamma_pred_seq.append(float(cbf_info.get("gamma_pred", np.nan)))
            robust_delta_seq.append(float(cbf_info.get("robust_delta", 0.0)))
            sim.traj.u[t][:] = u_cmd
            sim.traj.w[t][:] = sim.disturbance(sim.traj.q[t + 1], t)
            ok = sim.step(t, sim.diff_sol)
            u_seq.append(np.asarray(u_cmd, dtype=float).copy())
            gamma_t = (
                float(sim.traj.gamma[t][int(cfg.cbf_contact_index)])
                if len(sim.traj.gamma[t]) > int(cfg.cbf_contact_index)
                else np.nan
            )
            gamma_safe_seq.append(gamma_t)
            if (
                bool(cfg.use_rcbf)
                and np.isfinite(gamma_t)
                and np.isfinite(gamma_pred_seq[-1])
            ):
                h_act = float(cfg.f_max) - float(gamma_t)
                h_pred = float(cfg.f_max) - float(gamma_pred_seq[-1])
                under = max(0.0, float(h_pred - h_act))
                beta = float(cfg.robust_beta)
                cbf_params["_robust_under_ema"] = (1.0 - beta) * float(
                    cbf_params.get("_robust_under_ema", 0.0)
                ) + beta * under
            if not ok:
                status = False
                fail_t = t
                break

    n_cmp = min(len(ref_traj.q), len(sim.traj.q), len(u_seq) + 1)
    if n_cmp >= 2:
        q_ref_cmp = np.asarray(ref_traj.q[:n_cmp], dtype=float)
        q_sim_cmp = np.asarray(sim.traj.q[:n_cmp], dtype=float)
        dq = q_sim_cmp - q_ref_cmp
        rms_all = float(np.sqrt(np.mean(dq * dq)))
        final_l2_all = float(np.linalg.norm(dq[-1]))
        # Box pose subset: x, y, yaw
        dq_box = dq[:, [0, 1, 3]]
        rms_box = float(np.sqrt(np.mean(dq_box * dq_box)))
        final_l2_box = float(np.linalg.norm(dq_box[-1]))
        print(
            f"[tracking] n={n_cmp}, rms_all={rms_all:.6e}, final_all={final_l2_all:.6e}, "
            f"rms_box={rms_box:.6e}, final_box={final_l2_box:.6e}"
        )

    print(f"[cimpc] status={status}, fail_t={fail_t}, u_len={len(u_seq)}")
    if bool(cfg.use_cbf) and len(gamma_safe_seq) > 0:
        g_safe = np.asarray(gamma_safe_seq, dtype=float)
        g_nom = np.asarray(gamma_nom_seq, dtype=float)
        mode_lbl = "rCBF" if bool(cfg.use_rcbf) else "CBF"
        n_viol = int(np.sum(g_safe > float(cfg.f_max)))
        print(
            f"[{mode_lbl}] n_viol={n_viol}/{len(g_safe)}, "
            f"max_gamma_safe={float(np.nanmax(g_safe)):.6f}, "
            f"max_gamma_nom={float(np.nanmax(g_nom)):.6f}, F_max={float(cfg.f_max):.6f}"
        )

    out = {
        "h": float(h),
        "q0": q7_ref[0].tolist(),
        "u": [u.tolist() for u in u_seq],
        "u_nom": [u.tolist() for u in u_nom_seq],
        "gamma_nom": [float(x) for x in gamma_nom_seq],
        "gamma_pred": [float(x) for x in gamma_pred_seq],
        "gamma_safe": [float(x) for x in gamma_safe_seq],
        "robust_delta": [float(x) for x in robust_delta_seq],
        "fail_t": fail_t,
        "use_cbf": bool(cfg.use_cbf),
        "use_rcbf": bool(cfg.use_rcbf),
        "f_max": float(cfg.f_max),
        "ref_mode": str(cfg.ref_mode),
        "ref_path": str(cfg.ref_path) if cfg.ref_mode == "file" else None,
    }
    cfg.out_u_seq.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_u_seq.write_text(json.dumps(out, indent=2))
    print(f"[write] {cfg.out_u_seq}")


if __name__ == "__main__":
    main()
