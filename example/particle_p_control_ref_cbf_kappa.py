#!/usr/bin/env python3

import numpy as np

from example.particle_p_controller_cbf_kappa import run_sim
from src.cpp.particle_residual import make_particle_residual
from src.robots.particle.model_linear import Particle


def _build_base_params(kappa: float, alpha: float, f_max: float, seed: int):
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
        "robust_margin": 0.0,
        "robust_mode": "ema_max",
        "robust_base": 0.0,
        "robust_h_scale": 1e-3,
        "robust_hk_gain": 0.0,
        "robust_hnom_gain": 0.0,
        "robust_jac_gain": 0.0,
        "robust_eta_lr": 0.2,
        "robust_eta_min": 0.5,
        "robust_eta_max": 2.0,
        "robust_under_target": 5e-4,
        "robust_over_comp_gain": 0.0,
        "robust_beta": 0.2,
        "_robust_err_ema": 0.0,
    }
    params["_qp_timing"] = {"total_s": 0.0, "by_solver_s": {}, "calls": 0}
    params["_diff_timing"] = {"total_s": 0.0, "sync_s": 0.0, "step_s": 0.0, "calls": 0}
    params["_lin_err_log"] = []
    params["_qp_stats"] = {"success": 0, "fail": 0}
    params["_cbf_stats"] = {
        "gamma_nom": [],
        "gamma_safe": [],
        "gamma_safe_minus_nom": [],
        "delta_gamma_pred": [],
        "delta_gamma_actual": [],
        "delta_gamma_ratio": [],
        "delta_u_norm": [],
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
    }
    params["_step_log"] = []
    return params


def _run_mode(
    mode: str,
    *,
    kappa: float,
    alpha: float,
    f_max: float,
    seed: int,
    robust_h_scale: float,
    robust_c_nom: float,
):
    params = _build_base_params(kappa, alpha, f_max, seed)
    if mode == "rcbf":
        params["use_robust_cbf"] = True
        params["robust_margin"] = 0.0
        params["robust_mode"] = "under_gate_plus"
        params["robust_base"] = 0.0
        params["robust_h_scale"] = float(robust_h_scale)
        params["robust_hk_gain"] = 0.0
        params["robust_hnom_gain"] = float(robust_c_nom)
        params["robust_jac_gain"] = 0.0
    elif mode != "cbf":
        raise ValueError(f"Unknown mode: {mode}")

    sim_res = run_sim(params, use_cbf=True)
    rows = params["_step_log"]
    t = np.array([float(r["time_s"]) for r in rows], dtype=float)
    gamma_actual = np.array([float(r["gamma_actual"]) for r in rows], dtype=float)
    gamma_nom = np.array([float(r["gamma_nom"]) for r in rows], dtype=float)
    u_nom_z = np.array([float(r["u_nom_z"]) for r in rows], dtype=float)
    u_star_z = np.array([float(r["u_star_z"]) for r in rows], dtype=float)
    traj_q = np.asarray(sim_res["traj_q"], dtype=float)
    return {
        "t": t,
        "gamma_actual": gamma_actual,
        "gamma_nom": gamma_nom,
        "u_nom_z": u_nom_z,
        "u_star_z": u_star_z,
        "traj_q": traj_q,
    }






