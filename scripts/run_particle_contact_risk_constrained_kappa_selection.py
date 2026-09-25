#!/usr/bin/env python3
"""Boundary-focused kappa compatibility screening for particle contact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from example.particle_p_controller_cbf_kappa import run_sim
from src.residual_models.particle_residual import make_particle_residual
from src.robots.particle.model_linear import Particle

from scripts.risk_constrained_kappa_selection import (
    EvaluatorOutput,
    RolloutConfig,
    add_kappa_grid_args,
    add_risk_args,
    candidate_kappas,
    print_risk_summary,
    select_risk_constrained,
    write_outputs,
)




def _run_particle_case(
    kappa: float,
    alpha: float,
    f_max: float,
    use_slack: bool,
    seed: int,
    q0_override=None,
    v0_override=None,
    gamma_ref_override=None,
    dt_override: float | None = None,
    num_steps_override: int | None = None,
    use_robust_cbf: bool = False,
    eps_robust: float = 0.0,
    robust_mode: str = "ema_max",
    robust_base: float = 0.0,
    robust_h_scale: float = 1e-3,
    robust_hk_gain: float = 0.0,
    robust_hnom_gain: float = 0.0,
    robust_jac_gain: float = 0.0,
    robust_eta_lr: float = 0.2,
    robust_eta_min: float = 0.5,
    robust_eta_max: float = 2.0,
    robust_under_target: float = 5e-4,
    robust_over_comp_gain: float = 0.0,
):
    np.random.seed(seed)

    particle = Particle(nq=3, nc=1, nu=3, nw=0, nb=4, mb=1, mu=0.5, gravity=9.81)
    residuals = make_particle_residual(particle)

    dt = 0.01 if dt_override is None else float(dt_override)
    num_steps = 400 if num_steps_override is None else int(num_steps_override)
    t = np.arange(0.0, dt * num_steps, dt)
    if gamma_ref_override is None:
        gamma_ref = 0.225 + 0.075 * np.sin(2.0 * np.pi * 0.5 * t)
    else:
        gamma_ref = np.asarray(gamma_ref_override, dtype=float).reshape(-1)
        if gamma_ref.size != num_steps:
            raise ValueError(
                f"gamma_ref_override size ({gamma_ref.size}) must match num_steps ({num_steps})"
            )

    q0 = particle.nominal_configuration()
    q0[2] = particle.r
    v0 = np.zeros(particle.nq)
    if q0_override is not None:
        q0 = np.asarray(q0_override, dtype=float).copy()
    if v0_override is not None:
        v0 = np.asarray(v0_override, dtype=float).copy()

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
        "use_slack": bool(use_slack),
        "slack_weight": 1e4,
        "contact_index": 0,
        "use_impact_constraint": False,
        "phi_thresh": 1e-4,
        "v_n_max": 0.2,
        "use_robust_cbf": bool(use_robust_cbf),
        "robust_margin": float(eps_robust),
        "robust_mode": str(robust_mode),
        "robust_base": float(robust_base),
        "robust_h_scale": float(robust_h_scale),
        "robust_hk_gain": float(robust_hk_gain),
        "robust_hnom_gain": float(robust_hnom_gain),
        "robust_jac_gain": float(robust_jac_gain),
        "robust_eta_lr": float(robust_eta_lr),
        "robust_eta_min": float(robust_eta_min),
        "robust_eta_max": float(robust_eta_max),
        "robust_under_target": float(robust_under_target),
        "robust_over_comp_gain": float(robust_over_comp_gain),
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

    _ = run_sim(params, use_cbf=True)
    return params["_step_log"], params



def build_particle_boundary_scenarios(
    *,
    num_scenarios: int = 6,
    seed_base: int = 100,
    dt: float = 0.01,
    num_steps: int = 120,
) -> List[Dict[str, Any]]:
    """Construct a small, boundary-focused scenario set for particle selection.

    The intent is not to approximate the full benchmark distribution. Instead, this
    set deliberately targets near-boundary states where kappa affects safety most.
    """

    from src.robots.particle.model_linear import Particle

    particle = Particle(nq=3, nc=1, nu=3, nw=0, nb=4, mb=1, mu=0.5, gravity=9.81)
    base_q = particle.nominal_configuration()
    base_v = np.zeros(particle.nq, dtype=float)
    t = np.arange(0.0, dt * num_steps, dt)
    refs = [
        0.245 + 0.085 * np.sin(2.0 * np.pi * 0.7 * t + 0.2),
        0.255 + 0.090 * np.sin(2.0 * np.pi * 0.8 * t + 0.5),
        0.235 + 0.080 * np.sin(2.0 * np.pi * 0.6 * t - 0.4),
    ]
    # Deliberately cluster around contact boundary with mostly descending vertical velocity.
    z_offsets = [-0.0025, -0.0010, 0.0, 0.0015, 0.0035, 0.0060]
    vz_vals = [-0.14, -0.10, -0.06, -0.03, 0.0, 0.02]

    scenarios: List[Dict[str, Any]] = []
    for i in range(num_scenarios):
        q0 = base_q.copy()
        v0 = base_v.copy()
        q0[2] = particle.r + z_offsets[i % len(z_offsets)]
        v0[2] = vz_vals[i % len(vz_vals)]
        scenarios.append(
            {
                "seed": int(seed_base + i),
                "q0": q0.tolist(),
                "v0": v0.tolist(),
                "gamma_ref": refs[i % len(refs)].tolist(),
                "dt": float(dt),
                "num_steps": int(num_steps),
                "tag": f"boundary_{i}",
            }
        )
    return scenarios


class ParticleCBFEvaluator:
    """Particle CBF rollout adapter for boundary-focused compatibility screening."""

    def __init__(self, *, alpha: float = 0.95, f_max: float = 0.25, use_slack: bool = False):
        self.alpha = float(alpha)
        self.f_max = float(f_max)
        self.use_slack = bool(use_slack)

    def evaluate(self, kappa: float, scenario: Mapping[str, Any], config: RolloutConfig) -> EvaluatorOutput:
        seed = int(scenario.get("seed", 0))
        rows, _ = _run_particle_case(
            kappa=float(kappa),
            alpha=self.alpha,
            f_max=self.f_max,
            use_slack=self.use_slack,
            seed=seed,
            q0_override=scenario.get("q0"),
            v0_override=scenario.get("v0"),
            gamma_ref_override=scenario.get("gamma_ref"),
            dt_override=scenario.get("dt"),
            num_steps_override=scenario.get("num_steps"),
            use_robust_cbf=False,
        )
        if not rows:
            return EvaluatorOutput(
                h_pred=np.zeros(0, dtype=float),
                h_true=np.zeros(0, dtype=float),
                control_deviation=np.zeros(0, dtype=float),
                failed=True,
                info={"reason": "empty_rollout"},
            )

        hz = min(int(config.eval_horizon), len(rows))
        rows = rows[:hz]
        h_pred = np.array([float(r.get("h_next_pred", np.nan)) for r in rows], dtype=float)
        h_true = np.array([float(r.get("h_next_actual", np.nan)) for r in rows], dtype=float)

        # Support either full xyz controls or z-only logs.
        u_nom = np.zeros((hz, 3), dtype=float)
        u_star = np.zeros((hz, 3), dtype=float)
        for i, r in enumerate(rows):
            u_nom[i, 0] = float(r.get("u_nom_x", r.get("u_nom", 0.0)))
            u_nom[i, 1] = float(r.get("u_nom_y", 0.0))
            u_nom[i, 2] = float(r.get("u_nom_z", 0.0))
            u_star[i, 0] = float(r.get("u_star_x", r.get("u_safe_x", 0.0)))
            u_star[i, 1] = float(r.get("u_star_y", r.get("u_safe_y", 0.0)))
            u_star[i, 2] = float(r.get("u_star_z", r.get("u_safe_z", 0.0)))
        u_dev = np.linalg.norm(u_star - u_nom, axis=1)

        qp_ok_flags = np.array([int(r.get("qp_ok", 1)) for r in rows], dtype=int)
        failed = bool(np.any(qp_ok_flags == 0) or np.any(~np.isfinite(h_true)) or np.any(~np.isfinite(h_pred)))
        return EvaluatorOutput(
            h_pred=h_pred,
            h_true=h_true,
            control_deviation=u_dev,
            failed=failed,
            info={"seed": seed},
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Boundary-focused particle contact kappa compatibility screening."
    )
    add_kappa_grid_args(ap, kappa_min=1e-6, kappa_max=5e-3, num_kappas=12)
    ap.add_argument("--num-scenarios", type=int, default=40)
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--f-max", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--use-slack", action="store_true")
    add_risk_args(ap, q=0.1, beta=0.1)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/kappa_selection/particle"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    kappas = candidate_kappas(args)
    scenarios = build_particle_boundary_scenarios(
        num_scenarios=int(args.num_scenarios),
        seed_base=int(args.seed),
        dt=float(args.dt),
        num_steps=int(args.horizon),
    )
    config = RolloutConfig(eval_horizon=int(args.horizon))
    evaluator = ParticleCBFEvaluator(
        alpha=float(args.alpha),
        f_max=float(args.f_max),
        use_slack=bool(args.use_slack),
    )
    summary, rollout_rows = select_risk_constrained(
        kappas,
        scenarios,
        evaluator=evaluator,
        config=config,
        q=float(args.q),
        beta=float(args.beta),
        score_tol=float(args.score_tol),
        score_select_tol=float(args.score_select_tol),
        kappa_floor=float(args.kappa_floor),
        violation_tol=float(args.violation_tol),
        failure_control_penalty=float(args.failure_control_penalty),
        metadata={
            "task": "particle_contact",
            "alpha": float(args.alpha),
            "f_max": float(args.f_max),
            "use_slack": bool(args.use_slack),
            "dt": float(args.dt),
            "seed": int(args.seed),
        },
    )
    print_risk_summary(summary, title="Particle contact compatibility screening")
    write_outputs(args.outdir, summary, rollout_rows)


if __name__ == "__main__":
    main()
