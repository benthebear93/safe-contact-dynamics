#!/usr/bin/env python3
"""Boundary-focused kappa compatibility screening for box pivot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.risk_constrained_kappa_selection import (
    EvaluatorOutput,
    RolloutConfig,
    ScreeningEvaluator,
    add_kappa_grid_args,
    add_risk_args,
    candidate_kappas,
    print_risk_summary,
    select_risk_constrained,
    write_outputs,
)


import numpy as np

from example.tipover_push_ref_cbf_kappa import (
    build_tipover_model,
    load_ref_traj,
    run_sim,
)
from src.residual_models.tipover_push_residual import make_tipover_push_residual


def build_tipover_boundary_scenarios(
    q_ref: np.ndarray,
    u_ref: np.ndarray,
    *,
    num_scenarios: int = 4,
    horizon: int = 10,
) -> List[Dict[str, Any]]:
    """Construct a small probe set around the reference initial condition.

    The reference is already aggressive, so the probe set uses only mild state
    perturbations and slight input scaling.
    """

    q0_base = np.asarray(q_ref[0], dtype=float).copy()
    if q_ref.shape[0] >= 2:
        v0_base = np.asarray(q_ref[1] - q_ref[0], dtype=float).copy()
    else:
        v0_base = np.zeros_like(q0_base)
    u_nom = np.asarray(u_ref[:horizon], dtype=float).copy()

    theta_offsets = [-0.01, 0.0, 0.01, 0.015]
    px_offsets = [-0.004, 0.0, 0.003, 0.006]
    pusher_height_offsets = [0.0, 0.0, 0.003, -0.002]
    u_scales = [0.92, 1.0, 1.04, 1.08]

    scenarios: List[Dict[str, Any]] = []
    for i in range(num_scenarios):
        q0 = q0_base.copy()
        v0 = v0_base.copy()
        q0[0] += px_offsets[i % len(px_offsets)]
        q0[3] += theta_offsets[i % len(theta_offsets)]
        q0[6] += pusher_height_offsets[i % len(pusher_height_offsets)]
        u_probe = np.asarray(u_scales[i % len(u_scales)] * u_nom, dtype=float)
        scenarios.append(
            {
                "q0": q0.tolist(),
                "v0": v0.tolist(),
                "u_ref": u_probe.tolist(),
                "tag": f"tipover_boundary_{i}",
            }
        )
    return scenarios


class TipoverCBFEvaluator(ScreeningEvaluator):
    def __init__(
        self,
        *,
        ref_path: Path,
        f_max: float = 0.5,
        alpha: float = 0.05,
        use_cpp_jac: bool = True,
    ):
        ref = load_ref_traj(ref_path)
        self.q_ref = np.asarray(ref["q"], dtype=float)
        self.u_ref = np.asarray(ref["u"], dtype=float)
        self.dt = float(ref["h"])
        self.f_max = float(f_max)
        self.alpha = float(alpha)
        self.q0_ref = self.q_ref[0].copy()
        self.v0_ref = (
            (self.q_ref[1] - self.q_ref[0]) / self.dt
            if self.q_ref.shape[0] >= 2
            else np.zeros(self.q_ref.shape[1], dtype=float)
        )
        self.model = build_tipover_model(self.q0_ref.copy(), ref)
        self.model.set_nominal_configuration(self.q0_ref.copy())
        self.residuals = make_tipover_push_residual(self.model, use_cpp_jac=use_cpp_jac)

    def evaluate(
        self, kappa: float, scenario: Mapping[str, Any], config: RolloutConfig
    ) -> EvaluatorOutput:
        num_steps = min(
            int(config.eval_horizon), int(len(scenario.get("u_ref", self.u_ref)))
        )
        q0 = np.asarray(scenario.get("q0", self.q0_ref), dtype=float)
        v0 = np.asarray(scenario.get("v0", self.v0_ref), dtype=float)
        u_ref = np.asarray(scenario.get("u_ref", self.u_ref[:num_steps]), dtype=float)[
            :num_steps
        ]
        theta_ref = np.repeat(float(self.q_ref[-1, 3]), num_steps)
        if self.q_ref.shape[0] >= num_steps:
            theta_ref = self.q_ref[:num_steps, 3].copy()

        params = {
            "model": self.model,
            "residuals": self.residuals,
            "dt": float(self.dt),
            "num_steps": int(num_steps),
            "kappa_tol": float(kappa),
            "u_ref": u_ref,
            "theta_ref": theta_ref,
            "q0": q0,
            "v0": v0,
            "u_min": np.array([-6.0, -6.0], dtype=float),
            "u_max": np.array([6.0, 6.0], dtype=float),
            "F_max": float(self.f_max),
            "cbf_alpha": float(self.alpha),
            "qp_Q": np.array([1.0, 1.0], dtype=float),
            "n_floor_contacts": 4,
            "constrained_corner_idx": 0,
            "theta_idx": 3,
            "theta_target": float(theta_ref[-1]),
            "use_robust_cbf": False,
            "eps_robust": 0.0,
            "robust_mode": "under_gate_plus",
            "robust_base": 0.0,
            "robust_h_scale": 1e-3,
            "robust_hk_gain": 0.0,
            "robust_hnom_gain": 0.02,
            "robust_jac_gain": 0.0,
            "robust_beta": 0.2,
            "use_rhs_smoothing": True,
            "rhs_smooth_alpha": 0.2,
            "_rhs_cbf_prev": 0.0,
            "_robust_err_ema": 0.0,
            "_robust_under_ema": 0.0,
            "_robust_over_ema": 0.0,
        }
        res = run_sim(params, mode="cbf")
        gamma = np.asarray(res.get("gamma_corner0", np.zeros(0)), dtype=float)
        h_true_seq = np.asarray(res.get("h_actual", np.zeros(0)), dtype=float)
        n = min(
            len(gamma),
            len(h_true_seq) if h_true_seq.size else len(gamma),
            int(config.eval_horizon),
        )
        h_true = h_true_seq[:n] if h_true_seq.size else float(self.f_max) - gamma[:n]
        # Tipover selection currently uses realized h as a proxy because the
        # original probe workflow was defined that way.
        h_pred = h_true.copy()
        u_dev = np.asarray(res.get("u_delta_norm", np.zeros(0)), dtype=float)[:n]
        failed = not bool(res.get("status", False))
        return EvaluatorOutput(
            h_pred=np.asarray(h_pred, dtype=float),
            h_true=np.asarray(h_true, dtype=float),
            control_deviation=np.asarray(u_dev, dtype=float),
            failed=failed,
            info={
                "tag": scenario.get("tag", ""),
                "success": bool(res.get("success", False)),
                "n_viol": int(res.get("n_viol", 0)),
                "max_force": float(res.get("max_force", 0.0)),
            },
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Boundary-focused box-pivot kappa compatibility screening.")
    ap.add_argument("--ref", type=Path, default=Path("reference_trajectory/tipover_ref_traj.json"))
    add_kappa_grid_args(ap, kappa_min=1e-6, kappa_max=1e-3, num_kappas=10)
    ap.add_argument("--num-scenarios", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--f-max", type=float, default=0.5)
    ap.add_argument("--use-cpp-jac", action="store_true", default=True, help="Compatibility flag; C++ Jacobians are always enabled.")
    add_risk_args(ap, q=0.1, beta=0.1)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/kappa_selection/box_pivot"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    kappas = candidate_kappas(args)
    ref = load_ref_traj(args.ref)
    q_ref = np.asarray(ref["q"], dtype=float)
    u_ref = np.asarray(ref["u"], dtype=float)
    scenarios = build_tipover_boundary_scenarios(
        q_ref=q_ref,
        u_ref=u_ref,
        num_scenarios=int(args.num_scenarios),
        horizon=int(args.horizon),
    )
    config = RolloutConfig(eval_horizon=int(args.horizon))
    evaluator = TipoverCBFEvaluator(
        ref_path=args.ref,
        f_max=float(args.f_max),
        alpha=float(args.alpha),
        use_cpp_jac=bool(args.use_cpp_jac),
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
            "task": "box_pivot",
            "source_task": "tipover",
            "alpha": float(args.alpha),
            "f_max": float(args.f_max),
            "dt": float(ref["h"]),
            "ref": str(args.ref),
            "use_cpp_jac": bool(args.use_cpp_jac),
        },
    )
    print_risk_summary(summary, title="Box pivot compatibility screening")
    write_outputs(args.outdir, summary, rollout_rows)


if __name__ == "__main__":
    main()
