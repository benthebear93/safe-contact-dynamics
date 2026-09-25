#!/usr/bin/env python3
"""Boundary-focused kappa screening for the seven-coordinate planar Pusher."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.risk_constrained_kappa_selection import (  # noqa: E402
    RolloutConfig,
    add_kappa_grid_args,
    add_risk_args,
    candidate_kappas,
    print_risk_summary,
    select_risk_constrained,
    write_outputs,
)


# Import local modules after adding the repository root to sys.path.
from scripts.planar_push_screening import (  # noqa: E402
    PlanarPushCBFEvaluator,
    build_planar_boundary_scenarios,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Boundary-focused planar-push kappa compatibility screening."
    )
    ap.add_argument(
        "--ref",
        type=Path,
        default=Path("reference_trajectory/planar_push_linear_cimpc_ref.json"),
    )
    add_kappa_grid_args(ap, kappa_min=1e-6, kappa_max=5e-3, num_kappas=12)
    ap.add_argument("--num-scenarios", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--f-max", type=float, default=0.245)
    ap.add_argument("--seed", type=int, default=200)
    add_risk_args(ap, q=0.1, beta=0.1)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/kappa_selection/planar"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    kappas = candidate_kappas(args)
    evaluator = PlanarPushCBFEvaluator(
        ref_path=args.ref, f_max=float(args.f_max), alpha=float(args.alpha)
    )
    q_ref, u_ref, dt = evaluator.q_ref, evaluator.u_ref, evaluator.dt
    scenarios = build_planar_boundary_scenarios(
        q_ref=q_ref,
        u_ref=u_ref,
        dt=dt,
        num_scenarios=int(args.num_scenarios),
        horizon=int(args.horizon),
        seed_base=int(args.seed),
    )
    config = RolloutConfig(eval_horizon=int(args.horizon))
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
            "task": "planar_push",
            "alpha": float(args.alpha),
            "f_max": float(args.f_max),
            "dt": dt,
            "seed": int(args.seed),
            "ref": str(args.ref),
            "ref_sha256": hashlib.sha256(args.ref.read_bytes()).hexdigest(),
            "model": "src.robots.pusher.model_linear.Pusher",
            "nq": 7,
            "nc": 5,
            "contact_index": 4,
            "kappa_accept_ratio": 2.0,
            "margin_definition": "h = f_max - gamma_pusher; h_pred at applied control",
            "scenario_definition": "legacy six deterministic boundary patterns mapped to q7",
            "num_unique_scenarios": len(
                {json.dumps([s["q0"], s["v0"], s["u_ref"]]) for s in scenarios}
            ),
        },
    )
    print_risk_summary(summary, title="Planar push compatibility screening")
    write_outputs(args.outdir, summary, rollout_rows)
    (args.outdir / "scenarios.json").write_text(
        json.dumps(scenarios, indent=2), encoding="utf-8"
    )
    (args.outdir / "step_samples.json").write_text(
        json.dumps(evaluator.rollout_records, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
