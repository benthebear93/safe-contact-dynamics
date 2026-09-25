#!/usr/bin/env python3
"""Boundary-focused kappa compatibility screening for hopper.

Unlike the older hopper boundary probe, this script does not treat the
``before`` and ``after`` reference files as the only two scenarios. It builds
boundary-focused rollout scenarios by taking near-contact windows from the
reference trajectories and perturbing the initial state and input segment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

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

import example.hopper_simulate_test_before_cbf as hopper
from src.simulator.simulator import Simulator


def _build_hopper_model() -> hopper.Hopper:
    return hopper.Hopper(
        nq=4,
        nu=2,
        nw=0,
        nc=1,
        mb=1.0,
        ml=0.1,
        jb=0.25,
        jl=0.025,
        mu=1.0,
        gravity=9.81,
        r_min=0.1,
        r_max=0.5,
    )


def _load_hopper_reference(path: Path, model: hopper.Hopper) -> tuple[np.ndarray, np.ndarray, float]:
    u_ref, dt, q_ref, _ = hopper.load_u_from_json(path, model.nu)
    if q_ref is None:
        q0 = np.asarray(hopper.Q1, dtype=float).reshape(1, -1)
        q_ref = np.repeat(q0, len(u_ref) + 1, axis=0)
    return np.asarray(q_ref, dtype=float), np.asarray(u_ref, dtype=float), float(dt)


def build_hopper_boundary_scenarios(
    *,
    model: hopper.Hopper,
    scenario_set: str,
    num_scenarios: int,
    horizon: int,
    seed: int,
    rho_q: float,
    rho_v: float,
    u_noise: float,
) -> list[dict[str, Any]]:
    refs: list[tuple[str, Path]] = []
    if scenario_set in ("both", "after"):
        refs.append(("after", hopper.TRAJ_PATH_AFTER))
    if scenario_set in ("both", "before"):
        refs.append(("before", hopper.TRAJ_PATH_BEFORE))
    if not refs:
        raise ValueError(f"No hopper references selected for scenario_set={scenario_set}")

    ref_data = []
    for tag, path in refs:
        q_ref, u_ref, dt = _load_hopper_reference(path, model)
        max_start = min(len(u_ref), len(q_ref) - 1) - int(horizon)
        if max_start < 0:
            raise ValueError(f"Reference {path} is shorter than horizon={horizon}")
        phi = np.array(
            [float(np.asarray(model.signed_distance(q_ref[i]), dtype=float).reshape(-1)[0]) for i in range(max_start + 1)],
            dtype=float,
        )
        # Contact-boundary focused: prefer starts closest to contact onset.
        order = np.argsort(np.abs(phi - float(hopper.CONTACT_THRESH)))
        ref_data.append((tag, path, q_ref, u_ref, dt, order))

    rng = np.random.default_rng(int(seed))
    scenarios: list[dict[str, Any]] = []
    for sid in range(int(num_scenarios)):
        tag, path, q_ref, u_ref, dt, order = ref_data[sid % len(ref_data)]
        start = int(order[(sid // len(ref_data)) % len(order)])
        q_seg = q_ref[start : start + int(horizon) + 1].copy()
        u_seg = u_ref[start : start + int(horizon)].copy()
        v_seg = np.diff(q_seg, axis=0) / dt if len(q_seg) >= 2 else np.zeros((1, model.nq))

        q_span = np.maximum(np.ptp(q_seg, axis=0), 1e-3)
        v_span = np.maximum(np.ptp(v_seg, axis=0), 1e-2)
        q0 = q_seg[0].copy()
        v0 = v_seg[0].copy() if len(v_seg) else np.asarray(hopper.V1, dtype=float).copy()
        q0 += rng.uniform(-float(rho_q) * q_span, float(rho_q) * q_span, size=model.nq)
        v0 += rng.uniform(-float(rho_v) * v_span, float(rho_v) * v_span, size=model.nq)
        u_probe = u_seg.copy()
        u_probe += rng.uniform(-float(u_noise), float(u_noise), size=u_probe.shape)

        scenarios.append(
            {
                "tag": f"hopper_{tag}_{sid:03d}_s{start}",
                "seed": int(seed + sid),
                "source": str(path),
                "start": start,
                "dt": float(dt),
                "q0": q0.tolist(),
                "v0": v0.tolist(),
                "q_ref": q_seg.tolist(),
                "u_ref": u_probe.tolist(),
            }
        )
    return scenarios


class HopperPerturbedCBFEvaluator(ScreeningEvaluator):
    def __init__(self, *, alpha: float = 0.95, f_max: float = 1.3, use_robust_cbf: bool = False):
        self.alpha = float(alpha)
        self.f_max = float(f_max)
        self.use_robust_cbf = bool(use_robust_cbf)
        self.model = _build_hopper_model()
        self.residual, self.jacobian_z, self.jacobian_theta = hopper.make_hopper_residual(self.model)

    def evaluate(self, kappa: float, scenario: Mapping[str, Any], config: RolloutConfig) -> EvaluatorOutput:
        old_kappa = float(hopper.KAPPA_TOL)
        old_alpha = float(hopper.CBF_ALPHA)
        old_fmax = float(hopper.F_MAX)
        try:
            hopper.KAPPA_TOL = float(kappa)
            hopper.CBF_ALPHA = float(self.alpha)
            hopper.F_MAX = float(self.f_max)
            return self._evaluate_with_globals(float(kappa), scenario, config)
        finally:
            hopper.KAPPA_TOL = old_kappa
            hopper.CBF_ALPHA = old_alpha
            hopper.F_MAX = old_fmax

    def _evaluate_with_globals(
        self, kappa: float, scenario: Mapping[str, Any], config: RolloutConfig
    ) -> EvaluatorOutput:
        u_ref = np.asarray(scenario["u_ref"], dtype=float)
        q_ref = np.asarray(scenario.get("q_ref", []), dtype=float)
        dt = float(scenario.get("dt", hopper.H_DEFAULT))
        T = min(int(config.eval_horizon), int(len(u_ref)))

        sim = Simulator(
            model=self.model,
            T=T,
            h=dt,
            diff_sol=True,
            residual=self.residual,
            jacobian_z=self.jacobian_z,
            jacobian_theta=self.jacobian_theta,
            kappa_tol=float(kappa),
        )
        sim.ip.options.verbose = False
        sim.traj.reset()
        sim.grad.reset()
        sim.prev_z = None
        sim.set_state(
            np.asarray(scenario["q0"], dtype=float),
            np.asarray(scenario["v0"], dtype=float),
            0,
        )

        h_pred: list[float] = []
        h_true: list[float] = []
        u_dev: list[float] = []
        robust_state = {"err_ema": 0.0, "under_ema": 0.0}
        failed = False

        for t in range(T):
            u_nom = np.asarray(u_ref[t], dtype=float).copy()
            q_now = sim.traj.q[t + 1]
            q_prev = sim.traj.q[t]
            v_now = (q_now - q_prev) / dt
            if bool(hopper.USE_Q_TRACKING) and q_ref.ndim == 2 and t + 1 < len(q_ref):
                q_ref_now = q_ref[t + 1]
                q_ref_prev = q_ref[t]
                v_ref = (q_ref_now - q_ref_prev) / dt
                du0 = float(hopper.KP_THETA) * (float(q_ref_now[2]) - float(q_now[2])) + float(hopper.KD_THETA) * (
                    float(v_ref[2]) - float(v_now[2])
                )
                du1 = float(hopper.KP_R) * (float(q_ref_now[3]) - float(q_now[3])) + float(hopper.KD_R) * (
                    float(v_ref[3]) - float(v_now[3])
                )
                u_nom[0] += float(np.clip(du0, -float(hopper.DU0_TRACK_LIMIT), float(hopper.DU0_TRACK_LIMIT)))
                u_nom[1] += float(np.clip(du1, -float(hopper.DU1_TRACK_LIMIT), float(hopper.DU1_TRACK_LIMIT)))

            in_contact = float(self.model.signed_distance(sim.traj.q[t + 1])[0]) <= float(hopper.CONTACT_THRESH)
            if in_contact and u_nom[1] > 0.0:
                u_nom[1] *= float(hopper.U2_CONTACT_SCALE)
            if bool(hopper.THETA_FLIGHT_BIAS_ENABLE) and (not in_contact):
                u_nom[0] += float(hopper.THETA_FLIGHT_BIAS)
            u_nom = np.clip(u_nom, -100.0, 100.0)

            u_cmd, cbf_info = hopper.cbf_filter(
                sim,
                t,
                u_nom,
                use_robust_cbf=bool(self.use_robust_cbf),
                robust_state=robust_state,
            )
            if not bool(cbf_info.get("qp_solved", False)):
                failed = True
            h_pred.append(float(cbf_info.get("h_next_pred", np.nan)))
            u_dev.append(float(np.linalg.norm(np.asarray(u_cmd, dtype=float) - u_nom)))

            sim.traj.u[t][:] = u_cmd
            sim.traj.w[t][:] = np.zeros(self.model.nw, dtype=float)
            if not sim.step(t, sim.diff_sol):
                failed = True
                break

            gamma_actual = float(np.asarray(sim.traj.gamma[t], dtype=float).reshape(-1)[0])
            h_true.append(float(self.f_max - gamma_actual))
            if float(sim.traj.q[t + 2][1]) <= 0.0:
                failed = True
                break

        return EvaluatorOutput(
            h_pred=np.asarray(h_pred[: len(h_true)], dtype=float),
            h_true=np.asarray(h_true, dtype=float),
            control_deviation=np.asarray(u_dev[: len(h_true)], dtype=float),
            failed=bool(failed),
            info={"tag": scenario.get("tag", ""), "source": scenario.get("source", ""), "start": scenario.get("start", -1)},
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Boundary-focused hopper kappa compatibility screening.")
    add_kappa_grid_args(ap, kappa_min=1e-6, kappa_max=1e-3, num_kappas=10)
    ap.add_argument("--num-scenarios", type=int, default=40)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--f-max", type=float, default=1.3)
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--rho-q", type=float, default=0.02)
    ap.add_argument("--rho-v", type=float, default=0.05)
    ap.add_argument("--u-noise", type=float, default=0.05)
    ap.add_argument("--use-robust-cbf", action="store_true")
    ap.add_argument(
        "--scenario-set",
        type=str,
        default="both",
        choices=["both", "after", "before"],
        help="Reference set used to draw boundary-focused perturbed windows.",
    )
    add_risk_args(ap, q=0.1, beta=0.1)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/kappa_selection/hopper"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    kappas = candidate_kappas(args)
    scenario_model = _build_hopper_model()
    scenarios = build_hopper_boundary_scenarios(
        model=scenario_model,
        scenario_set=str(args.scenario_set),
        num_scenarios=int(args.num_scenarios),
        horizon=int(args.horizon),
        seed=int(args.seed),
        rho_q=float(args.rho_q),
        rho_v=float(args.rho_v),
        u_noise=float(args.u_noise),
    )
    config = RolloutConfig(eval_horizon=int(args.horizon))
    evaluator = HopperPerturbedCBFEvaluator(
        alpha=float(args.alpha),
        f_max=float(args.f_max),
        use_robust_cbf=bool(args.use_robust_cbf),
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
            "task": "hopper",
            "alpha": float(args.alpha),
            "f_max": float(args.f_max),
            "scenario_set": str(args.scenario_set),
            "seed": int(args.seed),
            "rho_q": float(args.rho_q),
            "rho_v": float(args.rho_v),
            "u_noise": float(args.u_noise),
            "use_robust_cbf": bool(args.use_robust_cbf),
        },
    )
    print_risk_summary(summary, title="Hopper compatibility screening")
    write_outputs(args.outdir, summary, rollout_rows)


if __name__ == "__main__":
    main()
