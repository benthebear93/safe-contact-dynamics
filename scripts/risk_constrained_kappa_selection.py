#!/usr/bin/env python3
"""Shared boundary-focused kappa compatibility screening utilities.

Candidates are screened for model-controller compatibility near the safety
boundary using standard-CBF rollouts:

    S_q(kappa) = Q_q(h_pred) - Q_{1-beta}((-Delta h)^+),

where Delta h = h_true - h_pred.  The violation rate is reported as the
standard-CBF residual violation rate, but it is not used as a hard screening
gate because the robust-CBF tightening is precisely meant to compensate for
this residual under-prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


@dataclass
class RolloutConfig:
    """Settings consumed by the short CBF rollout evaluators."""

    eval_horizon: int = 80


@dataclass
class EvaluatorOutput:
    """Output for one short closed-loop rollout under a given kappa."""

    h_pred: np.ndarray
    h_true: np.ndarray
    control_deviation: np.ndarray
    failed: bool
    info: dict[str, Any] = field(default_factory=dict)


class ScreeningEvaluator(Protocol):
    def evaluate(self, kappa: float, scenario: Mapping[str, Any], config: RolloutConfig) -> EvaluatorOutput:
        ...


@dataclass
class RolloutScreeningResult:
    kappa: float
    scenario_tag: str
    seed: int
    failed: bool
    violated: bool
    n_step_samples: int
    min_h_true: float
    min_h_pred: float
    min_delta_h: float
    max_under_error: float
    mean_control_deviation: float
    max_control_deviation: float


@dataclass
class _RolloutEvaluation:
    row: RolloutScreeningResult
    h_pred: np.ndarray
    h_true: np.ndarray
    control_deviation: np.ndarray


@dataclass
class ScreeningCandidateResult:
    kappa: float
    num_rollouts: int
    num_failed_rollouts: int
    num_violated_rollouts: int
    num_step_samples: int
    failure_rate: float
    violation_rate: float
    h_pred_lower_quantile: float
    under_error_quantile: float
    margin_error_score: float
    mean_control_deviation: float
    max_control_deviation: float
    min_h_true: float
    compatible: bool


@dataclass
class ScreeningSummary:
    config: dict[str, Any]
    candidate_kappas: list[float]
    results_by_kappa: list[ScreeningCandidateResult]
    selected_kappa: float
    selected_delta_kappa: float
    selected_from_compatible_set: bool
    fallback_best_kappa: float
    compatible_kappas: list[float]
    near_best_kappas: list[float]


def add_kappa_grid_args(
    ap: argparse.ArgumentParser,
    *,
    kappa_min: float,
    kappa_max: float,
    num_kappas: int,
) -> None:
    ap.add_argument("--kappa-min", type=float, default=float(kappa_min))
    ap.add_argument("--kappa-max", type=float, default=float(kappa_max))
    ap.add_argument("--num-kappas", type=int, default=int(num_kappas))
    ap.add_argument(
        "--kappas",
        type=str,
        default="",
        help="Optional comma-separated kappa list. Overrides min/max/num.",
    )


def add_screening_args(
    ap: argparse.ArgumentParser,
    *,
    q: float = 0.1,
    beta: float = 0.1,
    score_tol: float = 1e-12,
    score_select_tol: float = 1e-9,
    kappa_floor: float = 0.0,
) -> None:
    ap.add_argument("--q", type=float, default=float(q), help="Lower quantile for predicted margin Q_q(h_pred).")
    ap.add_argument(
        "--beta",
        type=float,
        default=float(beta),
        help="Tail level for under-prediction error: rho=Q_{1-beta}((-Delta h)^+).",
    )
    ap.add_argument(
        "--score-tol",
        type=float,
        default=float(score_tol),
        help="Numerical tolerance for compatibility score: require S_q >= -score_tol.",
    )
    ap.add_argument(
        "--score-select-tol",
        type=float,
        default=float(score_select_tol),
        help="Absolute tolerance for treating near-best S_q candidates as equivalent during selection.",
    )
    ap.add_argument(
        "--kappa-floor",
        type=float,
        default=float(kappa_floor),
        help="Smallest numerically acceptable kappa for final smallest-kappa tie breaking.",
    )
    ap.add_argument(
        "--violation-tol",
        type=float,
        default=0.0,
        help="Treat a rollout as violating when min(h_true) < -violation_tol.",
    )
    ap.add_argument(
        "--failure-control-penalty",
        type=float,
        default=1e3,
        help="Control-deviation cost assigned to failed/empty rollouts.",
    )


def add_risk_args(
    ap: argparse.ArgumentParser,
    *,
    epsilon_vio: float = 0.2,
    epsilon_fail: float = 0.2,
    eta: float = 0.05,
    q: float = 0.1,
    beta: float = 0.1,
    score_tol: float = 1e-12,
    score_select_tol: float = 1e-9,
    kappa_floor: float = 0.0,
) -> None:
    """Compatibility wrapper; risk-certificate args were removed by the rewrite."""

    _ = (epsilon_vio, epsilon_fail, eta)
    add_screening_args(
        ap,
        q=float(q),
        beta=float(beta),
        score_tol=float(score_tol),
        score_select_tol=float(score_select_tol),
        kappa_floor=float(kappa_floor),
    )


def candidate_kappas(args: argparse.Namespace) -> list[float]:
    if str(args.kappas).strip():
        kappas = [float(x) for x in str(args.kappas).split(",") if x.strip()]
    else:
        kappas = np.geomspace(
            float(args.kappa_min),
            float(args.kappa_max),
            int(args.num_kappas),
        ).tolist()
    if not kappas:
        raise ValueError("At least one kappa candidate is required.")
    if any(k <= 0.0 or not np.isfinite(k) for k in kappas):
        raise ValueError(f"All kappa candidates must be finite and positive: {kappas}")
    return [float(k) for k in kappas]


def _safe_quantile(values: Sequence[float] | np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, float(q)))


def _finite_array(values: Sequence[float] | np.ndarray, *, fallback: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([float(fallback)], dtype=float)
    return arr


def evaluate_rollout(
    *,
    kappa: float,
    scenario: Mapping[str, Any],
    evaluator: ScreeningEvaluator,
    config: Any,
    violation_tol: float,
    failure_control_penalty: float,
) -> _RolloutEvaluation:
    out = evaluator.evaluate(float(kappa), scenario, config)
    h_true_raw = np.asarray(out.h_true, dtype=float).reshape(-1)
    h_pred_raw = np.asarray(out.h_pred, dtype=float).reshape(-1)
    n = min(h_true_raw.size, h_pred_raw.size)
    if n > 0:
        h_true_pair = h_true_raw[:n]
        h_pred_pair = h_pred_raw[:n]
        mask = np.isfinite(h_true_pair) & np.isfinite(h_pred_pair)
        h_true = h_true_pair[mask]
        h_pred = h_pred_pair[mask]
    else:
        h_true = np.zeros(0, dtype=float)
        h_pred = np.zeros(0, dtype=float)

    u_dev = np.asarray(out.control_deviation, dtype=float).reshape(-1)
    failed = bool(out.failed or h_true.size == 0 or h_pred.size == 0)
    violated = bool(h_true.size > 0 and np.min(h_true) < -float(violation_tol))
    delta_h = h_true - h_pred if h_true.size and h_pred.size else np.zeros(0, dtype=float)
    under_error = np.maximum(0.0, -delta_h)

    if failed:
        u_dev_finite = np.array([float(failure_control_penalty)], dtype=float)
    else:
        u_dev_finite = _finite_array(u_dev, fallback=float(failure_control_penalty))

    row = RolloutScreeningResult(
        kappa=float(kappa),
        scenario_tag=str(scenario.get("tag", "")),
        seed=int(scenario.get("seed", -1)),
        failed=failed,
        violated=violated,
        n_step_samples=int(h_true.size),
        min_h_true=float(np.min(h_true)) if h_true.size else float("nan"),
        min_h_pred=float(np.min(h_pred)) if h_pred.size else float("nan"),
        min_delta_h=float(np.min(delta_h)) if delta_h.size else float("nan"),
        max_under_error=float(np.max(under_error)) if under_error.size else float("nan"),
        mean_control_deviation=float(np.mean(u_dev_finite)),
        max_control_deviation=float(np.max(u_dev_finite)),
    )
    return _RolloutEvaluation(row=row, h_pred=h_pred, h_true=h_true, control_deviation=u_dev_finite)


def summarize_candidate(
    *,
    kappa: float,
    evaluations: Sequence[_RolloutEvaluation],
    q: float,
    beta: float,
    score_tol: float,
) -> ScreeningCandidateResult:
    n_rollouts = len(evaluations)
    n_fail = int(sum(ev.row.failed for ev in evaluations))
    n_vio = int(sum(ev.row.violated for ev in evaluations))
    h_pred_all = np.concatenate([ev.h_pred for ev in evaluations if ev.h_pred.size]) if evaluations else np.zeros(0)
    h_true_all = np.concatenate([ev.h_true for ev in evaluations if ev.h_true.size]) if evaluations else np.zeros(0)
    u_all = np.concatenate([ev.control_deviation for ev in evaluations if ev.control_deviation.size]) if evaluations else np.zeros(0)
    n_samples = min(h_pred_all.size, h_true_all.size)
    if n_samples > 0:
        delta_h = h_true_all[:n_samples] - h_pred_all[:n_samples]
        under_error = np.maximum(0.0, -delta_h)
    else:
        under_error = np.zeros(0, dtype=float)

    h_pred_q = _safe_quantile(h_pred_all, float(q))
    rho = _safe_quantile(under_error, 1.0 - float(beta))
    score = float(h_pred_q - rho) if np.isfinite(h_pred_q) and np.isfinite(rho) else float("-inf")
    mean_u = float(np.mean(u_all)) if u_all.size else float("inf")
    max_u = float(np.max(u_all)) if u_all.size else float("inf")
    min_h = float(np.min(h_true_all)) if h_true_all.size else float("nan")
    # Standard-CBF violations are expected when Delta h is under-predicted.
    # We therefore screen the model-controller pair by feasibility and the
    # conservative margin-error score, while reporting r_vio separately.
    compatible = bool(n_fail == 0 and score >= -float(score_tol))

    return ScreeningCandidateResult(
        kappa=float(kappa),
        num_rollouts=n_rollouts,
        num_failed_rollouts=n_fail,
        num_violated_rollouts=n_vio,
        num_step_samples=int(n_samples),
        failure_rate=float(n_fail / n_rollouts) if n_rollouts else 1.0,
        violation_rate=float(n_vio / n_rollouts) if n_rollouts else 1.0,
        h_pred_lower_quantile=float(h_pred_q),
        under_error_quantile=float(rho),
        margin_error_score=float(score),
        mean_control_deviation=mean_u,
        max_control_deviation=max_u,
        min_h_true=min_h,
        compatible=compatible,
    )


def select_compatibility_screening(
    kappas: Sequence[float],
    scenarios: Sequence[Mapping[str, Any]],
    *,
    evaluator: ScreeningEvaluator,
    config: Any,
    q: float,
    beta: float,
    score_tol: float = 1e-12,
    score_select_tol: float = 1e-9,
    kappa_floor: float = 0.0,
    violation_tol: float,
    failure_control_penalty: float,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[ScreeningSummary, list[RolloutScreeningResult]]:
    if len(kappas) == 0:
        raise ValueError("No kappa candidates provided.")
    if len(scenarios) == 0:
        raise ValueError("No scenarios provided.")

    rollout_rows: list[RolloutScreeningResult] = []
    results: list[ScreeningCandidateResult] = []

    for idx, kappa in enumerate(kappas, start=1):
        print(f"[eval] {idx}/{len(kappas)} kappa={kappa:.3e}")
        evaluations = [
            evaluate_rollout(
                kappa=float(kappa),
                scenario=scenario,
                evaluator=evaluator,
                config=config,
                violation_tol=float(violation_tol),
                failure_control_penalty=float(failure_control_penalty),
            )
            for scenario in scenarios
        ]
        rollout_rows.extend(ev.row for ev in evaluations)
        result = summarize_candidate(
            kappa=float(kappa),
            evaluations=evaluations,
            q=float(q),
            beta=float(beta),
            score_tol=float(score_tol),
        )
        results.append(result)
        print(
            "[screen] "
            f"kappa={kappa:.3e} "
            f"vio={result.num_violated_rollouts}/{result.num_rollouts} "
            f"fail={result.num_failed_rollouts}/{result.num_rollouts} "
            f"hq={result.h_pred_lower_quantile:.3e} "
            f"rho={result.under_error_quantile:.3e} "
            f"S={result.margin_error_score:.3e} "
            f"mean_u={result.mean_control_deviation:.3e} "
            f"comp={int(result.compatible)}"
        )

    compatible_rows = [r for r in results if r.compatible and r.kappa >= float(kappa_floor)]
    if not compatible_rows:
        compatible_rows = [r for r in results if r.compatible]
    fallback = min(
        results,
        key=lambda r: (
            r.failure_rate,
            -r.margin_error_score,
            r.kappa,
        ),
    )
    if compatible_rows:
        best_score = max(r.margin_error_score for r in compatible_rows)
        near_best_rows = [
            r for r in compatible_rows if r.margin_error_score >= best_score - float(score_select_tol)
        ]
        selected_row = min(near_best_rows, key=lambda r: r.kappa)
        compatible_kappas = [float(r.kappa) for r in sorted(compatible_rows, key=lambda r: r.kappa)]
        near_best_kappas = [float(r.kappa) for r in sorted(near_best_rows, key=lambda r: r.kappa)]
        selected_from_compatible = True
    else:
        selected_row = fallback
        compatible_kappas = []
        near_best_kappas = []
        selected_from_compatible = False

    config_dict = {
        "q": float(q),
        "beta": float(beta),
        "score_tol": float(score_tol),
        "score_select_tol": float(score_select_tol),
        "kappa_floor": float(kappa_floor),
        "num_scenarios": int(len(scenarios)),
        "eval_horizon": int(getattr(config, "eval_horizon", 0)),
        "violation_tol": float(violation_tol),
        "failure_control_penalty": float(failure_control_penalty),
        "selection_rule": (
            "K_comp={r_fail=0,S_q>=-score_tol}; report r_vio as the standard-CBF "
            "residual violation rate; choose candidates within score_select_tol "
            "of the best S_q and select the smallest kappa; otherwise select by "
            "(r_fail,-S_q,kappa)"
        ),
        "score_definition": "S_q=Q_q(h_pred)-Q_{1-beta}((-Delta h)^+), Delta h=h_true-h_pred",
        "delta_kappa_definition": "delta_kappa=Q_{1-beta}((-Delta h)^+) for the selected kappa",
    }
    if metadata:
        config_dict.update(dict(metadata))

    summary = ScreeningSummary(
        config=config_dict,
        candidate_kappas=[float(k) for k in kappas],
        results_by_kappa=results,
        selected_kappa=float(selected_row.kappa),
        selected_delta_kappa=float(selected_row.under_error_quantile),
        selected_from_compatible_set=selected_from_compatible,
        fallback_best_kappa=float(fallback.kappa),
        compatible_kappas=compatible_kappas,
        near_best_kappas=near_best_kappas,
    )
    return summary, rollout_rows


def select_risk_constrained(
    kappas: Sequence[float],
    scenarios: Sequence[Mapping[str, Any]],
    *,
    evaluator: ScreeningEvaluator,
    config: Any,
    epsilon_vio: float | None = None,
    epsilon_fail: float | None = None,
    eta: float | None = None,
    q: float = 0.1,
    beta: float = 0.1,
    score_tol: float = 1e-12,
    score_select_tol: float = 1e-9,
    kappa_floor: float = 0.0,
    violation_tol: float,
    failure_control_penalty: float,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[ScreeningSummary, list[RolloutScreeningResult]]:
    """Compatibility wrapper for the previous risk-constrained API."""

    _ = (epsilon_vio, epsilon_fail, eta)
    return select_compatibility_screening(
        kappas,
        scenarios,
        evaluator=evaluator,
        config=config,
        q=float(q),
        beta=float(beta),
        score_tol=float(score_tol),
        score_select_tol=float(score_select_tol),
        kappa_floor=float(kappa_floor),
        violation_tol=float(violation_tol),
        failure_control_penalty=float(failure_control_penalty),
        metadata=metadata,
    )


def write_csv(path: Path, rows: Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dict_rows = [asdict(r) for r in rows]
    fieldnames = list(dict_rows[0].keys()) if dict_rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        if not fieldnames:
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict_rows)


def write_outputs(outdir: Path, summary: ScreeningSummary, rollout_rows: Sequence[RolloutScreeningResult]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    (outdir / "selected.txt").write_text(f"{summary.selected_kappa:.17g}\n", encoding="utf-8")
    write_csv(outdir / "screening_table.csv", summary.results_by_kappa)
    write_csv(outdir / "rollouts.csv", rollout_rows)
    print(f"[write] {outdir / 'summary.json'}")
    print(f"[write] {outdir / 'screening_table.csv'}")
    print(f"[write] {outdir / 'rollouts.csv'}")


def print_screening_summary(summary: ScreeningSummary, *, title: str = "Boundary-focused kappa screening") -> None:
    print(f"\n[{title}]")
    print("kappa       vio_rate fail_rate  Qq(h_pred)  delta_kappa      S_q     mean||u-u_nom||  comp")
    for r in sorted(summary.results_by_kappa, key=lambda x: x.kappa):
        print(
            f"{r.kappa:9.1e} "
            f"{r.violation_rate:8.3f} {r.failure_rate:9.3f} "
            f"{r.h_pred_lower_quantile:11.3e} {r.under_error_quantile:11.3e} "
            f"{r.margin_error_score:11.3e} {r.mean_control_deviation:16.3e} {int(r.compatible):5d}"
        )
    if summary.selected_from_compatible_set:
        print(f"[selected-compatible] {summary.selected_kappa}")
        print(f"[selected-delta-kappa] {summary.selected_delta_kappa}")
        print(f"[compatible-kappas] {summary.compatible_kappas}")
        print(f"[near-best-kappas] {summary.near_best_kappas}")
    else:
        print(f"[selected-fallback:max_Sq] {summary.selected_kappa}")
        print(f"[selected-delta-kappa] {summary.selected_delta_kappa}")


# Backward-compatible name for scripts created before the algorithm rewrite.
print_risk_summary = print_screening_summary
