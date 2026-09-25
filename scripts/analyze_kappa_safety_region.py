#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "results" / "kappa_safety_region.json"
ALL_TASKS = ["particle", "planar_push", "tipover_box_pivot", "hopper"]


@dataclass
class TaskRow:
    kappa: float
    n_steps: int
    any_violation: bool
    n_viol: int
    max_gamma: float
    f_max: float
    min_safety_margin: float
    task_success: bool
    overall_success: bool
    notes: str = ""


def kappa_grid(kappa_min: float, kappa_max: float, n: int) -> np.ndarray:
    return np.logspace(np.log10(float(kappa_min)), np.log10(float(kappa_max)), int(n))


def evaluate_particle(kappa: float, seed: int = 0, alpha: float = 0.2) -> TaskRow:
    import example.particle_p_control_ref_cbf_kappa as particle

    f_max = 0.25
    data = particle._run_mode(
        "cbf",
        kappa=float(kappa),
        alpha=float(alpha),
        f_max=f_max,
        seed=int(seed),
        robust_h_scale=1e-3,
        robust_c_nom=0.02,
    )
    gamma = np.asarray(data["gamma_actual"], dtype=float).reshape(-1)
    n_viol = int(np.sum(gamma > f_max))
    any_viol = bool(n_viol > 0)
    max_gamma = float(np.max(gamma)) if gamma.size else float("nan")
    min_margin = float(f_max - max_gamma) if np.isfinite(max_gamma) else float("nan")
    # No independent task-completion criterion is defined for this case.
    task_success = bool(not any_viol)
    return TaskRow(
        kappa=float(kappa),
        n_steps=int(gamma.size),
        any_violation=any_viol,
        n_viol=n_viol,
        max_gamma=max_gamma,
        f_max=float(f_max),
        min_safety_margin=min_margin,
        task_success=task_success,
        overall_success=bool((not any_viol) and task_success),
    )


def ensure_fixed_planar_reference(path: Path) -> None:
    import example.planar_push_linear_cimpc_cbf_kappa as planar

    if path.exists():
        return

    model = planar.build_pusher(mu_body=0.5)
    r, rz, rtheta = planar.make_pusher_residual(model)
    q7_ref, u_ref = planar.generate_push_reference(
        model=model,
        residuals=(r, rz, rtheta),
        h=0.05,
        kappa_tol=1e-4,
        kappa_accept_ratio=2.0,
        steps=50,
        goal_x=0.4,
        goal_y=0.4,
        approach_gain=1.0,
        push_gain=0.25,
        x_push_min=0.22,
        y_push_min=0.01,
        y_early_ratio=0.1,
        y_early_gain=0.0,
        x_taper_start_ratio=0.5,
        x_taper_end_scale=0.2,
        u_min_mag=0.12,
        u_max=0.30,
        contact_margin=0.01,
        phase1_ratio=0.2,
        lift_ratio=0.01,
        y_base=0.0,
        rot_offset_max=0.03,
    )
    payload = {"h": 0.05, "q": q7_ref.tolist(), "u": u_ref.tolist(), "meta": {"fixed_ref": True}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def evaluate_planar(
    kappa: float, idx: int, fixed_ref_path: Path, f_max: float, alpha: float
) -> TaskRow:
    import example.planar_push_linear_cimpc_cbf_kappa as planar

    tmp_dir = Path("/tmp/kappa_safety_region/planar")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_u_seq = tmp_dir / f"planar_u_seq_{idx:02d}.json"

    cfg_backup = dict(planar.CFG)
    try:
        planar.CFG["ref_mode"] = "push"
        planar.CFG["generated_ref_path"] = fixed_ref_path
        planar.CFG["force_ref_regen"] = False
        planar.CFG["run_cimpc"] = True
        planar.CFG["replay_ref_u"] = False
        planar.CFG["kappa"] = float(kappa)
        planar.CFG["use_cbf"] = True
        planar.CFG["use_rcbf"] = False
        planar.CFG["cbf_alpha"] = float(alpha)
        planar.CFG["f_max"] = float(f_max)
        planar.CFG["out_u_seq"] = out_u_seq
        planar.main()
    finally:
        planar.CFG.clear()
        planar.CFG.update(cfg_backup)

    payload = json.loads(out_u_seq.read_text())
    gamma = np.asarray(payload.get("gamma_safe", []), dtype=float).reshape(-1)
    f_max = float(payload.get("f_max", np.nan))
    n_viol = int(np.sum(gamma > f_max)) if np.isfinite(f_max) else 0
    any_viol = bool(n_viol > 0)
    max_gamma = float(np.max(gamma)) if gamma.size else float("nan")
    min_margin = float(f_max - max_gamma) if np.isfinite(max_gamma) else float("nan")
    task_success = bool(payload.get("fail_t", None) is None)
    return TaskRow(
        kappa=float(kappa),
        n_steps=int(gamma.size),
        any_violation=any_viol,
        n_viol=n_viol,
        max_gamma=max_gamma,
        f_max=f_max,
        min_safety_margin=min_margin,
        task_success=task_success,
        overall_success=bool((not any_viol) and task_success),
        notes=f"fail_t={payload.get('fail_t', None)}",
    )


def evaluate_tipover(kappa: float, alpha: float, f_max: float = 0.9) -> TaskRow:
    import example.tipover_push_ref_cbf_kappa as tipover

    ref = tipover.load_ref_traj(Path("reference_trajectory/tipover_ref_traj.json"))
    q_ref = np.asarray(ref["q"], dtype=float)
    u_ref = np.asarray(ref["u"], dtype=float)
    h = float(ref["h"])
    t_horizon = int(u_ref.shape[0])
    q0 = q_ref[0].copy()
    v0 = (q_ref[1] - q_ref[0]) / h if q_ref.shape[0] >= 2 else np.zeros(7)
    model = tipover.build_tipover_model(q0, ref)
    model.set_nominal_configuration(q0)
    residuals = tipover.make_tipover_push_residual(model, use_cpp_jac=True)
    theta_ref = q_ref[:t_horizon, 3].copy()

    params = {
        "model": model,
        "residuals": residuals,
        "dt": h,
        "num_steps": t_horizon,
        "kappa_tol": float(kappa),
        "u_ref": u_ref,
        "theta_ref": theta_ref,
        "q0": q0,
        "v0": v0,
        "u_min": np.array([-6.0, -6.0], dtype=float),
        "u_max": np.array([6.0, 6.0], dtype=float),
        "F_max": float(f_max),
        "cbf_alpha": float(alpha),
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
    result = tipover.run_sim(params, mode="cbf")
    max_gamma = float(result.get("max_force", float("nan")))
    f_max = float(params["F_max"])
    n_viol = int(result.get("n_viol", 0))
    any_viol = bool(n_viol > 0)
    min_margin = float(f_max - max_gamma) if np.isfinite(max_gamma) else float("nan")
    task_success = bool(result.get("success_tipover", False))
    return TaskRow(
        kappa=float(kappa),
        n_steps=int(len(result.get("gamma_corner0", []))),
        any_violation=any_viol,
        n_viol=n_viol,
        max_gamma=max_gamma,
        f_max=f_max,
        min_safety_margin=min_margin,
        task_success=task_success,
        overall_success=bool((not any_viol) and task_success),
        notes=f"tipover={task_success}",
    )


def evaluate_hopper(
    kappa: float, idx: int, alpha: float, f_max: float | None = None
) -> TaskRow:
    import example.hopper_simulate_test_before_cbf as hopper

    hopper.OUT_DIR = Path("/tmp/kappa_safety_region/hopper")
    hopper.OUT_DIR.mkdir(parents=True, exist_ok=True)
    hopper.KAPPA_TOL = float(kappa)
    hopper.CBF_ALPHA = float(alpha)
    if f_max is not None:
        hopper.F_MAX = float(f_max)

    model = hopper.Hopper(
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
    residual, jacobian_z, jacobian_theta = hopper.make_hopper_residual(model)
    result = hopper.simulate_from_trajectory(
        model,
        residual,
        jacobian_z,
        jacobian_theta,
        hopper.TRAJ_PATH_AFTER,
        f"kappa_region_{idx:02d}",
        enable_cbf=True,
        use_robust_cbf=False,
    )
    gamma = np.asarray(result.get("gamma", []), dtype=float).reshape(-1)
    f_max = float(hopper.F_MAX)
    n_viol = int(np.sum(gamma > f_max))
    any_viol = bool(n_viol > 0)
    max_gamma = float(np.max(gamma)) if gamma.size else float("nan")
    min_margin = float(f_max - max_gamma) if np.isfinite(max_gamma) else float("nan")
    task_success = bool(not result.get("fall", True))
    overall = bool((not any_viol) and task_success and bool(result.get("status", False)))
    note = (
        f"fall={bool(result.get('fall', True))}, "
        f"status={bool(result.get('status', False))}"
    )
    return TaskRow(
        kappa=float(kappa),
        n_steps=int(gamma.size),
        any_violation=any_viol,
        n_viol=n_viol,
        max_gamma=max_gamma,
        f_max=f_max,
        min_safety_margin=min_margin,
        task_success=task_success,
        overall_success=overall,
        notes=note,
    )


def write_outputs(
    rows_by_task: Dict[str, List[TaskRow]],
    tipover_alpha: float,
    hopper_alpha: float,
    planar_fmax: float,
    planar_alpha: float,
    kappas: np.ndarray,
    out_json: Path = OUT_JSON,
    particle_alpha: float = 0.2,
) -> None:
    out_payload = {
        "meta": {
            "kappa_min": float(kappas[0]),
            "kappa_max": float(kappas[-1]),
            "num_kappa": int(len(kappas)),
            "particle_alpha": float(particle_alpha),
            "tipover_alpha": float(tipover_alpha),
            "hopper_alpha": float(hopper_alpha),
            "planar_fmax": float(planar_fmax),
            "planar_alpha": float(planar_alpha),
            "safety_definition": "any_violation -> fail",
        },
        "tasks": {k: [asdict(r) for r in v] for k, v in rows_by_task.items()},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_payload, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa-min", type=float, default=1e-6)
    ap.add_argument("--kappa-max", type=float, default=1e-3)
    ap.add_argument("--num-kappa", type=int, default=10)
    ap.add_argument("--particle-alpha", type=float, default=0.2)
    ap.add_argument("--tipover-alpha", type=float, default=0.9)
    ap.add_argument("--tipover-fmax", type=float, default=0.9)
    ap.add_argument("--hopper-alpha", type=float, default=0.9)
    ap.add_argument("--hopper-fmax", type=float, default=None)
    ap.add_argument("--planar-fmax", type=float, default=0.245)
    ap.add_argument("--planar-alpha", type=float, default=0.7)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument(
        "--only",
        type=str,
        default="all",
        choices=("all", *ALL_TASKS),
    )
    args = ap.parse_args()

    kappas = kappa_grid(args.kappa_min, args.kappa_max, args.num_kappa)

    fixed_ref = Path("/tmp/kappa_safety_region/planar/fixed_planar_ref.json")
    if args.only in ("all", "planar_push"):
        ensure_fixed_planar_reference(fixed_ref)

    rows_by_task: Dict[str, List[TaskRow]] = {k: [] for k in ALL_TASKS}

    for i, kappa in enumerate(kappas):
        print(f"[scan] {i+1}/{len(kappas)} kappa={kappa:.6e}")
        if args.only in ("all", "particle"):
            rows_by_task["particle"].append(
                evaluate_particle(float(kappa), seed=0, alpha=float(args.particle_alpha))
            )
        if args.only in ("all", "planar_push"):
            rows_by_task["planar_push"].append(
                evaluate_planar(
                    float(kappa),
                    i,
                    fixed_ref,
                    f_max=float(args.planar_fmax),
                    alpha=float(args.planar_alpha),
                )
            )
        if args.only in ("all", "tipover_box_pivot"):
            rows_by_task["tipover_box_pivot"].append(
                evaluate_tipover(
                    float(kappa),
                    alpha=float(args.tipover_alpha),
                    f_max=float(args.tipover_fmax),
                )
            )
        if args.only in ("all", "hopper"):
            rows_by_task["hopper"].append(
                evaluate_hopper(
                    float(kappa),
                    i,
                    alpha=float(args.hopper_alpha),
                    f_max=args.hopper_fmax,
                )
            )

    write_outputs(
        rows_by_task,
        float(args.tipover_alpha),
        float(args.hopper_alpha),
        float(args.planar_fmax),
        float(args.planar_alpha),
        kappas,
        out_json=args.out_json,
        particle_alpha=float(args.particle_alpha),
    )
    print(f"[write] {args.out_json}")


if __name__ == "__main__":
    main()
