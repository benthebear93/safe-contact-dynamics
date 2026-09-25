"""Seven-coordinate planar screening using the paper comparison's CBF rollout."""

from pathlib import Path

import numpy as np

from example import planar_push_linear_cimpc_cbf_fixed_delta as planar
from scripts.risk_constrained_kappa_selection import EvaluatorOutput


def build_planar_boundary_scenarios(
    q_ref, u_ref, dt, *, num_scenarios=12, horizon=12, seed_base=200
):
    """Keep the legacy perturbations, mapping yaw to coordinate 3 of q7.

    The six deterministic patterns repeat when more than six rollouts are
    requested. Seeds are labels, as in the original screening adapter.
    """
    q_ref = np.asarray(q_ref, dtype=float)
    u_ref = np.asarray(u_ref, dtype=float)
    if q_ref.ndim != 2 or q_ref.shape[1] != 7 or len(q_ref) == 0:
        raise ValueError(f"Expected nonempty q shape (N,7), got {q_ref.shape}")
    if u_ref.ndim != 2 or u_ref.shape[1] != 2 or len(u_ref) == 0:
        raise ValueError(f"Expected nonempty u shape (N,2), got {u_ref.shape}")
    if dt <= 0 or num_scenarios <= 0 or horizon <= 0:
        raise ValueError("dt, num_scenarios and horizon must be positive")
    q0_base = q_ref[0].copy()
    v0_base = (q_ref[1] - q_ref[0]) / dt if len(q_ref) >= 2 else np.zeros(7)
    u_nom = u_ref[:horizon]
    scales = [1.0, 1.08, 1.15, 1.22, 1.05, 1.12]
    dx_vals = [-0.008, -0.004, 0.0, 0.003, 0.006, -0.002]
    dy_vals = [0.0, -0.003, 0.004, -0.002, 0.002, 0.005]
    th_vals = [0.0, -0.03, 0.03, -0.015, 0.015, 0.04]
    vp_vals = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02]
    scenarios = []
    for i in range(num_scenarios):
        q0, v0 = q0_base.copy(), v0_base.copy()
        q0[0] += dx_vals[i % len(dx_vals)]
        q0[1] += dy_vals[i % len(dy_vals)]
        q0[3] += th_vals[i % len(th_vals)]
        v0[0] += vp_vals[i % len(vp_vals)]
        scenarios.append(
            {
                "seed": int(seed_base + i),
                "q0": q0.tolist(),
                "v0": v0.tolist(),
                "u_ref": (scales[i % len(scales)] * u_nom).tolist(),
                "tag": f"planar_boundary_{i}",
            }
        )
    return scenarios


class PlanarPushCBFEvaluator:
    def __init__(self, *, ref_path: Path, f_max=0.245, alpha=0.95):
        self.q_ref, self.u_ref, self.dt = planar.base.load_q7_ref(ref_path)
        self.q0_ref = self.q_ref[0].copy()
        self.params = {
            "f_max": float(f_max),
            "cbf_alpha": float(alpha),
            "use_rcbf": False,
            "delta_kappa": 0.0,
            "contact_index": 4,
            "qp_weight": np.array([1.0, 1.0]),
            "u_min": np.array([-0.5, -0.5]),
            "u_max": np.array([0.5, 0.5]),
        }
        self.rollout_records = []

    def evaluate(self, kappa, scenario, config):
        u_ref = np.asarray(scenario.get("u_ref", self.u_ref), dtype=float)
        steps = min(int(config.eval_horizon), len(u_ref))
        if steps <= 0:
            raise ValueError("Screening requires at least one control step")
        u_ref = u_ref[:steps]
        q0 = np.asarray(scenario.get("q0", self.q0_ref), dtype=float)
        v0 = np.asarray(scenario.get("v0", np.zeros(7)), dtype=float)
        result = planar.run_rollout(
            u_ref,
            q0,
            self.dt,
            float(kappa),
            2.0,
            cbf_params=self.params,
            v0=v0,
        )
        # These are F_max - gamma at the applied control, with no subtraction
        # of the current-state CBF right-hand side from either margin.
        h_pred = np.asarray([row["h_pred"] for row in result["rows"]], dtype=float)
        h_true = np.asarray([row["h_actual"] for row in result["rows"]], dtype=float)
        n = len(h_true)
        u_dev = np.linalg.norm(result["u"] - u_ref[:n], axis=1) if n else np.zeros(0)
        failed = n != steps or not np.all(np.isfinite(h_pred) & np.isfinite(h_true))
        info = {
            "tag": scenario.get("tag", ""),
            "seed": scenario.get("seed", 0),
            "completed_steps": n,
            "requested_steps": steps,
            "qp_failed_steps": int(np.count_nonzero(~result["qp_solved"])),
        }
        self.rollout_records.append(
            {
                "kappa": float(kappa),
                **info,
                "failed": bool(failed),
                "h_pred": h_pred.tolist(),
                "h_true": h_true.tolist(),
                "control_deviation": u_dev.tolist(),
                "qp_solved": result["qp_solved"].tolist(),
                "rows": result["rows"],
            }
        )
        return EvaluatorOutput(h_pred, h_true, u_dev, bool(failed), info)
