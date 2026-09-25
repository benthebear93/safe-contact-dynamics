import numpy as np

from src.solver.interior_point_solver import (
    InteriorPointOptions,
    initialize_interior_point,
    interior_point_solve,
    bilinear_violation,
    residual_violation,
)
from src.simulator.trajectory import Trajectory, GradientTrajectory
from src.simulator.policy import empty_policy
from src.simulator.disturbance import empty_disturbances
from src.helper import Euclidean, num_data


class SimulatorOptions:
    def __init__(
        self,
        warmstart=True,
        record=False,
        z_warmstart=0.001,
        kappa_warmstart=0.001,
        failure_abort=50,
    ):
        self.warmstart = warmstart
        self.record = record
        self.z_warmstart = z_warmstart
        self.kappa_warmstart = kappa_warmstart
        self.failure_abort = failure_abort


class SimulatorStatistics:
    def __init__(self, H):
        self.policy_time = np.zeros(H)
        self.policy_mean = np.zeros(1)
        self.policy_std = np.zeros(1)
        self.sim_time = np.zeros(H)
        self.sim_mean = np.zeros(1)
        self.sim_std = np.zeros(1)


class Simulator:
    def __init__(
        self,
        model,
        T,
        h,
        diff_sol,
        residual,
        jacobian_z,
        jacobian_theta,
        temp_cmd=None,
        policy=None,
        kappa_tol=1e-8,
    ):
        # print("Initiate Simulator")
        self.model = model
        if temp_cmd is None:
            temp_cmd = np.zeros(model.nu)
        self.temp_cmd = temp_cmd
        self.h = h
        # Cache the current friction coefficients (mu_body, mu_pusher). We will refresh this
        # every step in case the model's friction is updated at runtime.
        self.f = model.friction_coefficients()
        # Initialize placeholders for policy and disturbance
        if policy is None:
            self.policy = empty_policy(model)
        else:
            self.policy = policy
        self.disturbance = empty_disturbances(model)

        # Initialize logging/statistics and options
        self.stats = SimulatorStatistics(T)
        self.opts = SimulatorOptions()

        # Index structures for optimization variables
        self.idx_z = model.indices_z()
        self.idx_theta = model.indices_theta()
        self.idx_opt = model.indices_optimization()
        self.prev_z = None

        # Prepare initial values for optimization variables
        nz = model.num_var()
        ntheta = num_data(model)
        z0 = np.zeros(nz)
        theta0 = np.zeros(ntheta)
        q0 = model.nominal_configuration()
        z0 = model.initialize_z(z0, q0)

        theta0 = self.idx_theta.initialize_theta(
            theta0,
            q0,
            q0,
            np.zeros(model.nu),
            np.zeros(model.nw),
            self.f,
            h,
        )
        # Construct the interior-point problem using all components
        self.ip = initialize_interior_point(
            z=z0,
            theta=theta0,
            space=Euclidean(len(z0)),
            idx=self.idx_opt,
            r_func=residual,
            rz_func=jacobian_z,
            rtheta_func=jacobian_theta,
            options=InteriorPointOptions(kappa_tol=kappa_tol, diff_sol=diff_sol),
        )
        self.ip.model = model
        self.diff_sol = diff_sol
        # Initialize trajectory containers
        nb = getattr(model, "nb", model.nc)
        self.traj = Trajectory(model, T, nb=nb, h=self.h)
        self.grad = GradientTrajectory(model, T, nc=model.nc, nb=nb)

    def set_state(self, q, v, t):
        # initial configuration and velocity
        self.traj.q[t + 1] = q  # q2
        self.traj.v[t] = v  # v1
        self.traj.q[t] = q - self.h * v

    def simulate_start(self, q, v, reset_traj=True):
        if reset_traj:
            self.traj.reset()
            self.grad.reset()
            self.prev_z = None
        self.set_state(q, v, 0)
        status = self.simulate()
        return status

    def simulate(self):
        status = False

        N = len(self.traj.u)
        p = self.policy
        w = self.disturbance
        for t in range(N):
            # print(f"traj time : {t}")
            # Support simulator policies (Policy subclasses) or CI-MPC policies.
            if isinstance(p, Policy):
                u_cmd = self.temp_cmd  # [0, 0.1 * t]  # p(self.traj, t)  # [0, 0.01]
            else:
                from src.mpc.policy import policy as mpc_policy

                u_cmd = mpc_policy(p, self.traj, t)
            self.traj.u[t][:] = u_cmd

            # disturbances
            self.traj.w[t][:] = w(
                self.traj.q[t + 1], t
            )  # disturbances(w, traj.q[t+1], t)

            status = self.step(t, self.diff_sol)  # FIXME!!!

            if not status:
                break

        return status

    def step(self, t, diff_sol):
        traj = self.traj
        ip = self.ip

        # Match Julia behavior: initialize via model-specific routine when available.
        try:
            self.model.initialize_z(ip.z, traj.q[t + 1])
        except TypeError:
            # Fall back to index-based initialization if the model signature differs.
            self.idx_z.initialize_z(ip.z, traj.q[t + 1])
        # Capture the post-initialization state for debugging.
        ip.last_z_init = ip.z.copy()

        # Refresh friction each step so any updates on the model are honored.
        self.f = self.model.friction_coefficients()
        self.idx_theta.initialize_theta(
            ip.theta, traj.q[t], traj.q[t + 1], traj.u[t], traj.w[t], self.f, self.h
        )

        ip.current_t = int(t)
        ip.suppress_violation_logs = True
        status = interior_point_solve(ip)
        # print("status: ", status)
        if not status:
            # Recompute residual to report which equality constraint is failing most.
            try:
                r = ip.residual_methods["r"](ip.z, ip.theta, 0.0)
            except TypeError:
                ip.residual_methods["r"](ip.r, ip.z, ip.theta, 0.0)
                r = ip.r
            if r is None:
                r = ip.r
            ip.suppress_violation_logs = False
            _ = bilinear_violation(ip, r)
            _ = residual_violation(ip, r)
            r_arr = np.asarray(r)
            r_eq = r_arr[ip.idx.equr]
            local_idx = int(np.argmax(np.abs(r_eq)))
            global_idx = int(ip.idx.equr[local_idx])
            val = float(r_eq[local_idx])
            detail = "unknown"
            if (
                hasattr(ip, "model")
                and hasattr(ip.model, "nq")
                and hasattr(ip.model, "nc")
            ):
                nq = int(ip.model.nq)
                nc = int(ip.model.nc)
                nb = int(getattr(ip.model, "nb", 0))
                if global_idx < nq:
                    detail = f"dyn[{global_idx}]"
                elif global_idx < nq + nc:
                    detail = f"res_sd[{global_idx - nq}]"
                elif global_idx < nq + nc + nb:
                    detail = f"res_vT[{global_idx - nq - nc}]"
                elif global_idx < nq + nc + nb + nc:
                    detail = f"res_fric_ineq[{global_idx - nq - nc - nb}]"
            phi = self.model.signed_distance(traj.q[t + 1])
            u = traj.u[t]
            r_vol = getattr(ip, "last_r_vol", None)
            k_vol = getattr(ip, "last_k_vol", None)
            print(
                f"solver failed at t={t}, r_vol={r_vol}, k_vol={k_vol}, "
                f"u={u}, phi={phi}, r_max_idx={global_idx}, r_max_val={val}, "
                f"r_max_detail={detail}"
            )
            return False

        self.solution(ip.z, t)
        if self.opts.warmstart:
            self.prev_z = ip.z.copy()
        if diff_sol:
            self.gradient(self.h, ip.delta_z, t)
        return status

    def _apply_complementarity_eps(self, z):
        eps = float(np.sqrt(max(self.opts.kappa_warmstart, 0.0)))
        if eps <= 0.0:
            return
        idx = self.idx_z
        if len(idx.gamma) == 0:
            return
        nc = len(idx.gamma)
        nb = len(idx.b)
        fric_dim = nb // nc if nc > 0 else 0
        mu = np.asarray(self.model.friction_coefficients()).reshape(-1)
        if mu.size >= 2 and nc >= 1:
            mu_contact = np.concatenate([np.full(nc - 1, mu[0]), np.array([mu[1]])])
        elif mu.size >= 1:
            mu_contact = np.full(nc, mu[0])
        else:
            mu_contact = np.ones(nc)
        has_soc = (
            hasattr(self, "idx_opt") and len(getattr(self.idx_opt, "socz", [])) > 0
        )
        # Keep friction margin positive: sum(b) <= mu*gamma
        b_scale = 0.5
        for i in range(nc):
            gamma_i = eps
            mu_i = mu_contact[i]
            b_sum_target = b_scale * mu_i * gamma_i
            if len(idx.gamma) > 0:
                z[idx.gamma[i]] = gamma_i
            if len(idx.s_gamma) > 0:
                z[idx.s_gamma[i]] = eps
            if has_soc:
                psi_i = max(eps, mu_i * gamma_i)
                if len(idx.psi) > 0:
                    z[idx.psi[i]] = psi_i
                if len(idx.s_psi) > 0:
                    z[idx.s_psi[i]] = psi_i
                if fric_dim > 0:
                    start = i * fric_dim
                    end = start + fric_dim
                    z[idx.b[start:end]] = 0.0
                    if len(idx.s_b) > 0:
                        z[idx.s_b[start:end]] = 0.0
            else:
                if len(idx.psi) > 0:
                    z[idx.psi[i]] = eps
                if len(idx.s_psi) > 0:
                    spsi = max(eps, mu_i * gamma_i - b_sum_target)
                    z[idx.s_psi[i]] = spsi
                if fric_dim > 0:
                    start = i * fric_dim
                    end = start + fric_dim
                    b_val = b_sum_target / fric_dim
                    z[idx.b[start:end]] = b_val
                    if len(idx.s_b) > 0:
                        z[idx.s_b[start:end]] = eps

    def step_with_state(self, q, v, u, t, diff_sol):
        self.set_state(q, v, t)
        self.traj.u[t] = u
        self.step(t, diff_sol)
        return self.traj.q[t + 2]

    def solution(self, z, t):
        self.q = z[self.idx_z.q]
        self.gamma = z[self.idx_z.gamma]
        self.s_gamma = z[self.idx_z.s_gamma]
        self.b = z[self.idx_z.b]
        self.s_b = z[self.idx_z.s_b]
        self.psi = z[self.idx_z.psi]
        self.s_psi = z[self.idx_z.s_psi]

        self.traj.q[t + 2] = self.q
        self.traj.v[t + 1] = (self.traj.q[t + 2] - self.traj.q[t + 1]) / self.h
        self.traj.gamma[t] = self.gamma
        self.traj.s_gamma[t] = self.s_gamma
        self.traj.kappa[t] = np.full(
            self.model.nc, float(self.ip.options.kappa_tol), dtype=float
        )
        self.traj.b[t] = self.b
        self.traj.s_b[t] = self.s_b
        self.traj.psi[t] = self.psi
        self.traj.s_psi[t] = self.s_psi
        # Persist full optimization variables and data
        if t < len(self.traj.z):
            self.traj.z[t] = z.copy()
        if t < len(self.traj.theta):
            self.idx_theta.initialize_theta(
                self.traj.theta[t],
                self.traj.q[t],
                self.traj.q[t + 1],
                self.traj.u[t],
                self.traj.w[t],
                # Persist the friction coefficients used for this step.
                self.model.friction_coefficients(),
                self.h,
            )
    def gradient(self, h, delta_z, t):
        def copy_view(dst, src):
            dst[:] = src

        copy_view(
            self.grad.dq3_dq1[t], delta_z[np.ix_(self.idx_z.q, self.idx_theta.q1)]
        )
        copy_view(
            self.grad.dq3_dq2[t], delta_z[np.ix_(self.idx_z.q, self.idx_theta.q2)]
        )
        copy_view(self.grad.dq3_du1[t], delta_z[np.ix_(self.idx_z.q, self.idx_theta.u)])
        copy_view(
            self.grad.dgamma1_dq1[t],
            delta_z[np.ix_(self.idx_z.gamma, self.idx_theta.q1)],
        )
        copy_view(
            self.grad.dgamma1_dq2[t],
            delta_z[np.ix_(self.idx_z.gamma, self.idx_theta.q2)],
        )
        copy_view(
            self.grad.dgamma1_du1[t],
            delta_z[np.ix_(self.idx_z.gamma, self.idx_theta.u)],
        )

        db1_dq1 = delta_z[np.ix_(self.idx_z.b, self.idx_theta.q1)]
        db1_dq2 = delta_z[np.ix_(self.idx_z.b, self.idx_theta.q2)]
        db1_du1 = delta_z[np.ix_(self.idx_z.b, self.idx_theta.u)]

        copy_view(self.grad.db1_dq1[t], db1_dq1)
        copy_view(self.grad.db1_dq2[t], db1_dq2)
        copy_view(self.grad.db1_du1[t], db1_du1)

        self.grad.dq3_dv1[t] = (self.grad.dq3_dq2[t] - self.grad.dq3_dq1[t]) / h
        self.grad.dgamma1_dv1[t] = (
            self.grad.dgamma1_dq2[t] - self.grad.dgamma1_dq1[t]
        ) / h
        self.grad.db1_dv1[t] = (self.grad.db1_dq2[t] - self.grad.db1_dq1[t]) / h

    def local_affine_xdot(self, t):
        """Linearize xdot = (q_{t+2}-q_{t+1})/h around the current step.

        Returns (xdot0, A, B, q1, u0) where:
          xdot ≈ xdot0 + A @ (q1 - q1_0) + B @ (u - u0)
        Requires diff_sol=True for the step so gradients are populated.
        """
        q1 = self.traj.q[t + 1]
        q2 = self.traj.q[t + 2]
        u0 = self.traj.u[t]
        h = self.h
        xdot0 = (q2 - q1) / h
        dq2_dq1 = self.grad.dq3_dq2[t]
        A = (dq2_dq1 - np.eye(len(q1))) / h
        B = self.grad.dq3_du1[t] / h
        return xdot0, A, B, q1.copy(), u0.copy()
