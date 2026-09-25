from autograd import jacobian
from autograd import numpy as anp

from src.solver.indices import IndicesOptimization, IndicesTheta, IndicesZ
from src.simulator.dynamics import dynamics_auto


class Hopper:
    """Contact-implicit hopper model (Python port structure of motion_planning/models/hopper.jl).

    q = [x, z, theta, r]
    u = [M, r_dot]
    """

    def __init__(
        self,
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
    ):
        self.nq = nq
        self.nu = nu
        self.nw = nw
        self.nc = nc

        self.mb = mb
        self.ml = ml
        self.jb = jb
        self.jl = jl
        self.mu = mu
        self.gravity = gravity
        self.r_min = r_min
        self.r_max = r_max

        self.lc_fric = 2
        self.nb = self.lc_fric * self.nc

        self.friction_coeff = anp.array([self.mu])
        self.nf = len(self.friction_coeff)

        self.idx_q1 = anp.arange(0, nq)
        self.idx_q2 = anp.arange(nq, 2 * nq)
        self.idx_u = anp.arange(2 * nq, 2 * nq + nu)
        self.idx_w = anp.arange(2 * nq + nu, 2 * nq + nu + nw)
        self.idx_f = anp.arange(2 * nq + nu + nw, 2 * nq + nu + nw + self.nf)
        self.idx_h = anp.arange(
            2 * nq + nu + nw + self.nf, 2 * nq + nu + nw + self.nf + 1
        )

        # Stand with nominal leg length.
        self._nominal_configuration = anp.array([0.0, r_max + 1.0e-8, 0.0, r_max])

    def M_func(self, q):
        return anp.diag(
            [
                self.mb + self.ml,
                self.mb + self.ml,
                self.jb + self.jl,
                self.ml,
            ]
        )

    def C_func(self, q, qdot):
        return anp.array([0.0, (self.mb + self.ml) * self.gravity, 0.0, 0.0])

    def kinematics(self, q):
        return anp.array(
            [
                q[0] + q[3] * anp.sin(q[2]),
                q[1] - q[3] * anp.cos(q[2]),
            ]
        )

    def signed_distance(self, q):
        # Single contact distance (foot to flat ground).
        return anp.array([q[1] - q[3] * anp.cos(q[2])])

    def contact_jacobian(self, q):
        phi_fun = lambda x: self.signed_distance(x)
        return jacobian(phi_fun)(q)

    def P_function(self, q):
        # Friction rays (linearized cone, 2 rays in 2D).
        return anp.array(
            [
                [1.0, 0.0, q[3] * anp.cos(q[2]), anp.sin(q[2])],
                [-1.0, 0.0, -q[3] * anp.cos(q[2]), -anp.sin(q[2])],
            ]
        )

    def input_jacobian(self, q):
        # Same as transpose(B_func(model, q)) in hopper.jl.
        return anp.array(
            [
                [0.0, -anp.sin(q[2])],
                [0.0, anp.cos(q[2])],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

    def nominal_configuration(self):
        return anp.array(self._nominal_configuration)

    def set_nominal_configuration(self, q):
        q_array = anp.array(q)
        if q_array.shape != (self.nq,):
            raise ValueError(
                f"Nominal configuration must have shape ({self.nq},), got {q_array.shape}"
            )
        self._nominal_configuration = q_array

    def friction_coefficients(self):
        return self.friction_coeff

    def num_var(self):
        # z = [q2, gamma1, b1, psi1, s_gamma1, s_b1, s_psi1]
        nq = self.nq
        nc = self.nc
        nb = self.nb
        return nq + nc + nb + nc + nc + nb + nc

    def indices_z(self):
        idx = 0
        q = anp.arange(idx, idx + self.nq)
        idx += self.nq

        gamma = anp.arange(idx, idx + self.nc)
        idx += self.nc

        b = anp.arange(idx, idx + self.nb)
        idx += self.nb

        psi = anp.arange(idx, idx + self.nc)
        idx += self.nc

        s_gamma = anp.arange(idx, idx + self.nc)
        idx += self.nc

        s_b = anp.arange(idx, idx + self.nb)
        idx += self.nb

        s_psi = anp.arange(idx, idx + self.nc)
        idx += self.nc

        self.q = q
        self.gamma = gamma
        self.b = b
        self.psi = psi
        self.s_gamma = s_gamma
        self.s_b = s_b
        self.s_psi = s_psi

        return IndicesZ(q, gamma, s_gamma, psi, b, s_psi, s_b)

    def initialize_z(self, z, q, idx=None):
        eps = 3.16227766e-2
        z[self.q] = q
        z[self.gamma] = eps
        z[self.b] = eps
        z[self.psi] = eps
        z[self.s_gamma] = eps
        z[self.s_b] = eps
        z[self.s_psi] = eps
        return z

    def indices_theta(self):
        return IndicesTheta(self)

    def indices_optimization(self):
        if not hasattr(self, "q"):
            self.indices_z()

        nq = self.nq
        nz = self.num_var()
        nc = self.nc
        nb = self.nb

        # Orthant blocks: primal [gamma, b, psi], dual slacks [s_gamma, s_b, s_psi]
        ortz = [
            anp.concatenate([self.gamma, self.b, self.psi]),
            anp.concatenate([self.s_gamma, self.s_b, self.s_psi]),
        ]
        ort_delta = ortz

        # Residual order (matches common pattern in this repo):
        # [dyn, res_sd, res_vT, res_fric_ineq, res_normal_comp, res_b_comp, res_psi_comp]
        equr = anp.arange(0, nq + nc + nb + nc)
        ortr = anp.arange(nq + nc + nb + nc, nq + nc + nb + nc + nc + nb + nc)

        return IndicesOptimization(
            nz=nz,
            n_delta=nz,
            ortz=ortz,
            ort_delta=ort_delta,
            socz=[],
            soc_delta=[],
            equr=equr,
            ortr=ortr,
            socr=[],
            socri=[],
            bil=ortr,
        )

    def residual(self, z, theta, kappa):
        if not hasattr(self, "q"):
            self.indices_z()

        z = anp.reshape(z, (-1,))
        theta = anp.reshape(theta, (-1,))

        q0 = theta[self.idx_q1]
        q1 = theta[self.idx_q2]
        u1 = theta[self.idx_u]
        w1 = theta[self.idx_w]
        mu = theta[self.idx_f]
        h = theta[self.idx_h][0]

        q2 = z[self.q]
        gamma1 = z[self.gamma]
        b1 = z[self.b]
        psi1 = z[self.psi]
        sgamma1 = z[self.s_gamma]
        sb1 = z[self.s_b]
        spsi1 = z[self.s_psi]

        v1 = (q2 - q1) / h
        vT_stack = self.P_function(q2) @ v1

        # Contact generalized force.
        J = self.contact_jacobian(q2)
        P_force = self.P_function(q2)
        Lambda1 = J.T @ gamma1 + P_force.T @ b1

        dyn = dynamics_auto(self, h, q0, q1, u1, w1, Lambda1, q2)
        phi = self.signed_distance(q2)

        E = anp.zeros((self.nc, self.nb))
        for j in range(self.nc):
            start = j * self.lc_fric
            end = start + self.lc_fric
            E[j, start:end] = 1.0

        psi_stack = E.T @ psi1
        res_sd = phi - sgamma1
        res_vT = sb1 - vT_stack - psi_stack
        res_fric_ineq = spsi1 - (mu * gamma1 - E @ b1)

        res_normal_comp = gamma1 * sgamma1 - kappa
        res_b_comp = b1 * sb1 - kappa
        res_psi_comp = psi1 * spsi1 - kappa

        return anp.concatenate(
            [
                dyn,
                anp.atleast_1d(res_sd),
                anp.atleast_1d(res_vT),
                anp.atleast_1d(res_fric_ineq),
                anp.atleast_1d(res_normal_comp),
                anp.atleast_1d(res_b_comp),
                anp.atleast_1d(res_psi_comp),
            ]
        )
