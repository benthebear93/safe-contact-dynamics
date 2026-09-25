from autograd import jacobian
from autograd import numpy as anp

from src.simulator.dynamics import dynamics_auto
from src.solver.indices import IndicesOptimization, IndicesTheta, IndicesZ


def rotation_matrix_pitch(a):
    c = anp.cos(a)
    s = anp.sin(a)
    # Rotation about y axis (right-handed): affects x-z plane.
    return anp.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ]
    )


class TipOverPusher:
    """IPM-ready 2D tip-over model in x-z plane with a dynamic pusher.

    State q = [box_x, box_y, box_z, box_pitch, pusher_x, pusher_y, pusher_z]
    - box rotates about y axis (pitch in x-z view).
    - pusher is modeled as a point/sphere proxy with x-z actuation.
    """

    def __init__(
        self,
        nq=7,
        nu=2,
        nw=0,
        nc=5,
        box_half_width=0.09,
        box_half_height=0.11,
        pusher_radius=0.015,
        mb=1.2,
        mp=10.0,
        I=None,
        mu_floor=0.6,
        mu_pusher=0.5,
        gravity=9.81,
        pusher_gap=0.004,
        pusher_height=0.17,
    ):
        self.nq = nq
        self.nu = nu
        self.nw = nw
        self.nc = nc
        self.mb = mb
        self.mp = mp
        self.box_half_width = box_half_width
        self.box_half_height = box_half_height
        self.pusher_radius = pusher_radius
        self.gravity = gravity
        self.mu_floor = mu_floor
        self.mu_pusher = mu_pusher
        self.friction_coeff = anp.array([mu_floor, mu_pusher])
        self.nf = len(self.friction_coeff)

        # Two friction rays (+t, -t) per contact in this 2D setting.
        self.lc_fric = 2
        self.nb = self.lc_fric * self.nc

        if I is None:
            w = 2.0 * box_half_width
            h = 2.0 * box_half_height
            I = (1.0 / 12.0) * mb * (w * w + h * h)
        self.I = I

        self.idx_q1 = anp.arange(0, nq)
        self.idx_q2 = anp.arange(nq, 2 * nq)
        self.idx_u = anp.arange(2 * nq, 2 * nq + nu)
        self.idx_w = anp.arange(2 * nq + nu, 2 * nq + nu + nw)
        self.idx_f = anp.arange(2 * nq + nu + nw, 2 * nq + nu + nw + self.nf)
        self.idx_h = anp.arange(
            2 * nq + nu + nw + self.nf, 2 * nq + nu + nw + self.nf + 1
        )

        # Four x-z corners in local frame (right/left x bottom/top z).
        self.floor_corner_offsets = [
            anp.array([+self.box_half_width, 0.0, -self.box_half_height]),
            anp.array([-self.box_half_width, 0.0, -self.box_half_height]),
            anp.array([+self.box_half_width, 0.0, +self.box_half_height]),
            anp.array([-self.box_half_width, 0.0, +self.box_half_height]),
        ]

        box_x = 0.0
        box_y = 0.0
        box_z = self.box_half_height + 1e-8
        box_pitch = 0.0
        pusher_x = -self.box_half_width - self.pusher_radius - pusher_gap
        pusher_y = 0.0
        pusher_z = pusher_height
        self._nominal_configuration = anp.array(
            [box_x, box_y, box_z, box_pitch, pusher_x, pusher_y, pusher_z]
        )

    def M_func(self, q):
        return anp.diag([self.mb, self.mb, self.mb, self.I, self.mp, self.mp, self.mp])

    def C_func(self, q, qdot):
        # Gravity acts on the box only in this setup.
        return anp.array([0.0, 0.0, self.mb * self.gravity, 0.0, 0.0, 0.0, 0.0])

    def kinematics(self, q):
        return anp.array(q)

    def input_jacobian(self, q):
        # Controls act on pusher x and z.
        return anp.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
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
        return anp.array([self.mu_floor, self.mu_pusher])

    def signed_distance(self, q):
        box_pos = q[0:3]
        box_pitch = q[3]
        pusher_pos = q[4:7]
        R = rotation_matrix_pitch(box_pitch)

        phi = []
        for corner in self.floor_corner_offsets:
            corner_world = box_pos + R @ corner
            phi.append(corner_world[2])

        # Smooth pusher-vs-box signed distance in the x-z box frame.
        delta = pusher_pos - box_pos
        p_local = R.T @ delta
        a = self.box_half_width + self.pusher_radius
        b = self.box_half_height + self.pusher_radius
        s_exp = 10
        nx = p_local[0] / a
        nz = p_local[2] / b
        lp = (anp.maximum(nx**s_exp + nz**s_exp, 1e-12)) ** (1.0 / s_exp)
        phi_pusher = lp - 1.0
        phi.append(phi_pusher)
        return anp.array(phi)

    def contact_jacobian(self, q):
        phi_fun = lambda x: self.signed_distance(x)
        return jacobian(phi_fun)(q)

    def P_function(self, q):
        # Two rays per contact: +t and -t.
        rows = []
        R = rotation_matrix_pitch(q[3])

        # Floor contacts: tangential direction along world +x.
        for corner in self.floor_corner_offsets:
            r_i = R @ corner
            for sgn in (+1.0, -1.0):
                tx = sgn
                tz = 0.0
                tau_y = r_i[2] * tx - r_i[0] * tz
                rows.append(anp.array([tx, 0.0, tz, tau_y, 0.0, 0.0, 0.0]))

        # Pusher contact: tangent is perpendicular to local normal in x-z plane.
        N = self.contact_jacobian(q)
        n_push = N[-1, 4:7]
        nx = n_push[0]
        nz = n_push[2]
        n_norm = anp.sqrt(nx * nx + nz * nz + 1e-12)
        tx0 = -nz / n_norm
        tz0 = nx / n_norm
        r_bp = q[4:7] - q[0:3]
        for sgn in (+1.0, -1.0):
            tx = sgn * tx0
            tz = sgn * tz0
            # Force on box is -t, on pusher is +t.
            tau_y = -r_bp[2] * tx + r_bp[0] * tz
            rows.append(anp.array([-tx, 0.0, -tz, tau_y, tx, 0.0, tz]))

        return anp.vstack(rows)

    def num_var(self):
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

        sb = anp.arange(idx, idx + self.nb)
        idx += self.nb

        spsi = anp.arange(idx, idx + self.nc)
        idx += self.nc

        self.q = q
        self.gamma = gamma
        self.s_gamma = s_gamma
        self.psi = psi
        self.b = b
        self.spsi = spsi
        self.sb = sb
        return IndicesZ(q, gamma, s_gamma, psi, b, spsi, sb)

    def initialize_z(self, z, q, idx=None):
        eps = 3.16227766e-2
        z[self.q] = q
        z[self.gamma] = eps
        z[self.s_gamma] = eps
        z[self.psi] = eps
        z[self.b] = eps
        z[self.spsi] = eps
        z[self.sb] = eps
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
        ortz = [
            anp.concatenate([self.gamma, self.b, self.psi]),
            anp.concatenate([self.s_gamma, self.sb, self.spsi]),
        ]
        ort_delta = ortz
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
        sgamma1 = z[self.s_gamma]
        psi1 = z[self.psi]
        b1 = z[self.b]
        spsi1 = z[self.spsi]
        sb1 = z[self.sb]

        v1 = (q2 - q1) / h
        P = self.P_function(q2)
        vT_stack = P @ v1

        J = self.contact_jacobian(q1)
        P_force = self.P_function(q1)
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

        mu_floor = mu[0]
        mu_push = mu[1]
        mu_contact = anp.concatenate([anp.full(self.nc - 1, mu_floor), anp.array([mu_push])])
        res_fric_ineq = spsi1 - (mu_contact * gamma1 - E @ b1)

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
