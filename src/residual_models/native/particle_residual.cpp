#include <cmath>
#include <cstddef>

namespace {
double g_mb = 1.0;
double g_gravity = 9.81;
double g_radius = 0.1;
}  // namespace

extern "C" {

void particle_set_params(double mb, double gravity, double radius) {
    g_mb = mb;
    g_gravity = gravity;
    g_radius = radius;
}

void particle_residual(const double* z, const double* theta, double kappa, double* out) {
    // Fixed dimensions for the current Particle setup:
    // nq=3, nc=1, nb=4, nu=3, nw=0, nf=1.
    const double* q2 = z;          // [0..2]
    const double gamma1 = z[3];
    const double* b1 = z + 4;      // [4..7]
    const double psi1 = z[8];
    const double sgamma1 = z[9];
    const double* sb1 = z + 10;    // [10..13]
    const double spsi1 = z[14];

    const double* q0 = theta;      // [0..2]
    const double* q1 = theta + 3;  // [3..5]
    const double* u1 = theta + 6;  // [6..8]
    const double mu = theta[9];
    const double h = theta[10];

    const double inv_h = 1.0 / h;

    // v1 = (q2 - q1) / h
    const double v1x = (q2[0] - q1[0]) * inv_h;
    const double v1y = (q2[1] - q1[1]) * inv_h;
    const double v1z = (q2[2] - q1[2]) * inv_h;

    // vT_stack = P * v1, P = [[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]]
    const double vT0 = v1x;
    const double vT1 = v1y;
    const double vT2 = -v1x;
    const double vT3 = -v1y;

    // Lambda1 = [b0 - b2, b1 - b3, gamma1]
    const double lambda0 = b1[0] - b1[2];
    const double lambda1 = b1[1] - b1[3];
    const double lambda2 = gamma1;

    // dyn = mb*(2*q1 - q0 - q2)/h + u1 + Lambda1 + [0,0,-h*g*mb]
    const double mb_over_h = g_mb * inv_h;
    const double grav_term = -h * g_gravity * g_mb;
    out[0] = mb_over_h * (2.0 * q1[0] - q0[0] - q2[0]) + u1[0] + lambda0;
    out[1] = mb_over_h * (2.0 * q1[1] - q0[1] - q2[1]) + u1[1] + lambda1;
    out[2] = mb_over_h * (2.0 * q1[2] - q0[2] - q2[2]) + u1[2] + lambda2 + grav_term;

    // res_sd = sgamma1 - (q2[2] - r)
    out[3] = sgamma1 - (q2[2] - g_radius);

    // res_vT = sb1 - vT_stack - psi_stack (psi_stack = [psi1]*4)
    out[4] = sb1[0] - vT0 - psi1;
    out[5] = sb1[1] - vT1 - psi1;
    out[6] = sb1[2] - vT2 - psi1;
    out[7] = sb1[3] - vT3 - psi1;

    // res_fric_ineq = spsi1 - (mu*gamma1 - sum(b1))
    const double bsum = b1[0] + b1[1] + b1[2] + b1[3];
    out[8] = spsi1 - (mu * gamma1 - bsum);

    // res_normal_comp = gamma1 * sgamma1 - kappa
    out[9] = gamma1 * sgamma1 - kappa;

    // res_b_comp = b1 * sb1 - kappa
    out[10] = b1[0] * sb1[0] - kappa;
    out[11] = b1[1] * sb1[1] - kappa;
    out[12] = b1[2] * sb1[2] - kappa;
    out[13] = b1[3] * sb1[3] - kappa;

    // res_psi_comp = psi1 * spsi1 - kappa
    out[14] = psi1 * spsi1 - kappa;
}

void particle_rz(const double* z, const double* theta, double* out) {
    // Jacobian of residual w.r.t z (15x15, row-major).
    const double* q2 = z;
    const double gamma1 = z[3];
    const double* b1 = z + 4;
    const double psi1 = z[8];
    const double sgamma1 = z[9];
    const double* sb1 = z + 10;
    const double spsi1 = z[14];

    const double h = theta[10];
    const double inv_h = 1.0 / h;
    const double mb_over_h = g_mb * inv_h;

    // Zero-initialize.
    for (int i = 0; i < 15 * 15; ++i) {
        out[i] = 0.0;
    }

    // dyn rows (0..2)
    // d dyn / d q2
    out[0 * 15 + 0] = -mb_over_h;
    out[1 * 15 + 1] = -mb_over_h;
    out[2 * 15 + 2] = -mb_over_h;

    // d dyn / d gamma1
    out[2 * 15 + 3] = 1.0;

    // d dyn / d b1
    out[0 * 15 + 4] = 1.0;
    out[0 * 15 + 6] = -1.0;
    out[1 * 15 + 5] = 1.0;
    out[1 * 15 + 7] = -1.0;

    // res_sd row (3)
    out[3 * 15 + 2] = -1.0;
    out[3 * 15 + 9] = 1.0;

    // res_vT rows (4..7): sb1 - vT - psi1
    out[4 * 15 + 0] = -inv_h;
    out[5 * 15 + 1] = -inv_h;
    out[6 * 15 + 0] = inv_h;
    out[7 * 15 + 1] = inv_h;

    // d res_vT / d sb1
    out[4 * 15 + 10] = 1.0;
    out[5 * 15 + 11] = 1.0;
    out[6 * 15 + 12] = 1.0;
    out[7 * 15 + 13] = 1.0;

    // d res_vT / d psi1
    out[4 * 15 + 8] = -1.0;
    out[5 * 15 + 8] = -1.0;
    out[6 * 15 + 8] = -1.0;
    out[7 * 15 + 8] = -1.0;

    // res_fric_ineq row (8)
    out[8 * 15 + 14] = 1.0;
    out[8 * 15 + 3] = -theta[9];  // -mu
    out[8 * 15 + 4] = 1.0;
    out[8 * 15 + 5] = 1.0;
    out[8 * 15 + 6] = 1.0;
    out[8 * 15 + 7] = 1.0;

    // res_normal_comp row (9)
    out[9 * 15 + 3] = sgamma1;
    out[9 * 15 + 9] = gamma1;

    // res_b_comp rows (10..13)
    out[10 * 15 + 4] = sb1[0];
    out[10 * 15 + 10] = b1[0];
    out[11 * 15 + 5] = sb1[1];
    out[11 * 15 + 11] = b1[1];
    out[12 * 15 + 6] = sb1[2];
    out[12 * 15 + 12] = b1[2];
    out[13 * 15 + 7] = sb1[3];
    out[13 * 15 + 13] = b1[3];

    // res_psi_comp row (14)
    out[14 * 15 + 8] = spsi1;
    out[14 * 15 + 14] = psi1;
}

void particle_rtheta(const double* z, const double* theta, double* out) {
    // Jacobian of residual w.r.t theta (15x11, row-major).
    const double* q2 = z;
    const double gamma1 = z[3];

    const double* q0 = theta;
    const double* q1 = theta + 3;
    const double* u1 = theta + 6;
    const double mu = theta[9];
    const double h = theta[10];

    const double inv_h = 1.0 / h;
    const double mb_over_h = g_mb * inv_h;

    // vT = (1/h) * P * (q2 - q1)
    const double v1x = (q2[0] - q1[0]) * inv_h;
    const double v1y = (q2[1] - q1[1]) * inv_h;
    const double vT0 = v1x;
    const double vT1 = v1y;
    const double vT2 = -v1x;
    const double vT3 = -v1y;

    // Zero-initialize.
    for (int i = 0; i < 15 * 11; ++i) {
        out[i] = 0.0;
    }

    // dyn rows (0..2)
    // d dyn / d q0
    out[0 * 11 + 0] = -mb_over_h;
    out[1 * 11 + 1] = -mb_over_h;
    out[2 * 11 + 2] = -mb_over_h;

    // d dyn / d q1
    out[0 * 11 + 3] = 2.0 * mb_over_h;
    out[1 * 11 + 4] = 2.0 * mb_over_h;
    out[2 * 11 + 5] = 2.0 * mb_over_h;

    // d dyn / d u1
    out[0 * 11 + 6] = 1.0;
    out[1 * 11 + 7] = 1.0;
    out[2 * 11 + 8] = 1.0;

    // d dyn / d h
    const double h2 = h * h;
    const double d_base = -g_mb / h2;
    out[0 * 11 + 10] = d_base * (2.0 * q1[0] - q0[0] - q2[0]);
    out[1 * 11 + 10] = d_base * (2.0 * q1[1] - q0[1] - q2[1]);
    out[2 * 11 + 10] = d_base * (2.0 * q1[2] - q0[2] - q2[2]) - g_gravity * g_mb;

    // res_vT rows (4..7): d/d q1 and d/d h
    // d res_vT / d q1
    out[4 * 11 + 3] = inv_h;
    out[5 * 11 + 4] = inv_h;
    out[6 * 11 + 3] = -inv_h;
    out[7 * 11 + 4] = -inv_h;

    // d res_vT / d h = vT / h
    out[4 * 11 + 10] = vT0 * inv_h;
    out[5 * 11 + 10] = vT1 * inv_h;
    out[6 * 11 + 10] = vT2 * inv_h;
    out[7 * 11 + 10] = vT3 * inv_h;

    // res_fric_ineq row (8): d/d mu
    out[8 * 11 + 9] = -gamma1;
    (void)mu;
}

}  // extern "C"
