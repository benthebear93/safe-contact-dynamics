#include <cmath>
#include <cstddef>

namespace {
double g_mb = 1.0;
double g_mp = 1.0;
double g_I = 1.0;
double g_r_box = 0.1;
double g_r_pusher = 0.025;
double g_gravity = 9.81;
}  // namespace

constexpr int kNq = 7;
constexpr int kNc = 5;
constexpr int kNb = 20;
constexpr int kNu = 2;
constexpr int kNf = 2;
constexpr int kNz = 67;
constexpr int kNtheta = 2 * kNq + kNu + kNf + 1;

inline double sign_double(double v) {
    return (v > 0.0) - (v < 0.0);
}

void rotation_yaw(double theta, double R[3][3]) {
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    R[0][0] = c;
    R[0][1] = -s;
    R[0][2] = 0.0;
    R[1][0] = s;
    R[1][1] = c;
    R[1][2] = 0.0;
    R[2][0] = 0.0;
    R[2][1] = 0.0;
    R[2][2] = 1.0;
}

void signed_distance(const double* q, double phi[kNc], double grad_last[7]) {
    // q = [box x,y,z, theta, pusher x,y,z]
    const double box_x = q[0];
    const double box_y = q[1];
    const double box_z = q[2];
    const double theta = q[3];
    const double push_x = q[4];
    const double push_y = q[5];

    // First 4 corners: z coordinate of box corners.
    for (int i = 0; i < 4; ++i) {
        phi[i] = box_z - g_r_box;
    }

    // Pusher contact SDF using Lp norm in box frame.
    const double dx = push_x - box_x;
    const double dy = push_y - box_y;
    const double c = std::cos(theta);
    const double s = std::sin(theta);

    const double ax = c * dx + s * dy;
    const double ay = -s * dx + c * dy;

    const double s_exp = 10.0;
    const double ax_abs = std::abs(ax);
    const double ay_abs = std::abs(ay);
    const double ax_s = std::pow(ax_abs, s_exp);
    const double ay_s = std::pow(ay_abs, s_exp);
    const double sum_s = ax_s + ay_s;
    const double lp = std::pow(sum_s, 1.0 / s_exp);
    phi[4] = lp - (g_r_box + g_r_pusher);

    // Gradient of lp wrt ax, ay.
    double dlp_dax = 0.0;
    double dlp_day = 0.0;
    if (sum_s > 0.0) {
        const double factor = std::pow(sum_s, (1.0 / s_exp) - 1.0);
        dlp_dax = sign_double(ax) * std::pow(ax_abs, s_exp - 1.0) * factor;
        dlp_day = sign_double(ay) * std::pow(ay_abs, s_exp - 1.0) * factor;
    }

    // grad_world = R * grad_local (R = R_T^T).
    const double grad_x = c * dlp_dax - s * dlp_day;
    const double grad_y = s * dlp_dax + c * dlp_day;

    // dlp/dtheta = grad_local dot (dR_T/dtheta * p_rel)
    const double dR_ax = -s * dx + c * dy;
    const double dR_ay = -c * dx - s * dy;
    const double dlp_dtheta = dlp_dax * dR_ax + dlp_day * dR_ay;

    // Fill gradient for last row (pusher SDF) in order q.
    grad_last[0] = -grad_x;
    grad_last[1] = -grad_y;
    grad_last[2] = 0.0;
    grad_last[3] = dlp_dtheta;
    grad_last[4] = grad_x;
    grad_last[5] = grad_y;
    grad_last[6] = 0.0;
}

void contact_jacobian(const double* q, double J[kNc][kNq]) {
    for (int i = 0; i < kNc; ++i) {
        for (int j = 0; j < kNq; ++j) {
            J[i][j] = 0.0;
        }
    }

    // First 4 rows: dphi/dq = [0,0,1,0,0,0,0]
    for (int i = 0; i < 4; ++i) {
        J[i][2] = 1.0;
    }

    double grad_last[7] = {0.0};
    double phi[kNc] = {0.0};
    signed_distance(q, phi, grad_last);
    for (int j = 0; j < kNq; ++j) {
        J[4][j] = grad_last[j];
    }
}

void p_box_function(const double* q, double P_box[16][kNq]) {
    // P_box is Jacobian of p_box (16x7).
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < kNq; ++j) {
            P_box[i][j] = 0.0;
        }
    }

    const double theta = q[3];
    const double c = std::cos(theta);
    const double s = std::sin(theta);

    // Corner offsets (x, y)
    const double offsets[4][2] = {
        {+g_r_box, +g_r_box},
        {-g_r_box, +g_r_box},
        {+g_r_box, -g_r_box},
        {-g_r_box, -g_r_box},
    };

    for (int i = 0; i < 4; ++i) {
        const double ox = offsets[i][0];
        const double oy = offsets[i][1];
        const double dxy_theta_x = -s * ox - c * oy;
        const double dxy_theta_y = c * ox - s * oy;

        const int row = i * 4;
        // map_mat rows: [1,0], [0,1], [-1,0], [0,-1]
        // Row 0
        P_box[row + 0][0] = 1.0;
        P_box[row + 0][1] = 0.0;
        P_box[row + 0][3] = dxy_theta_x;
        // Row 1
        P_box[row + 1][0] = 0.0;
        P_box[row + 1][1] = 1.0;
        P_box[row + 1][3] = dxy_theta_y;
        // Row 2
        P_box[row + 2][0] = -1.0;
        P_box[row + 2][1] = 0.0;
        P_box[row + 2][3] = -dxy_theta_x;
        // Row 3
        P_box[row + 3][0] = 0.0;
        P_box[row + 3][1] = -1.0;
        P_box[row + 3][3] = -dxy_theta_y;
    }
}

void pusher_P(const double* q, double P[20][kNq]) {
    double P_box[16][kNq] = {{0.0}};
    p_box_function(q, P_box);

    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < kNq; ++j) {
            P[i][j] = P_box[i][j];
        }
    }

    double J[kNc][kNq] = {{0.0}};
    contact_jacobian(q, J);
    const double nx = J[4][4];
    const double ny = J[4][5];
    const double n_norm = std::sqrt(nx * nx + ny * ny);
    const double inv_norm = 1.0 / (n_norm + 1e-12);
    const double n_dir_x = nx * inv_norm;
    const double n_dir_y = ny * inv_norm;
    const double t_dir_x = -n_dir_y;
    const double t_dir_y = n_dir_x;

    const double map_mat[4][2] = {
        {1.0, 0.0},
        {0.0, 1.0},
        {-1.0, 0.0},
        {0.0, -1.0},
    };

    const double rx = q[4] - q[0];
    const double ry = q[5] - q[1];

    for (int i = 0; i < 4; ++i) {
        // Elementwise map_mat * t_dir (no summation).
        const double row_ax = map_mat[i][0] * t_dir_x;
        const double row_ay = map_mat[i][1] * t_dir_y;
        const double m = rx * row_ay - ry * row_ax;

        const int row = 16 + i;
        P[row][0] = -row_ax;
        P[row][1] = -row_ay;
        P[row][2] = 0.0;
        P[row][3] = -m;
        P[row][4] = row_ax;
        P[row][5] = row_ay;
        P[row][6] = 0.0;
    }
}

void compute_residual(const double* z, const double* theta, double kappa, double* out) {
    // Parse z
    const double* q2 = z;
    const double* gamma1 = z + 7;
    const double* b1 = z + 12;
    const double* psi1 = z + 32;
    const double* sgamma1 = z + 37;
    const double* sb1 = z + 42;
    const double* spsi1 = z + 62;

    // Parse theta
    const double* q0 = theta;
    const double* q1 = theta + 7;
    const double* u1 = theta + 14;
    const double mu_body = theta[16];
    const double mu_pusher = theta[17];
    const double h = theta[18];

    const double inv_h = 1.0 / h;

    // v1 = (q2 - q1) / h
    double v1[kNq];
    for (int i = 0; i < kNq; ++i) {
        v1[i] = (q2[i] - q1[i]) * inv_h;
    }

    // P and vT_stack
    double P[20][kNq] = {{0.0}};
    pusher_P(q2, P);
    double vT_stack[20] = {0.0};
    for (int i = 0; i < 20; ++i) {
        double sum = 0.0;
        for (int j = 0; j < kNq; ++j) {
            sum += P[i][j] * v1[j];
        }
        vT_stack[i] = sum;
    }

    // signed distance
    double phi[kNc] = {0.0};
    double grad_last[7] = {0.0};
    signed_distance(q2, phi, grad_last);

    // contact jacobian at q1
    double J[kNc][kNq] = {{0.0}};
    contact_jacobian(q1, J);

    // Lambda_normal = J.T @ gamma1
    double lambda_normal[kNq] = {0.0};
    for (int j = 0; j < kNq; ++j) {
        double sum = 0.0;
        for (int i = 0; i < kNc; ++i) {
            sum += J[i][j] * gamma1[i];
        }
        lambda_normal[j] = sum;
    }

    // Lambda_tangent = P(q1).T @ b1
    double P_force[20][kNq] = {{0.0}};
    pusher_P(q1, P_force);
    double lambda_tangent[kNq] = {0.0};
    for (int j = 0; j < kNq; ++j) {
        double sum = 0.0;
        for (int i = 0; i < kNb; ++i) {
            sum += P_force[i][j] * b1[i];
        }
        lambda_tangent[j] = sum;
    }

    double lambda1[kNq] = {0.0};
    for (int i = 0; i < kNq; ++i) {
        lambda1[i] = lambda_normal[i] + lambda_tangent[i];
    }

    // dynamics
    const double mb_over_h = g_mb * inv_h;
    const double mp_over_h = g_mp * inv_h;
    const double I_over_h = g_I * inv_h;
    const double grav_term = -h * g_mb * g_gravity;

    out[0] = mb_over_h * (2.0 * q1[0] - q0[0] - q2[0]) + lambda1[0];
    out[1] = mb_over_h * (2.0 * q1[1] - q0[1] - q2[1]) + lambda1[1];
    out[2] = mb_over_h * (2.0 * q1[2] - q0[2] - q2[2]) + lambda1[2] + grav_term;
    out[3] = I_over_h * (2.0 * q1[3] - q0[3] - q2[3]) + lambda1[3];
    out[4] = mp_over_h * (2.0 * q1[4] - q0[4] - q2[4]) + u1[0] + lambda1[4];
    out[5] = mp_over_h * (2.0 * q1[5] - q0[5] - q2[5]) + u1[1] + lambda1[5];
    out[6] = mp_over_h * (2.0 * q1[6] - q0[6] - q2[6]) + lambda1[6];

    // res_sd
    for (int i = 0; i < kNc; ++i) {
        out[7 + i] = phi[i] - sgamma1[i];
    }

    // psi_stack and res_vT
    double psi_stack[20] = {0.0};
    for (int i = 0; i < kNc; ++i) {
        for (int k = 0; k < 4; ++k) {
            psi_stack[i * 4 + k] = psi1[i];
        }
    }
    for (int i = 0; i < kNb; ++i) {
        out[12 + i] = sb1[i] - vT_stack[i] - psi_stack[i];
    }

    // res_fric_ineq
    for (int i = 0; i < kNc; ++i) {
        double sum_b = 0.0;
        for (int k = 0; k < 4; ++k) {
            sum_b += b1[i * 4 + k];
        }
        const double mu_contact = (i < 4) ? mu_body : mu_pusher;
        out[32 + i] = spsi1[i] - (mu_contact * gamma1[i] - sum_b);
    }

    // res_normal_comp
    for (int i = 0; i < kNc; ++i) {
        out[37 + i] = gamma1[i] * sgamma1[i] - kappa;
    }

    // res_b_comp
    for (int i = 0; i < kNb; ++i) {
        out[42 + i] = b1[i] * sb1[i] - kappa;
    }

    // res_psi_comp
    for (int i = 0; i < kNc; ++i) {
        out[62 + i] = psi1[i] * spsi1[i] - kappa;
    }
}

extern "C" {

void pusher_set_params(
    double mb,
    double mp,
    double I,
    double r_box,
    double r_pusher,
    double gravity) {
    g_mb = mb;
    g_mp = mp;
    g_I = I;
    g_r_box = r_box;
    g_r_pusher = r_pusher;
    g_gravity = gravity;
}

void pusher_residual(const double* z, const double* theta, double kappa, double* out) {
    compute_residual(z, theta, kappa, out);
}

void pusher_rz(const double* z, const double* theta, double* out) {
    const double eps = 1e-6;
    double base[kNz] = {0.0};
    compute_residual(z, theta, 0.0, base);

    double z_perturbed[kNz];
    for (int i = 0; i < kNz; ++i) {
        z_perturbed[i] = z[i];
    }

    for (int col = 0; col < kNz; ++col) {
        z_perturbed[col] += eps;
        double r1[kNz] = {0.0};
        compute_residual(z_perturbed, theta, 0.0, r1);
        z_perturbed[col] = z[col];
        for (int row = 0; row < kNz; ++row) {
            out[row * kNz + col] = (r1[row] - base[row]) / eps;
        }
    }
}

void pusher_rtheta(const double* z, const double* theta, double* out) {
    const double eps = 1e-6;
    double base[kNz] = {0.0};
    compute_residual(z, theta, 0.0, base);

    double theta_perturbed[kNtheta];
    for (int i = 0; i < kNtheta; ++i) {
        theta_perturbed[i] = theta[i];
    }

    for (int col = 0; col < kNtheta; ++col) {
        theta_perturbed[col] += eps;
        double r1[kNz] = {0.0};
        compute_residual(z, theta_perturbed, 0.0, r1);
        theta_perturbed[col] = theta[col];
        for (int row = 0; row < kNz; ++row) {
            out[row * kNtheta + col] = (r1[row] - base[row]) / eps;
        }
    }
}

}  // extern "C"
