#include <algorithm>
#include <array>
#include <cmath>

namespace {

constexpr int kNq = 7;
constexpr int kNu = 2;
constexpr int kNw = 0;
constexpr int kNc = 5;
constexpr int kLcFric = 2;
constexpr int kNb = kNc * kLcFric;  // 10
constexpr int kNf = 2;
constexpr int kNz = kNq + kNc + kNb + kNc + kNc + kNb + kNc;      // 47
constexpr int kNtheta = 2 * kNq + kNu + kNw + kNf + 1;             // 19
constexpr int kNres = kNz;

// Runtime params from Python model.
double g_mb = 1.2;
double g_mp = 10.0;
double g_I = 0.00842;
double g_box_half_width = 0.09;
double g_box_half_height = 0.11;
double g_pusher_radius = 0.015;
double g_gravity = 9.81;
double g_mu_floor = 0.6;
double g_mu_pusher = 0.5;

inline void matvec7(const double M[7][7], const double x[7], double y[7]) {
  for (int i = 0; i < 7; ++i) {
    double s = 0.0;
    for (int j = 0; j < 7; ++j) {
      s += M[i][j] * x[j];
    }
    y[i] = s;
  }
}

inline void input_jacobian(double B[7][2]) {
  for (int i = 0; i < 7; ++i) {
    B[i][0] = 0.0;
    B[i][1] = 0.0;
  }
  B[4][0] = 1.0;  // pusher x
  B[6][1] = 1.0;  // pusher z
}

inline void mass_diag(double M[7][7]) {
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 7; ++j) {
      M[i][j] = 0.0;
    }
  }
  M[0][0] = g_mb;
  M[1][1] = g_mb;
  M[2][2] = g_mb;
  M[3][3] = g_I;
  M[4][4] = g_mp;
  M[5][5] = g_mp;
  M[6][6] = g_mp;
}

inline void coriolis_gravity(double C[7]) {
  C[0] = 0.0;
  C[1] = 0.0;
  C[2] = g_mb * g_gravity;
  C[3] = 0.0;
  C[4] = 0.0;
  C[5] = 0.0;
  C[6] = 0.0;
}

inline void rotation_pitch(double a, double R[3][3]) {
  const double c = std::cos(a);
  const double s = std::sin(a);
  R[0][0] = c;
  R[0][1] = 0.0;
  R[0][2] = s;
  R[1][0] = 0.0;
  R[1][1] = 1.0;
  R[1][2] = 0.0;
  R[2][0] = -s;
  R[2][1] = 0.0;
  R[2][2] = c;
}

inline void mat3_vec3(const double A[3][3], const double x[3], double y[3]) {
  for (int i = 0; i < 3; ++i) {
    y[i] = A[i][0] * x[0] + A[i][1] * x[1] + A[i][2] * x[2];
  }
}

inline void mat3T_vec3(const double A[3][3], const double x[3], double y[3]) {
  for (int i = 0; i < 3; ++i) {
    y[i] = A[0][i] * x[0] + A[1][i] * x[1] + A[2][i] * x[2];
  }
}

inline void signed_distance(const double q[7], double phi[5]) {
  const double box_pos[3] = {q[0], q[1], q[2]};
  const double pusher_pos[3] = {q[4], q[5], q[6]};

  double R[3][3];
  rotation_pitch(q[3], R);

  const double corners[4][3] = {
      {+g_box_half_width, 0.0, -g_box_half_height},
      {-g_box_half_width, 0.0, -g_box_half_height},
      {+g_box_half_width, 0.0, +g_box_half_height},
      {-g_box_half_width, 0.0, +g_box_half_height},
  };
  double cw[3];
  for (int i = 0; i < 4; ++i) {
    mat3_vec3(R, corners[i], cw);
    phi[i] = box_pos[2] + cw[2];
  }

  const double d[3] = {
      pusher_pos[0] - box_pos[0],
      pusher_pos[1] - box_pos[1],
      pusher_pos[2] - box_pos[2],
  };
  double p_local[3];
  mat3T_vec3(R, d, p_local);

  const double a = g_box_half_width + g_pusher_radius;
  const double b = g_box_half_height + g_pusher_radius;
  const double nx = p_local[0] / a;
  const double nz = p_local[2] / b;
  const double nx2 = nx * nx;
  const double nz2 = nz * nz;
  const double nx10 = nx2 * nx2 * nx2 * nx2 * nx2;
  const double nz10 = nz2 * nz2 * nz2 * nz2 * nz2;
  const double base = nx10 + nz10 + 1e-12;
  const double lp = std::pow(base, 0.1);
  phi[4] = lp - 1.0;
}

inline void contact_jacobian(const double q[7], double J[5][7]) {
  double qtmp[7];
  for (int i = 0; i < 7; ++i) {
    qtmp[i] = q[i];
  }

  const double eps = 1e-6;
  double phi_p[5];
  double phi_m[5];
  for (int j = 0; j < 7; ++j) {
    qtmp[j] = q[j] + eps;
    signed_distance(qtmp, phi_p);
    qtmp[j] = q[j] - eps;
    signed_distance(qtmp, phi_m);
    qtmp[j] = q[j];
    for (int i = 0; i < 5; ++i) {
      J[i][j] = (phi_p[i] - phi_m[i]) / (2.0 * eps);
    }
  }
}

inline void p_function(const double q[7], double P[10][7]) {
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 7; ++j) {
      P[i][j] = 0.0;
    }
  }

  double R[3][3];
  rotation_pitch(q[3], R);

  const double corners[4][3] = {
      {+g_box_half_width, 0.0, -g_box_half_height},
      {-g_box_half_width, 0.0, -g_box_half_height},
      {+g_box_half_width, 0.0, +g_box_half_height},
      {-g_box_half_width, 0.0, +g_box_half_height},
  };

  // Floor contacts: tangential +/- x on box.
  for (int ci = 0; ci < 4; ++ci) {
    double r[3];
    mat3_vec3(R, corners[ci], r);
    for (int k = 0; k < 2; ++k) {
      const double sgn = (k == 0) ? 1.0 : -1.0;
      const int row = 2 * ci + k;
      const double tx = sgn;
      const double tz = 0.0;
      const double tau_y = r[2] * tx - r[0] * tz;
      P[row][0] = tx;
      P[row][3] = tau_y;
    }
  }

  // Pusher contact tangent in x-z plane.
  double J[5][7];
  contact_jacobian(q, J);
  const double nx = J[4][4];
  const double nz = J[4][6];
  const double n_norm = std::sqrt(nx * nx + nz * nz + 1e-12);
  const double tx0 = -nz / n_norm;
  const double tz0 = nx / n_norm;

  const double r_bp_x = q[4] - q[0];
  const double r_bp_z = q[6] - q[2];

  for (int k = 0; k < 2; ++k) {
    const double sgn = (k == 0) ? 1.0 : -1.0;
    const double tx = sgn * tx0;
    const double tz = sgn * tz0;
    const double tau_y = -r_bp_z * tx + r_bp_x * tz;
    const int row = 8 + k;

    // on box
    P[row][0] = -tx;
    P[row][2] = -tz;
    P[row][3] = tau_y;
    // on pusher
    P[row][4] = tx;
    P[row][6] = tz;
  }
}

inline void residual_impl(const double* z, const double* theta, double kappa, double* out) {
  const double* q0 = theta + 0;
  const double* q1 = theta + 7;
  const double* u1 = theta + 14;
  const double h = theta[18];

  const double* q2 = z + 0;
  // z layout (TipOverPusher.indices_z):
  // q[0:7], gamma[7:12], b[12:22], psi[22:27], s_gamma[27:32], sb[32:42], spsi[42:47]
  const double* gamma1 = z + 7;
  const double* b1 = z + 12;
  const double* psi1 = z + 22;
  const double* sgamma1 = z + 27;
  const double* sb1 = z + 32;
  const double* spsi1 = z + 42;

  double v1[7];
  for (int i = 0; i < 7; ++i) {
    v1[i] = (q2[i] - q1[i]) / h;
  }

  double P_q2[10][7];
  p_function(q2, P_q2);
  double vT_stack[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 7; ++j) {
      vT_stack[i] += P_q2[i][j] * v1[j];
    }
  }

  // Match optimization_dynamics/src/models/tipover_push/model.jl:
  // dyn = ... + B*u + N(q2)^T * gamma + P(q2)^T * b
  double J_q2[5][7];
  contact_jacobian(q2, J_q2);

  double Lambda[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int i = 0; i < 7; ++i) {
    for (int c = 0; c < 5; ++c) {
      Lambda[i] += J_q2[c][i] * gamma1[c];
    }
    for (int r = 0; r < 10; ++r) {
      Lambda[i] += P_q2[r][i] * b1[r];
    }
  }

  double M[7][7];
  mass_diag(M);
  double C[7];
  coriolis_gravity(C);
  double B[7][2];
  input_jacobian(B);

  double vm1[7], vm2[7];
  for (int i = 0; i < 7; ++i) {
    vm1[i] = (q1[i] - q0[i]) / h;
    vm2[i] = (q2[i] - q1[i]) / h;
  }
  double D2L1[7], D2L2[7];
  matvec7(M, vm1, D2L1);
  matvec7(M, vm2, D2L2);

  double Bu[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int i = 0; i < 7; ++i) {
    Bu[i] = B[i][0] * u1[0] + B[i][1] * u1[1];
  }

  double dyn[7];
  for (int i = 0; i < 7; ++i) {
    const double D1L1 = -C[i];
    const double D1L2 = -C[i];
    dyn[i] = D2L1[i] - D2L2[i] + 0.5 * h * D1L1 + 0.5 * h * D1L2 + Bu[i] + Lambda[i];
  }

  double phi[5];
  signed_distance(q2, phi);

  const double E[5][10] = {
      {1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0},
  };

  double psi_stack[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int j = 0; j < 10; ++j) {
    for (int i = 0; i < 5; ++i) {
      psi_stack[j] += E[i][j] * psi1[i];
    }
  }

  double res_sd[5], res_vT[10], res_fric[5], res_ncomp[5], res_bcomp[10], res_psicomp[5];
  for (int i = 0; i < 5; ++i) {
    res_sd[i] = phi[i] - sgamma1[i];
  }
  for (int i = 0; i < 10; ++i) {
    res_vT[i] = sb1[i] - vT_stack[i] - psi_stack[i];
  }

  const double mu_contact[5] = {g_mu_floor, g_mu_floor, g_mu_floor, g_mu_floor, g_mu_pusher};
  for (int i = 0; i < 5; ++i) {
    double Eb = 0.0;
    for (int j = 0; j < 10; ++j) {
      Eb += E[i][j] * b1[j];
    }
    res_fric[i] = spsi1[i] - (mu_contact[i] * gamma1[i] - Eb);
    res_ncomp[i] = gamma1[i] * sgamma1[i] - kappa;
    res_psicomp[i] = psi1[i] * spsi1[i] - kappa;
  }
  for (int i = 0; i < 10; ++i) {
    res_bcomp[i] = b1[i] * sb1[i] - kappa;
  }

  int idx = 0;
  for (int i = 0; i < 7; ++i) {
    out[idx++] = dyn[i];
  }
  for (int i = 0; i < 5; ++i) {
    out[idx++] = res_sd[i];
  }
  for (int i = 0; i < 10; ++i) {
    out[idx++] = res_vT[i];
  }
  for (int i = 0; i < 5; ++i) {
    out[idx++] = res_fric[i];
  }
  for (int i = 0; i < 5; ++i) {
    out[idx++] = res_ncomp[i];
  }
  for (int i = 0; i < 10; ++i) {
    out[idx++] = res_bcomp[i];
  }
  for (int i = 0; i < 5; ++i) {
    out[idx++] = res_psicomp[i];
  }
}

}  // namespace

extern "C" {

void tipover_push_set_params(
    double mb,
    double mp,
    double I,
    double box_half_width,
    double box_half_height,
    double pusher_radius,
    double gravity,
    double mu_floor,
    double mu_pusher) {
  g_mb = mb;
  g_mp = mp;
  g_I = I;
  g_box_half_width = box_half_width;
  g_box_half_height = box_half_height;
  g_pusher_radius = pusher_radius;
  g_gravity = gravity;
  g_mu_floor = mu_floor;
  g_mu_pusher = mu_pusher;
}

void tipover_push_residual(const double* z, const double* theta, double kappa, double* out) {
  residual_impl(z, theta, kappa, out);
}

void tipover_push_rz(const double* z, const double* theta, double* out) {
  const double eps = 1e-6;
  double base[kNres] = {0.0};
  residual_impl(z, theta, 0.0, base);

  double z_pert[kNz];
  for (int i = 0; i < kNz; ++i) {
    z_pert[i] = z[i];
  }

  for (int col = 0; col < kNz; ++col) {
    z_pert[col] += eps;
    double r1[kNres] = {0.0};
    residual_impl(z_pert, theta, 0.0, r1);
    z_pert[col] = z[col];
    for (int row = 0; row < kNres; ++row) {
      out[row * kNz + col] = (r1[row] - base[row]) / eps;
    }
  }
}

void tipover_push_rtheta(const double* z, const double* theta, double* out) {
  const double eps = 1e-6;
  double base[kNres] = {0.0};
  residual_impl(z, theta, 0.0, base);

  double th_pert[kNtheta];
  for (int i = 0; i < kNtheta; ++i) {
    th_pert[i] = theta[i];
  }

  for (int col = 0; col < kNtheta; ++col) {
    th_pert[col] += eps;
    double r1[kNres] = {0.0};
    residual_impl(z, th_pert, 0.0, r1);
    th_pert[col] = theta[col];
    for (int row = 0; row < kNres; ++row) {
      out[row * kNtheta + col] = (r1[row] - base[row]) / eps;
    }
  }
}

}  // extern "C"
