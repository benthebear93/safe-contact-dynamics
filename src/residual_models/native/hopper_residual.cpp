// Four-coordinate hopper. Same midpoint dynamics and linear friction cone as
// src/robots/hopper/model_linear.py. Jacobians are exact derivatives, not FD.
// z = [q2(4), gamma, b+(1), b-(1), psi, s_gamma, s_b+, s_b-, s_psi]
// theta = [q0(4), q1(4), u(2), mu, h]; Jacobians use row-major storage.
#include <algorithm>
#include <cmath>

namespace {
constexpr int n = 12;

// Parameters are passed per call so independent model instances do not share
// mutable library state: [mb, ml, jb, jl, gravity].
void masses(const double* p, double* m) {
  m[0] = p[0] + p[1];
  m[1] = m[0];
  m[2] = p[2] + p[3];
  m[3] = p[1];
}

void geometry(const double* q, double* N, double* P) {
  const double s = std::sin(q[2]), c = std::cos(q[2]);
  N[0] = 0.0; N[1] = 1.0; N[2] = q[3] * s; N[3] = -c;
  P[0] = 1.0; P[1] = 0.0; P[2] = q[3] * c; P[3] = s;
}
}  // namespace

extern "C" {

void hopper_residual(const double* p, const double* z, const double* t,
                     double kappa, double* out) {
  const double h = t[11], a = (t[6] + z[2]) / 2.0;
  double m[4], N[4], P[4], v[4];
  masses(p, m);
  geometry(z, N, P);
  const double B[4][2] = {{0.0, -std::sin(a)}, {0.0, std::cos(a)},
                          {1.0, 0.0}, {0.0, 1.0}};
  for (int i = 0; i < 4; ++i) {
    v[i] = (z[i] - t[4 + i]) / h;
    const double vm1 = (t[4 + i] - t[i]) / h;
    const double D1L = i == 1 ? -(m[1] * p[4]) : -0.0;
    const double impulse = N[i] * z[4] + (P[i] * z[5] + (-P[i]) * z[6]);
    out[i] = m[i] * vm1 - m[i] * v[i]
             + 0.5 * h * D1L + 0.5 * h * D1L
             + (B[i][0] * t[8] + B[i][1] * t[9]) + impulse;
  }
  double vp = 0.0, vn = 0.0;
  for (int i = 0; i < 4; ++i) {
    vp += P[i] * v[i];
    vn += (-P[i]) * v[i];
  }
  out[4] = (z[1] - z[3] * std::cos(z[2])) - z[8];
  out[5] = z[9] - vp - z[7];
  out[6] = z[10] - vn - z[7];
  out[7] = z[11] - (t[10] * z[4] - (z[5] + z[6]));
  out[8] = z[4] * z[8] - kappa;
  out[9] = z[5] * z[9] - kappa;
  out[10] = z[6] * z[10] - kappa;
  out[11] = z[7] * z[11] - kappa;
}

void hopper_rz(const double* p, const double* z, const double* t, double* out) {
  std::fill(out, out + n * n, 0.0);
  const double h = t[11], s = std::sin(z[2]), c = std::cos(z[2]);
  const double a = (t[6] + z[2]) / 2.0;
  double m[4], N[4], P[4];
  masses(p, m);
  geometry(z, N, P);
  for (int i = 0; i < 4; ++i) {
    out[i * n + i] = -m[i] / h;
    out[i * n + 4] = N[i];
    out[i * n + 5] = P[i];
    out[i * n + 6] = -P[i];
  }
  out[0 * n + 2] += -std::cos(a) * t[9] * 0.5;
  out[1 * n + 2] += -std::sin(a) * t[9] * 0.5;
  out[2 * n + 2] += z[3] * c * z[4] - z[3] * s * z[5] + z[3] * s * z[6];
  out[3 * n + 2] += s * z[4] + c * z[5] - c * z[6];
  out[2 * n + 3] += s * z[4] + c * z[5] - c * z[6];
  for (int i = 0; i < 4; ++i) {
    out[4 * n + i] = N[i];
    out[5 * n + i] = -P[i] / h;
    out[6 * n + i] = P[i] / h;
  }
  const double va = (z[2] - t[6]) / h, vl = (z[3] - t[7]) / h;
  const double dtheta = -z[3] * s * va + c * vl;
  const double dlength = c * va;
  out[5 * n + 2] -= dtheta;
  out[6 * n + 2] += dtheta;
  out[5 * n + 3] -= dlength;
  out[6 * n + 3] += dlength;
  out[4 * n + 8] = -1.0;
  out[5 * n + 7] = out[6 * n + 7] = -1.0;
  out[5 * n + 9] = out[6 * n + 10] = 1.0;
  out[7 * n + 4] = -t[10];
  out[7 * n + 5] = out[7 * n + 6] = out[7 * n + 11] = 1.0;
  for (int i = 0; i < 4; ++i) {
    out[(8 + i) * n + (4 + i)] = z[8 + i];
    out[(8 + i) * n + (8 + i)] = z[4 + i];
  }
}

void hopper_rtheta(const double* p, const double* z, const double* t, double* out) {
  std::fill(out, out + n * n, 0.0);
  const double h = t[11], a = (t[6] + z[2]) / 2.0;
  double m[4], N[4], P[4];
  masses(p, m);
  geometry(z, N, P);
  double vp = 0.0, vn = 0.0;
  for (int i = 0; i < 4; ++i) {
    out[i * n + i] = -m[i] / h;
    out[i * n + 4 + i] = m[i] / h + m[i] / h;
    const double vm1 = (t[4 + i] - t[i]) / h;
    const double vm2 = (z[i] - t[4 + i]) / h;
    out[i * n + 11] = -m[i] * vm1 / h + m[i] * vm2 / h;
    out[5 * n + 4 + i] = P[i] / h;
    out[6 * n + 4 + i] = -P[i] / h;
    vp += P[i] * vm2;
    vn += (-P[i]) * vm2;
  }
  out[0 * n + 6] += -std::cos(a) * t[9] * 0.5;
  out[1 * n + 6] += -std::sin(a) * t[9] * 0.5;
  out[1 * n + 11] -= m[1] * p[4];
  out[0 * n + 9] = -std::sin(a);
  out[1 * n + 9] = std::cos(a);
  out[2 * n + 8] = out[3 * n + 9] = 1.0;
  out[5 * n + 11] = vp / h;
  out[6 * n + 11] = vn / h;
  out[7 * n + 10] = -z[4];
}

}  // extern "C"
