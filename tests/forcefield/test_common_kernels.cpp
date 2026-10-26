/*
   Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
#include <onika/math/basic_types_operators.h>

#include <exaDEM/forcefield/common_kernels.hpp>

#include "unit_test.hpp"

using namespace exaDEM;
using exanb::Vec3d;

namespace {
constexpr double tol = 1e-12;
}

// ============================================================
// === Effective quantities and damping
// ============================================================

UNIT_TEST(effective_mass) {
  CHECK_NEAR(compute_effective_mass(2.0, 6.0), 1.5, tol);
  CHECK_NEAR(compute_effective_mass(3.0, 3.0), 1.5, tol);  // m / 2 for equal masses
  CHECK_NEAR(compute_effective_mass(2.0, 6.0), compute_effective_mass(6.0, 2.0), 0.0);
}

UNIT_TEST(effective_radius) {
  CHECK_NEAR(compute_effective_radius(1.0, 3.0), 0.75, tol);
  CHECK_NEAR(compute_effective_radius(0.5, 0.5), 0.25, tol);
}

UNIT_TEST(damping_coefficient) {
  // damp = damp_rate * 2 * sqrt(kn * meff)
  CHECK_NEAR(compute_damp(0.5, 100.0, 4.0), 20.0, tol);
  CHECK_NEAR(compute_damp(0.0, 100.0, 4.0), 0.0, 0.0);
  CHECK_NEAR(compute_damp(1.0, 1000.0, 0.5), 2.0 * std::sqrt(500.0), tol);  // critical damping
}

// ============================================================
// === Relative velocity
// ============================================================

UNIT_TEST(relative_velocity_translation) {
  const Vec3d zero = {0, 0, 0};
  const Vec3d v = compute_relative_velocity({1, 0, 0}, {0, 0, 0}, {1, 2, 3}, zero, {2, 0, 0}, {4, 4, 4}, zero);
  CHECK_VEC3_NEAR(v, (Vec3d{3, 2, 1}), tol);  // v_j - v_i
}

UNIT_TEST(relative_velocity_rotation) {
  // particle i spins around z at the origin: the velocity of the contact point (1,0,0) is w x r = (0,1,0)
  const Vec3d zero = {0, 0, 0};
  const Vec3d v = compute_relative_velocity({1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {0, 0, 1}, {2, 0, 0}, zero, zero);
  CHECK_VEC3_NEAR(v, (Vec3d{-1, -1, 0}), tol);
  // same rotation applied to j (contact point at -r from j's center)
  const Vec3d w = compute_relative_velocity({1, 0, 0}, {0, 0, 0}, zero, zero, {2, 0, 0}, zero, {0, 0, 1});
  CHECK_VEC3_NEAR(w, (Vec3d{0, -1, 0}), tol);
}

// ============================================================
// === Normal force
// ============================================================

UNIT_TEST(normal_force_elastic_and_damping) {
  // dn < 0: overlap, vn > 0: approaching
  CHECK_NEAR(normal_force(1000.0, 0.0, -0.01, 0.0), 10.0, tol);
  CHECK_NEAR(normal_force(1000.0, 10.0, -0.01, 0.5), 15.0, tol);
  CHECK_NEAR(normal_force(1000.0, 10.0, -0.01, -0.5), 5.0, tol);
}

UNIT_TEST(normal_force_never_attractive) {
  CHECK_NEAR(normal_force(1000.0, 10.0, -0.01, -2.0), 0.0, 0.0);  // separating fast
  CHECK_NEAR(normal_force(1000.0, 0.0, 0.01, 0.0), 0.0, 0.0);     // no overlap
}

UNIT_TEST(normal_force_vector) {
  const Vec3d n = {0, 0.6, 0.8};
  const Vec3d fn = compute_normal_force<false>(1000.0, 2.0, 10.0, -0.01, 0.5, n);
  CHECK_VEC3_NEAR(fn, 15.0 * n, tol);  // fc is ignored without cohesion
}

UNIT_TEST(normal_force_cohesive) {
  const Vec3d n = {1, 0, 0};
  CHECK_VEC3_NEAR(compute_normal_force<true>(1000.0, 2.0, 10.0, -0.01, 0.5, n), (Vec3d{13, 0, 0}), tol);
  // repulsion vanishes: only the cohesive force remains
  CHECK_VEC3_NEAR(compute_normal_force<true>(1000.0, 2.0, 10.0, -0.01, -2.0, n), (Vec3d{-2, 0, 0}), tol);
}

UNIT_TEST(normal_force_value_with_cohesive_force) {
  CHECK_NEAR(compute_normal_force_value_with_cohesive_force(1000.0, 2.0, 10.0, -0.01, 0.5), 13.0, tol);
  CHECK_NEAR(compute_normal_force_value_with_cohesive_force(1000.0, 2.0, 10.0, -0.01, -2.0), -2.0, tol);  // >= -fc
}

// ============================================================
// === Tangential force and Coulomb criterion
// ============================================================

UNIT_TEST(tangential_force) {
  const Vec3d n = {1, 0, 0};
  const Vec3d vel = {3, 4, 0};
  const double vn = exanb::dot(vel, n);
  CHECK_VEC3_NEAR(compute_tangential_force(0.1, vn, n, vel), (Vec3d{0, 0.4, 0}), tol);        // dt * vt
  CHECK_VEC3_NEAR(compute_tangential_force(100.0, 0.1, vn, n, vel), (Vec3d{0, 40, 0}), tol);  // kt * dt * vt
}

UNIT_TEST(tangential_force_is_tangential) {
  const Vec3d n = {0, 0.6, 0.8};
  const Vec3d vel = {1, -2, 5};
  const Vec3d ft = compute_tangential_force(10.0, 0.01, exanb::dot(vel, n), n, vel);
  CHECK_NEAR(exanb::dot(ft, n), 0.0, tol);
}

UNIT_TEST(coulomb_threshold) {
  CHECK_NEAR(compute_threshold_ft(0.5, 1000.0, -0.01), 5.0, tol);  // mu * kn * |dn|
  CHECK_NEAR(compute_threshold_ft(0.0, 1000.0, -0.01), 0.0, 0.0);
  CHECK_NEAR(compute_threshold_ft_with_cohesive_force(0.5, 13.0, 2.0), 7.5, tol);  // mu * (fn + fc)
}

UNIT_TEST(fit_tangential_force) {
  Vec3d ft = {3, 4, 0};  // |ft| = 5
  fit_tangential_force(5.0, ft);
  CHECK_VEC3_NEAR(ft, (Vec3d{3, 4, 0}), tol);  // on the threshold: unchanged

  ft = {6, 8, 0};
  fit_tangential_force(5.0, ft);
  CHECK_VEC3_NEAR(ft, (Vec3d{3, 4, 0}), tol);  // projected on the Coulomb cone, same direction

  ft = {1, 1, 1};
  fit_tangential_force(0.0, ft);
  CHECK_VEC3_NEAR(ft, (Vec3d{0, 0, 0}), tol);  // no friction

  ft = {0, 0, 0};
  fit_tangential_force(0.0, ft);
  CHECK_VEC3_NEAR(ft, (Vec3d{0, 0, 0}), 0.0);  // no division by zero
}

// ============================================================
// === Moments
// ============================================================

UNIT_TEST(moments) {
  // (contact_position - p) x f + m
  const Vec3d mom = compute_moments({1, 0, 0}, {0, 0, 0}, {0, 2, 0}, {0, 0, 1});
  CHECK_VEC3_NEAR(mom, (Vec3d{0, 0, 3}), tol);
  // a central force produces no moment
  const Vec3d central = compute_moments({1, 0, 0}, {0, 0, 0}, {-5, 0, 0}, {0, 0, 0});
  CHECK_VEC3_NEAR(central, (Vec3d{0, 0, 0}), tol);
}

UNIT_TEST_MAIN()
