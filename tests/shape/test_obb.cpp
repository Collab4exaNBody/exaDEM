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
#include <exaDEM/type/OBB.hpp>

#include <algorithm>
#include <random>

#include "unit_test.hpp"

using exanb::AABB;
using exanb::Quaternion;
using exanb::Vec3d;

namespace {
constexpr double tol = 1e-12;

// Rotation of angle theta (rad) around the normalized axis u.
Quaternion axis_angle(const Vec3d& u, double theta) {
  const double s = std::sin(0.5 * theta);
  return Quaternion{std::cos(0.5 * theta), s * u.x, s * u.y, s * u.z};
}

OBB make_obb(const Vec3d& center, const Vec3d& extent, const Quaternion& Q) {
  OBB obb;
  obb.e1 = Q * Vec3d{1, 0, 0};
  obb.e2 = Q * Vec3d{0, 1, 0};
  obb.e3 = Q * Vec3d{0, 0, 1};
  obb.center = center;
  obb.extent = extent;
  return obb;
}

OBB make_obb(const Vec3d& center, const Vec3d& extent) { return make_obb(center, extent, Quaternion{1, 0, 0, 0}); }

// Reference separating axis test written independently from OBB::intersect:
// returns the largest gap between the projections of A and B over the 15 candidate axes.
// gap > 0 means separated, gap < 0 means overlapping.
// min_cross_norm is the smallest |a_i x b_j|, i.e. the sine of the angle between the two most parallel edges.
double sat_reference_gap(const OBB& A, const OBB& B, bool* separated_only_by_cross_axis = nullptr,
                         double* min_cross_norm = nullptr) {
  const Vec3d a[3] = {A.e1, A.e2, A.e3};
  const Vec3d b[3] = {B.e1, B.e2, B.e3};
  const double ea[3] = {A.extent.x, A.extent.y, A.extent.z};
  const double eb[3] = {B.extent.x, B.extent.y, B.extent.z};
  const Vec3d t = B.center - A.center;

  auto gap_along = [&](Vec3d L) -> double {
    const double len = exanb::norm(L);
    L = L / len;
    double ra = 0, rb = 0;
    for (int i = 0; i < 3; i++) {
      ra += ea[i] * std::fabs(exanb::dot(a[i], L));
      rb += eb[i] * std::fabs(exanb::dot(b[i], L));
    }
    return std::fabs(exanb::dot(t, L)) - (ra + rb);
  };

  double face_gap = -1e30;
  for (int i = 0; i < 3; i++) {
    face_gap = std::max(face_gap, gap_along(a[i]));
    face_gap = std::max(face_gap, gap_along(b[i]));
  }
  double cross_gap = -1e30;
  double min_norm = 1.0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      const Vec3d L = exanb::cross(a[i], b[j]);
      min_norm = std::min(min_norm, exanb::norm(L));
      if (exanb::norm(L) > 1e-6) {  // parallel edges: axis already covered by face axes
        cross_gap = std::max(cross_gap, gap_along(L));
      }
    }
  }
  if (separated_only_by_cross_axis) {
    *separated_only_by_cross_axis = (face_gap < 0) && (cross_gap > 0);
  }
  if (min_cross_norm) {
    *min_cross_norm = min_norm;
  }
  return std::max(face_gap, cross_gap);
}

Quaternion random_quaternion(std::mt19937& gen) {
  std::normal_distribution<double> gauss(0.0, 1.0);
  Quaternion q{gauss(gen), gauss(gen), gauss(gen), gauss(gen)};
  const double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  return Quaternion{q.w / n, q.x / n, q.y / n, q.z / n};
}
}  // namespace

// ============================================================
// === Basic operations
// ============================================================

UNIT_TEST(default_constructor) {
  OBB obb;
  CHECK_VEC3_NEAR(obb.center, (Vec3d{0, 0, 0}), 0.0);
  CHECK_VEC3_NEAR(obb.e1, (Vec3d{1, 0, 0}), 0.0);
  CHECK_VEC3_NEAR(obb.e2, (Vec3d{0, 1, 0}), 0.0);
  CHECK_VEC3_NEAR(obb.e3, (Vec3d{0, 0, 1}), 0.0);
  CHECK_VEC3_NEAR(obb.extent, (Vec3d{0, 0, 0}), 0.0);
}

UNIT_TEST(enlarge) {
  OBB obb = make_obb({1, 2, 3}, {0.5, 1.0, 1.5});
  obb.enlarge(0.25);
  CHECK_VEC3_NEAR(obb.extent, (Vec3d{0.75, 1.25, 1.75}), tol);
  CHECK_VEC3_NEAR(obb.center, (Vec3d{1, 2, 3}), 0.0);  // center is not modified
}

UNIT_TEST(translate) {
  OBB obb = make_obb({1, 2, 3}, {1, 1, 1});
  obb.translate({-1, 0.5, 2});
  CHECK_VEC3_NEAR(obb.center, (Vec3d{0, 2.5, 5}), tol);
  CHECK_VEC3_NEAR(obb.e1, (Vec3d{1, 0, 0}), 0.0);  // axes are not modified
  CHECK_VEC3_NEAR(obb.extent, (Vec3d{1, 1, 1}), 0.0);
}

UNIT_TEST(rotate_in_place_rotates_axes_and_center) {
  // 90 degrees around z: x -> y, y -> -x
  OBB obb = make_obb({2, 0, 0}, {1, 2, 3});
  obb.rotate_in_place(axis_angle({0, 0, 1}, M_PI / 2));
  CHECK_VEC3_NEAR(obb.e1, (Vec3d{0, 1, 0}), tol);
  CHECK_VEC3_NEAR(obb.e2, (Vec3d{-1, 0, 0}), tol);
  CHECK_VEC3_NEAR(obb.e3, (Vec3d{0, 0, 1}), tol);
  CHECK_VEC3_NEAR(obb.center, (Vec3d{0, 2, 0}), tol);  // the center is rotated around the origin
  CHECK_VEC3_NEAR(obb.extent, (Vec3d{1, 2, 3}), 0.0);
}

UNIT_TEST(rotate_in_place_keeps_orthonormal_axes) {
  std::mt19937 gen(42);
  for (int k = 0; k < 100; k++) {
    OBB obb = make_obb({1, -2, 3}, {1, 2, 3});
    obb.rotate_in_place(random_quaternion(gen));
    CHECK_NEAR(exanb::norm(obb.e1), 1.0, 1e-12);
    CHECK_NEAR(exanb::norm(obb.e2), 1.0, 1e-12);
    CHECK_NEAR(exanb::norm(obb.e3), 1.0, 1e-12);
    CHECK_NEAR(exanb::dot(obb.e1, obb.e2), 0.0, 1e-12);
    CHECK_NEAR(exanb::dot(obb.e1, obb.e3), 0.0, 1e-12);
    CHECK_NEAR(exanb::dot(obb.e2, obb.e3), 0.0, 1e-12);
    CHECK_NEAR(exanb::norm(obb.center), std::sqrt(14.0), 1e-12);
  }
}

// ============================================================
// === Point inclusion
// ============================================================

UNIT_TEST(intersect_point_axis_aligned) {
  const OBB obb = make_obb({1, 1, 1}, {1, 2, 3});
  CHECK(obb.intersect(Vec3d{1, 1, 1}));        // center
  CHECK(obb.intersect(Vec3d{1.9, 2.9, 3.9}));  // inside, near a corner
  CHECK(obb.intersect(Vec3d{2, 1, 1}));        // on a face (inclusive)
  CHECK(!obb.intersect(Vec3d{2.01, 1, 1}));    // just outside along x
  CHECK(!obb.intersect(Vec3d{1, 3.01, 1}));    // just outside along y
  CHECK(!obb.intersect(Vec3d{1, 1, -2.01}));   // just outside along z
}

UNIT_TEST(intersect_point_rotated) {
  // unit cube rotated by 45 degrees around z: its corners reach x = +/- sqrt(2)
  const OBB obb = make_obb({0, 0, 0}, {1, 1, 1}, axis_angle({0, 0, 1}, M_PI / 4));
  CHECK(obb.intersect(Vec3d{1.4, 0, 0}));
  CHECK(!obb.intersect(Vec3d{1.42, 0, 0}));
  CHECK(!obb.intersect(Vec3d{0.9, 0.9, 0}));  // inside the axis-aligned cube, outside the rotated one
}

// ============================================================
// === OBB / OBB intersection
// ============================================================

UNIT_TEST(intersect_obb_axis_aligned) {
  const OBB A = make_obb({0, 0, 0}, {1, 1, 1});
  CHECK(A.intersect(A));                                            // itself
  CHECK(A.intersect(make_obb({1.5, 0, 0}, {1, 1, 1})));             // overlap
  CHECK(A.intersect(make_obb({0, 0, 0}, {0.1, 0.1, 0.1})));         // contained
  CHECK(A.intersect(make_obb({0, 0, 0}, {5, 5, 5})));               // container
  CHECK(!A.intersect(make_obb({2.1, 0, 0}, {1, 1, 1})));            // separated along x
  CHECK(!A.intersect(make_obb({0, -2.1, 0}, {1, 1, 1})));           // separated along y
  CHECK(!A.intersect(make_obb({0, 0, 2.1}, {1, 1, 1})));            // separated along z
  CHECK(!A.intersect(make_obb({1.5, 1.5, 1.5}, {0.4, 0.4, 0.4})));  // near the corner, outside
}

UNIT_TEST(intersect_obb_touching_faces) {
  // boxes sharing a face are considered intersecting
  const OBB A = make_obb({0, 0, 0}, {1, 1, 1});
  CHECK(A.intersect(make_obb({2, 0, 0}, {1, 1, 1})));
}

UNIT_TEST(intersect_obb_rotated) {
  const OBB A = make_obb({0, 0, 0}, {1, 1, 1});
  const Quaternion Q = axis_angle({0, 0, 1}, M_PI / 4);
  // B is a unit cube rotated by 45 degrees, its corner reaches center.x - sqrt(2)
  CHECK(A.intersect(make_obb({2.3, 0, 0}, {1, 1, 1}, Q)));   // 2.3 - 1.414 < 1
  CHECK(!A.intersect(make_obb({2.5, 0, 0}, {1, 1, 1}, Q)));  // 2.5 - 1.414 > 1
}

UNIT_TEST(intersect_obb_is_symmetric) {
  std::mt19937 gen(1234);
  std::uniform_real_distribution<double> pos(-3.0, 3.0);
  std::uniform_real_distribution<double> ext(0.1, 1.5);
  for (int k = 0; k < 2000; k++) {
    const OBB A = make_obb({pos(gen), pos(gen), pos(gen)}, {ext(gen), ext(gen), ext(gen)}, random_quaternion(gen));
    const OBB B = make_obb({pos(gen), pos(gen), pos(gen)}, {ext(gen), ext(gen), ext(gen)}, random_quaternion(gen));
    CHECK(A.intersect(B) == B.intersect(A));
  }
}

UNIT_TEST(intersect_obb_matches_reference_sat) {
  std::mt19937 gen(2026);
  std::uniform_real_distribution<double> pos(-3.0, 3.0);
  std::uniform_real_distribution<double> ext(0.1, 1.5);
  // Configurations too close to contact are skipped, as well as nearly parallel edges: OBB::intersect
  // does not normalize the cross axes, so its tolerance term is amplified by 1/sin(angle).
  constexpr double margin = 1e-4;
  constexpr double min_sine = 0.05;
  int n_checked = 0;
  int n_intersect = 0;

  for (int k = 0; k < 20000; k++) {
    const OBB A = make_obb({pos(gen), pos(gen), pos(gen)}, {ext(gen), ext(gen), ext(gen)}, random_quaternion(gen));
    const OBB B = make_obb({pos(gen), pos(gen), pos(gen)}, {ext(gen), ext(gen), ext(gen)}, random_quaternion(gen));
    double min_cross_norm;
    const double gap = sat_reference_gap(A, B, nullptr, &min_cross_norm);
    if (std::fabs(gap) < margin || min_cross_norm < min_sine) {
      continue;
    }
    const bool expected = gap < 0;
    CHECK(A.intersect(B) == expected);
    n_checked++;
    n_intersect += expected;
  }

  // make sure that the random sampling covers both cases
  CHECK(n_checked > 18000);
  CHECK(n_intersect > 500);
  CHECK(n_checked - n_intersect > 500);
}

UNIT_TEST(intersect_obb_separated_by_edge_edge_axis) {
  // Two thin rods: A along x, B along (0,1,1)/sqrt(2) and twisted by 45 degrees around its own axis.
  // All 6 face axes overlap, only the axis x ^ B.e1 = (0,-1,1)/sqrt(2) can separate them:
  // gap along this axis = h / sqrt(2) - 2 * 0.1 * sqrt(2).
  const double s = std::sqrt(0.5);
  const OBB A = make_obb({0, 0, 0}, {2, 0.1, 0.1});
  auto make_B = [s](double h) {
    OBB B;
    B.e1 = {0, s, s};
    B.e2 = {s, -0.5, 0.5};
    B.e3 = {s, 0.5, -0.5};
    B.extent = {2, 0.1, 0.1};
    B.center = {0, 0, h};
    return B;
  };

  bool edge_edge = false;
  const OBB B_far = make_B(0.6);
  CHECK(sat_reference_gap(A, B_far, &edge_edge) > 0);
  CHECK(edge_edge);  // the configuration really tests an edge x edge axis
  CHECK(!A.intersect(B_far));
  CHECK(!B_far.intersect(A));

  const OBB B_close = make_B(0.2);
  CHECK(sat_reference_gap(A, B_close) < 0);
  CHECK(A.intersect(B_close));
  CHECK(B_close.intersect(A));
}

UNIT_TEST(intersect_obb_enlarge_creates_contact) {
  const OBB A = make_obb({0, 0, 0}, {1, 1, 1});
  OBB B = make_obb({2.5, 0, 0}, {1, 1, 1});
  CHECK(!A.intersect(B));
  B.enlarge(0.6);
  CHECK(A.intersect(B));
}

// ============================================================
// === AABB conversions
// ============================================================

UNIT_TEST(conv_to_obb) {
  const AABB aabb = {Vec3d{-1, 0, 2}, Vec3d{3, 1, 6}};
  const OBB obb = conv_to_obb(aabb);
  CHECK_VEC3_NEAR(obb.center, (Vec3d{1, 0.5, 4}), tol);
  CHECK_VEC3_NEAR(obb.extent, (Vec3d{2, 0.5, 2}), tol);
  CHECK_VEC3_NEAR(obb.e1, (Vec3d{1, 0, 0}), 0.0);
  CHECK_VEC3_NEAR(obb.e2, (Vec3d{0, 1, 0}), 0.0);
  CHECK_VEC3_NEAR(obb.e3, (Vec3d{0, 0, 1}), 0.0);
}

UNIT_TEST(conv_to_aabb_round_trip) {
  const AABB aabb = {Vec3d{-1, 0, 2}, Vec3d{3, 1, 6}};
  const AABB res = conv_to_aabb(conv_to_obb(aabb));
  CHECK_VEC3_NEAR(res.bmin, aabb.bmin, tol);
  CHECK_VEC3_NEAR(res.bmax, aabb.bmax, tol);
}

UNIT_TEST(conv_to_aabb_rotated) {
  // unit cube rotated by 45 degrees around z
  const OBB obb = make_obb({1, 2, 3}, {1, 1, 1}, axis_angle({0, 0, 1}, M_PI / 4));
  const AABB res = conv_to_aabb(obb);
  const double s = std::sqrt(2.0);
  CHECK_VEC3_NEAR(res.bmin, (Vec3d{1 - s, 2 - s, 2}), 1e-12);
  CHECK_VEC3_NEAR(res.bmax, (Vec3d{1 + s, 2 + s, 4}), 1e-12);
}

UNIT_TEST(conv_to_aabb_contains_obb_corners) {
  std::mt19937 gen(7);
  std::uniform_real_distribution<double> pos(-3.0, 3.0);
  std::uniform_real_distribution<double> ext(0.1, 1.5);
  for (int k = 0; k < 200; k++) {
    const OBB obb = make_obb({pos(gen), pos(gen), pos(gen)}, {ext(gen), ext(gen), ext(gen)}, random_quaternion(gen));
    const AABB box = conv_to_aabb(obb);
    for (int c = 0; c < 8; c++) {
      const double sx = (c & 1) ? 1 : -1;
      const double sy = (c & 2) ? 1 : -1;
      const double sz = (c & 4) ? 1 : -1;
      const Vec3d corner =
          obb.center + sx * obb.extent.x * obb.e1 + sy * obb.extent.y * obb.e2 + sz * obb.extent.z * obb.e3;
      CHECK(corner.x >= box.bmin.x - tol && corner.x <= box.bmax.x + tol);
      CHECK(corner.y >= box.bmin.y - tol && corner.y <= box.bmax.y + tol);
      CHECK(corner.z >= box.bmin.z - tol && corner.z <= box.bmax.z + tol);
    }
  }
}

UNIT_TEST_MAIN()
