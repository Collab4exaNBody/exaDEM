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
// Note: CylinderFields::axis_ is a component mask, not a direction: {1, 0, 1} is a cylinder along y.
#include "driver_test.hpp"

using namespace exaDEM;
using exaDEM::unit_test::contains;
using exaDEM::unit_test::is_finite;
using exanb::Vec3d;

namespace {
constexpr double tol = 1e-12;

// Cylinder of radius 1 along y (mask {1, 0, 1}), axis through the origin.
Cylinder make_cylinder(MotionType motion = STATIONARY, const Vec3d& mask = {1, 0, 1}) {
  Cylinder cyl;
  cyl.fields_.radius_ = 1.0;
  cyl.fields_.axis_ = mask;
  cyl.fields_.center_ = {0, 0, 0};
  cyl.motion_type_ = motion;
  return cyl;
}
}  // namespace

// ============================================================
// === Contact detection (particles inside the cylinder)
// ============================================================

UNIT_TEST(detector_contact) {
  Cylinder cyl = make_cylinder();
  // vertex of radius 0.2 at 0.9 from the axis: overlap of 0.1 with the wall, the y coordinate is ignored
  auto [contact, dn, n, pos] = cyl.detector(0.2, Vec3d{0.9, 5, 0});
  CHECK(contact);
  CHECK_NEAR(dn, -0.1, tol);
  CHECK_VEC3_NEAR(n, (Vec3d{-1, 0, 0}), tol);      // toward the axis
  CHECK_VEC3_NEAR(pos, (Vec3d{1.05, 5, 0}), tol);  // middle of the overlap
}

UNIT_TEST(detector_no_contact) {
  Cylinder cyl = make_cylinder();
  auto [contact, dn, n, pos] = cyl.detector(0.2, Vec3d{0, -3, 0.5});
  CHECK(!contact);
  CHECK_NEAR(dn, 0.3, tol);
}

UNIT_TEST(detector_shifted_axis) {
  Cylinder cyl = make_cylinder();
  cyl.fields_.center_ = {2, 0, 3};
  auto [contact, dn, n, pos] = cyl.detector(0.1, Vec3d{2, 7, 3.95});
  CHECK(contact);
  CHECK_NEAR(dn, -0.05, tol);
  CHECK_VEC3_NEAR(n, (Vec3d{0, 0, -1}), tol);
}

UNIT_TEST(filter) {
  Cylinder cyl = make_cylinder();
  CHECK(cyl.filter(0.2, Vec3d{0.9, 0, 0}));
  CHECK(cyl.filter(0.2, Vec3d{0, 0, 1.5}));  // outside the cylinder
  CHECK(!cyl.filter(0.2, Vec3d{0.5, 100, 0}));
}

UNIT_TEST(initialize_projects_the_center) {
  Cylinder cyl = make_cylinder();
  cyl.fields_.center_ = {1, 2, 3};
  Driver_params motion;
  cyl.initialize(motion);
  CHECK_VEC3_NEAR(cyl.fields_.center_, (Vec3d{1, 0, 3}), 0.0);  // the coordinate along the axis is removed
}

// ============================================================
// === Normal (axis direction used by the paraview output)
// ============================================================

UNIT_TEST(compute_normal_y_and_z) {
  Cylinder cyl_y = make_cylinder(STATIONARY, {1, 0, 1});
  CHECK_NEAR(std::fabs(exanb::dot(cyl_y.compute_normal(), Vec3d{0, 1, 0})), 1.0, tol);
  Cylinder cyl_z = make_cylinder(STATIONARY, {1, 1, 0});
  CHECK_NEAR(std::fabs(exanb::dot(cyl_z.compute_normal(), Vec3d{0, 0, 1})), 1.0, tol);
}

UNIT_TEST(axis_mask_validation) {
  CHECK(Cylinder::is_valid_axis_mask({0, 1, 1}));
  CHECK(Cylinder::is_valid_axis_mask({1, 0, 1}));
  CHECK(Cylinder::is_valid_axis_mask({1, 1, 0}));
  CHECK(!Cylinder::is_valid_axis_mask({0, 0, 1}));  // a direction, not a mask
  CHECK(!Cylinder::is_valid_axis_mask({1, 1, 1}));
  CHECK(!Cylinder::is_valid_axis_mask({0, 0, 0}));
  CHECK(!Cylinder::is_valid_axis_mask({0.6, 0, 0.8}));
  CHECK(!Cylinder::is_valid_axis_mask({2, 0, 2}));
}

UNIT_TEST(compute_normal_x) {
  Cylinder cyl_x = make_cylinder(STATIONARY, {0, 1, 1});
  const Vec3d n = cyl_x.compute_normal();
  CHECK(is_finite(n));
  CHECK_NEAR(std::fabs(exanb::dot(n, Vec3d{1, 0, 0})), 1.0, tol);
}

// ============================================================
// === Motion
// ============================================================

UNIT_TEST(stationary_does_not_move) {
  Cylinder cyl = make_cylinder();
  Driver_params motion;
  cyl.push_f_v(motion, 0.1);
  cyl.push_f_v_r(motion, 0.0, 0.1);
  CHECK_VEC3_NEAR(cyl.fields_.center_, (Vec3d{0, 0, 0}), 0.0);
}

UNIT_TEST(shaker) {
  Cylinder cyl = make_cylinder(SHAKER);
  Driver_params motion;
  motion.amplitude_ = 0.1;
  motion.omega_ = 2.0;
  motion.shaker_dir_ = {1, 0, 0};
  const double dt = 0.01;
  double time = 0;
  for (int i = 0; i < 50; i++) {
    cyl.push_f_v_r(motion, time, dt);
    time += dt;
  }
  // the displacement follows the signal exactly (no integration error)
  CHECK_VEC3_NEAR(cyl.fields_.center_, (Vec3d{0.1 * std::sin(2.0 * time), 0, 0}), 1e-12);
  CHECK_VEC3_NEAR(cyl.fields_.vel_, (Vec3d{0.1 * 2.0 * std::cos(2.0 * time), 0, 0}), 1e-12);
}

// ============================================================
// === YAML / dump
// ============================================================

UNIT_TEST(yaml_fields) {
  const CylinderFields f =
      YAML::Load("{ radius: 2, axis: [1,1,0], center: [1,2,0], vrot: [0,0,1] }").as<CylinderFields>();
  CHECK_NEAR(f.radius_, 2.0, 0.0);
  CHECK_VEC3_NEAR(f.axis_, (Vec3d{1, 1, 0}), 0.0);
  CHECK_VEC3_NEAR(f.center_, (Vec3d{1, 2, 0}), 0.0);
  CHECK_VEC3_NEAR(f.vrot_, (Vec3d{0, 0, 1}), 0.0);
  CylinderFields g;
  CHECK(!YAML::convert<CylinderFields>::decode(YAML::Load("{ radius: 2, center: [0,0,0] }"), g));  // no axis
}

UNIT_TEST(dump_and_read_back) {
  Cylinder cyl = make_cylinder();
  cyl.fields_.radius_ = 3.5;
  cyl.fields_.center_ = {1, 0, -1};
  Driver_params motion;
  std::stringstream stream;
  cyl.dump_data(motion, 1, stream);
  CHECK(contains(stream.str(), "type: CYLINDER"));
  const CylinderFields f = YAML::Load(stream.str())[0]["state"].as<CylinderFields>();
  CHECK_NEAR(f.radius_, 3.5, 1e-6);
  CHECK_VEC3_NEAR(f.axis_, (Vec3d{1, 0, 1}), 1e-6);
  CHECK_VEC3_NEAR(f.center_, (Vec3d{1, 0, -1}), 1e-6);
}

UNIT_TEST_MAIN()
