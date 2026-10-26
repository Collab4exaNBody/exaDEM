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
#include "driver_test.hpp"

using namespace exaDEM;
using exaDEM::unit_test::contains;
using exanb::Vec3d;

namespace {
constexpr double tol = 1e-12;

// Plane z = offset, initialized (center_proj_ is set by initialize).
Surface make_surface(MotionType motion = STATIONARY, double offset = 1.0, Driver_params motion_params = {}) {
  Surface surf;
  surf.fields_.offset_ = offset;
  surf.fields_.normal_ = {0, 0, 1};
  surf.fields_.center_ = offset * surf.fields_.normal_;
  surf.motion_type_ = motion;
  surf.initialize(motion_params);
  return surf;
}
}  // namespace

// ============================================================
// === Contact detection (two-sided wall)
// ============================================================

UNIT_TEST(detector_above) {
  Surface surf = make_surface();
  auto [contact, dn, n, pos] = surf.detector(0.2, Vec3d{3, 4, 1.1});
  CHECK(contact);
  CHECK_NEAR(dn, -0.1, tol);
  CHECK_VEC3_NEAR(n, (Vec3d{0, 0, 1}), tol);       // from the wall to the vertex
  CHECK_VEC3_NEAR(pos, (Vec3d{3, 4, 0.95}), tol);  // middle of the overlap
}

UNIT_TEST(detector_below) {
  Surface surf = make_surface();
  auto [contact, dn, n, pos] = surf.detector(0.2, Vec3d{-3, 0, 0.9});
  CHECK(contact);
  CHECK_NEAR(dn, -0.1, tol);
  CHECK_VEC3_NEAR(n, (Vec3d{0, 0, -1}), tol);
  CHECK_VEC3_NEAR(pos, (Vec3d{-3, 0, 1.05}), tol);
}

UNIT_TEST(detector_no_contact) {
  Surface surf = make_surface();
  auto [contact, dn, n, pos] = surf.detector(0.2, Vec3d{0, 0, 2});
  CHECK(!contact);
  CHECK_NEAR(dn, 0.8, tol);
}

UNIT_TEST(detector_tilted_plane) {
  Surface surf;
  surf.fields_.normal_ = {0, std::sqrt(0.5), std::sqrt(0.5)};
  surf.fields_.offset_ = 0.0;
  surf.motion_type_ = STATIONARY;
  Driver_params motion;
  surf.initialize(motion);
  const Vec3d p = Vec3d{5, 0, 0} + 0.15 * surf.fields_.normal_;
  auto [contact, dn, n, pos] = surf.detector(0.2, p);
  CHECK(contact);
  CHECK_NEAR(dn, -0.05, tol);
  CHECK_VEC3_NEAR(n, surf.fields_.normal_, tol);
}

UNIT_TEST(filter) {
  Surface surf = make_surface();
  CHECK(surf.filter(0.2, Vec3d{100, -50, 1.1}));
  CHECK(surf.filter(0.2, Vec3d{0, 0, 0.85}));
  CHECK(!surf.filter(0.2, Vec3d{0, 0, 1.5}));
}

// ============================================================
// === Initialization
// ============================================================

UNIT_TEST(initialize_center_proj) {
  Surface surf = make_surface(STATIONARY, 2.5);
  CHECK_VEC3_NEAR(surf.fields_.center_proj_, (Vec3d{0, 0, 2.5}), 0.0);
}

UNIT_TEST(initialize_moves_the_center_on_the_plane) {
  // a center outside the plane is projected on it (its tangential coordinates are kept)
  for (double z : {2.0, -2.0}) {
    Surface surf;
    surf.fields_.offset_ = 0.0;
    surf.fields_.normal_ = {0, 0, 1};
    surf.fields_.center_ = {1, 0, z};
    surf.motion_type_ = STATIONARY;
    Driver_params motion;
    surf.initialize(motion);
    CHECK_VEC3_NEAR(surf.fields_.center_, (Vec3d{1, 0, 0}), tol);
  }
}

UNIT_TEST(initialize_aligns_motion_vector) {
  Driver_params motion;
  motion.motion_vector_ = {1, 0, 0};
  motion.const_vel_ = 1.0;
  Surface surf;
  surf.fields_.normal_ = {0, 0, 1};
  surf.motion_type_ = LINEAR_MOTION;
  surf.initialize(motion);
  CHECK_VEC3_NEAR(motion.motion_vector_, (Vec3d{0, 0, 1}), 0.0);  // forced along the normal

  Driver_params opposite;
  opposite.motion_vector_ = {0, 0, -1};
  opposite.const_vel_ = 1.0;
  Surface surf2;
  surf2.fields_.normal_ = {0, 0, 1};
  surf2.motion_type_ = LINEAR_MOTION;
  surf2.initialize(opposite);
  CHECK_VEC3_NEAR(opposite.motion_vector_, (Vec3d{0, 0, -1}), 0.0);  // -normal is accepted
}

// ============================================================
// === Motion
// ============================================================

UNIT_TEST(stationary_does_not_move) {
  Surface surf = make_surface();
  surf.fields_.vel_ = {0, 0, 1};
  Driver_params motion;
  surf.push_f_v(motion, 0.1);
  surf.push_f_v_r(motion, 0.0, 0.1);
  CHECK_VEC3_NEAR(surf.fields_.vel_, (Vec3d{0, 0, 0}), 0.0);
  CHECK_NEAR(surf.fields_.offset_, 1.0, 0.0);
}

UNIT_TEST(linear_motion) {
  Driver_params motion;
  motion.motion_vector_ = {0, 0, -1};
  motion.const_vel_ = 0.5;
  Surface surf = make_surface(LINEAR_MOTION, 1.0, motion);
  const double dt = 0.1;
  for (int i = 0; i < 10; i++) {
    surf.push_f_v(motion, dt);
    surf.push_f_v_r(motion, i * dt, dt);
  }
  CHECK_VEC3_NEAR(surf.fields_.vel_, (Vec3d{0, 0, -0.5}), tol);
  CHECK_NEAR(surf.fields_.offset_, 0.5, 1e-12);
  CHECK_VEC3_NEAR(surf.fields_.center_, (Vec3d{0, 0, 0.5}), 1e-12);
  CHECK_VEC3_NEAR(surf.fields_.center_proj_, (Vec3d{0, 0, 0.5}), 1e-12);
}

UNIT_TEST(compressive_force_to_accel) {
  // acc = (F - sigma * S * mv) . mv / (0.5 * m)
  Driver_params motion;
  motion.motion_vector_ = {0, 0, -1};
  motion.sigma_ = 2.0;
  motion.damprate_ = 0.5;
  Surface surf;
  surf.fields_.normal_ = {0, 0, 1};
  surf.fields_.offset_ = 1.0;
  surf.fields_.center_ = {0, 0, 1};
  surf.fields_.surface_ = 3.0;
  surf.fields_.mass_ = 4.0;
  surf.motion_type_ = LINEAR_COMPRESSIVE_MOTION;
  surf.initialize(motion);

  surf.fields_.forces_ = {1, 1, 10};  // particles push the wall upward
  surf.force_to_accel(motion);
  CHECK_NEAR(surf.fields_.acc_, (-10.0 - 2.0 * 3.0) / 2.0, tol);  // projected on (0,0,-1)

  surf.push_f_v(motion, 0.1);
  CHECK_VEC3_NEAR(surf.fields_.vel_, (Vec3d{0, 0, 0.8}), tol);  // dt * acc * mv
}

UNIT_TEST(compressive_uses_system_mass_by_default) {
  Driver_params motion;
  motion.motion_vector_ = {0, 0, 1};
  motion.sigma_ = 1.0;
  motion.system_mass_ = 10.0;
  Surface surf;
  surf.fields_.surface_ = 2.0;
  surf.motion_type_ = LINEAR_COMPRESSIVE_MOTION;
  surf.initialize(motion);
  surf.force_to_accel(motion);
  CHECK_NEAR(surf.fields_.acc_, -2.0 / (0.5 * 5.0), tol);  // m = system_mass / 2
}

UNIT_TEST(shaker) {
  Driver_params motion;
  motion.amplitude_ = 0.1;
  motion.omega_ = 5.0;
  motion.shaker_dir_ = {0, 0, 1};
  Surface surf = make_surface(SHAKER, 1.0, motion);
  const double dt = 0.01;
  double time = 0;
  for (int i = 0; i < 30; i++) {
    surf.push_f_v_r(motion, time, dt);
    time += dt;
  }
  CHECK_NEAR(surf.fields_.offset_, 1.0 + 0.1 * std::sin(5.0 * time), 1e-12);
  CHECK_VEC3_NEAR(surf.fields_.vel_, (Vec3d{0, 0, 0.5 * std::cos(5.0 * time)}), 1e-12);
}

UNIT_TEST(shaker_tilted_direction) {
  // only the component of the shaker direction along the normal moves the wall
  Driver_params motion;
  motion.amplitude_ = 0.1;
  motion.omega_ = 5.0;
  motion.shaker_dir_ = {std::sqrt(0.75), 0, 0.5};
  Surface surf = make_surface(SHAKER, 1.0, motion);
  surf.push_f_v_r(motion, 0.0, 0.1);
  CHECK_NEAR(surf.fields_.offset_, 1.0 + 0.5 * 0.1 * std::sin(0.5), 1e-12);
}

// ============================================================
// === YAML / dump
// ============================================================

UNIT_TEST(yaml_fields) {
  const SurfaceFields f = YAML::Load("{ offset: 2, normal: [0,1,0], vel: [0,1,0], surface: 4 }").as<SurfaceFields>();
  CHECK_NEAR(f.offset_, 2.0, 0.0);
  CHECK_VEC3_NEAR(f.normal_, (Vec3d{0, 1, 0}), 0.0);
  CHECK_VEC3_NEAR(f.center_, (Vec3d{0, 2, 0}), 0.0);  // default center: offset * normal
  CHECK_NEAR(f.surface_, 4.0, 0.0);

  const SurfaceFields g = YAML::Load("{ offset: 2, normal: [0,1,0], center: [5,2,5] }").as<SurfaceFields>();
  CHECK_VEC3_NEAR(g.center_, (Vec3d{5, 2, 5}), 0.0);

  SurfaceFields h;
  CHECK(!YAML::convert<SurfaceFields>::decode(YAML::Load("{ normal: [0,1,0] }"), h));  // no offset
  // a rotating surface requires a center
  CHECK(!YAML::convert<SurfaceFields>::decode(YAML::Load("{ offset: 0, normal: [0,0,1], vrot: [0,0,1] }"), h));
}

UNIT_TEST(dump_and_read_back) {
  Surface surf = make_surface(STATIONARY, -0.5);
  std::stringstream stream;
  Driver_params motion;
  surf.dump_data(motion, 2, stream);
  CHECK(contains(stream.str(), "type: SURFACE"));
  const SurfaceFields f = YAML::Load(stream.str())[0]["state"].as<SurfaceFields>();
  CHECK_NEAR(f.offset_, -0.5, 1e-6);
  CHECK_VEC3_NEAR(f.normal_, (Vec3d{0, 0, 1}), 1e-6);
  CHECK_VEC3_NEAR(f.center_, (Vec3d{0, 0, -0.5}), 1e-6);
}

UNIT_TEST_MAIN()
