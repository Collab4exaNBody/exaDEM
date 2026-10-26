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

// Ball of radius 1 centered at the origin.
Ball make_ball(MotionType motion = STATIONARY, double radius = 1.0) {
  BallFields fields;
  fields.radius_ = radius;
  fields.center_ = {0, 0, 0};
  return Ball(fields, motion);
}
}  // namespace

// ============================================================
// === Contact detection
// ============================================================

UNIT_TEST(detector_outside) {
  // vertex of radius 0.2 at 1.1 from the center: overlap of 0.1 with the outer surface
  Ball ball = make_ball();
  auto [contact, dn, n, pos] = ball.detector(0.2, Vec3d{1.1, 0, 0});
  CHECK(contact);
  CHECK_NEAR(dn, -0.1, tol);
  CHECK_VEC3_NEAR(n, (Vec3d{1, 0, 0}), tol);       // from the ball to the vertex
  CHECK_VEC3_NEAR(pos, (Vec3d{0.95, 0, 0}), tol);  // middle of the overlap
}

UNIT_TEST(detector_inside) {
  // the vertex is inside the ball and touches its inner surface
  Ball ball = make_ball();
  auto [contact, dn, n, pos] = ball.detector(0.2, Vec3d{0, 0.9, 0});
  CHECK(contact);
  CHECK_NEAR(dn, -0.1, tol);
  CHECK_VEC3_NEAR(n, (Vec3d{0, -1, 0}), tol);  // toward the center
  CHECK_VEC3_NEAR(pos, (Vec3d{0, 1.05, 0}), tol);
}

UNIT_TEST(detector_no_contact) {
  Ball ball = make_ball();
  auto [c1, dn1, n1, p1] = ball.detector(0.2, Vec3d{2, 0, 0});
  CHECK(!c1);
  CHECK_NEAR(dn1, 0.8, tol);
  auto [c2, dn2, n2, p2] = ball.detector(0.2, Vec3d{0.5, 0, 0});
  CHECK(!c2);
  CHECK_NEAR(dn2, 0.3, tol);
}

UNIT_TEST(detector_moved_ball) {
  Ball ball = make_ball();
  ball.fields_.center_ = {10, 0, 0};
  auto [contact, dn, n, pos] = ball.detector(0.2, Vec3d{10, 0, 1.1});
  CHECK(contact);
  CHECK_NEAR(dn, -0.1, tol);
  CHECK_VEC3_NEAR(n, (Vec3d{0, 0, 1}), tol);
}

UNIT_TEST(filter) {
  Ball ball = make_ball();
  CHECK(ball.filter(0.2, Vec3d{1.1, 0, 0}));
  CHECK(ball.filter(0.2, Vec3d{0.9, 0, 0}));
  CHECK(!ball.filter(0.2, Vec3d{1.5, 0, 0}));
  CHECK(!ball.filter(0.2, Vec3d{0, 0, 0}));
}

// ============================================================
// === Geometry
// ============================================================

UNIT_TEST(surface) {
  Ball ball = make_ball(STATIONARY, 2.0);
  CHECK_NEAR(ball.surface(), 16.0 * M_PI, tol);
}

UNIT_TEST(volume) {
  Ball ball = make_ball(STATIONARY, 2.0);
  CHECK_NEAR(ball.volume(), 4.0 / 3.0 * M_PI * 8.0, tol);
}

// ============================================================
// === Motion
// ============================================================

UNIT_TEST(stationary_does_not_move) {
  Ball ball = make_ball();
  ball.fields_.vel_ = {1, 1, 1};
  Driver_params motion;
  ball.push_f_v_r(motion, 0.0, 0.1);
  CHECK_VEC3_NEAR(ball.fields_.center_, (Vec3d{0, 0, 0}), 0.0);
}

UNIT_TEST(linear_motion) {
  Ball ball = make_ball(LINEAR_MOTION);
  Driver_params motion;
  motion.motion_vector_ = {0, 0, 1};
  motion.const_vel_ = 2.0;
  ball.push_f_v(motion, 0.1);
  CHECK_VEC3_NEAR(ball.fields_.vel_, (Vec3d{0, 0, 2}), tol);  // velocity is reset to the prescribed one
  for (int i = 0; i < 10; i++) {
    ball.push_f_v_r(motion, i * 0.1, 0.1);
  }
  CHECK_VEC3_NEAR(ball.fields_.center_, (Vec3d{0, 0, 2}), 1e-12);
}

UNIT_TEST(force_to_accel_kinematic_motion) {
  Ball ball = make_ball(LINEAR_MOTION);
  ball.fields_.forces_ = {1, 2, 3};
  ball.fields_.acc_ = {4, 5, 6};
  Driver_params motion;
  ball.force_to_accel(motion);
  CHECK_VEC3_NEAR(ball.fields_.acc_, (Vec3d{0, 0, 0}), 0.0);
}

UNIT_TEST(compressive_radial_acceleration) {
  // ra = (|F| - sigma * S - damprate * rv) / (0.5 * m)
  Ball ball = make_ball(COMPRESSIVE_FORCE);
  ball.fields_.mass_ = 2.0;
  ball.fields_.forces_ = {0, 30, 40};  // |F| = 50
  ball.fields_.rv_ = 0.5;
  Driver_params motion;
  motion.sigma_ = 1.0;
  motion.damprate_ = 4.0;
  ball.f_ra(motion, 0.1);
  CHECK_NEAR(ball.fields_.ra_, (50.0 - 4.0 * M_PI - 2.0) / 1.0, tol);
}

UNIT_TEST(compressive_uses_system_mass_by_default) {
  Ball ball = make_ball(COMPRESSIVE_FORCE);  // default mass: undefined
  ball.fields_.forces_ = {0, 0, 50};
  Driver_params motion;
  motion.sigma_ = 1.0;
  motion.system_mass_ = 8.0;
  ball.f_ra(motion, 0.1);
  CHECK_NEAR(ball.fields_.ra_, (50.0 - 4.0 * M_PI) / (0.5 * 4.0), tol);  // m = system_mass / 2
}

UNIT_TEST(compressive_radius_integration) {
  Ball ball = make_ball(COMPRESSIVE_FORCE);
  ball.fields_.vel_ = {1, 0, 0};
  ball.fields_.rv_ = 0.5;
  ball.fields_.ra_ = 2.0;
  Driver_params motion;
  motion.sigma_ = 1.0;
  const double dt = 0.1;

  ball.push_f_v(motion, dt);
  CHECK_VEC3_NEAR(ball.fields_.vel_, (Vec3d{0, 0, 0}), 0.0);  // the center does not move
  CHECK_NEAR(ball.fields_.rv_, 0.5 + dt * 2.0, tol);

  ball.push_f_v_r(motion, 0.0, dt);
  CHECK_NEAR(ball.fields_.radius_, 1.0 + dt * 0.7 + 0.5 * dt * dt * 2.0, tol);
  CHECK_VEC3_NEAR(ball.fields_.center_, (Vec3d{0, 0, 0}), 0.0);
}

UNIT_TEST(compressive_without_sigma) {
  // sigma = 0: stress control is inactive, rv is not integrated
  Ball ball = make_ball(COMPRESSIVE_FORCE);
  ball.fields_.rv_ = 0.5;
  ball.fields_.ra_ = 2.0;
  Driver_params motion;
  ball.push_f_v(motion, 0.1);
  CHECK_NEAR(ball.fields_.rv_, 0.5, 0.0);
}

// ============================================================
// === YAML / dump
// ============================================================

UNIT_TEST(yaml_fields) {
  const BallFields f = YAML::Load("{ radius: 0.5, center: [1,2,3], vel: [0,0,1], mass: 7 }").as<BallFields>();
  CHECK_NEAR(f.radius_, 0.5, 0.0);
  CHECK_VEC3_NEAR(f.center_, (Vec3d{1, 2, 3}), 0.0);
  CHECK_VEC3_NEAR(f.vel_, (Vec3d{0, 0, 1}), 0.0);
  CHECK_VEC3_NEAR(f.vrot_, (Vec3d{0, 0, 0}), 0.0);  // default
  CHECK_NEAR(f.mass_, 7.0, 0.0);

  BallFields g;
  CHECK(!YAML::convert<BallFields>::decode(YAML::Load("{ center: [1,2,3] }"), g));  // no radius
  CHECK(!YAML::convert<BallFields>::decode(YAML::Load("{ radius: 1 }"), g));        // no center
}

UNIT_TEST(yaml_mass_from_density) {
  // mass = density * 4/3 pi r^3
  const BallFields f = YAML::Load("{ radius: 0.5, center: [0,0,0], density: 2000 }").as<BallFields>();
  CHECK_NEAR(f.mass_, 2000.0 * 4.0 / 3.0 * M_PI * 0.125, 1e-9);
}

UNIT_TEST(dump_and_read_back) {
  Ball ball = make_ball(LINEAR_MOTION, 0.75);
  ball.fields_.center_ = {1, -2, 3};
  ball.fields_.vel_ = {0, 0, 2};
  Driver_params motion;
  motion.motion_vector_ = {0, 0, 1};
  motion.const_vel_ = 2.0;

  std::stringstream stream;
  ball.dump_data(motion, 3, stream);
  const std::string text = stream.str();
  CHECK(contains(text, "type: BALL"));
  CHECK(contains(text, "id: 3"));

  const YAML::Node node = YAML::Load(text)[0];
  const BallFields f = node["state"].as<BallFields>();
  const Driver_params m = node["params"].as<Driver_params>();
  CHECK_NEAR(f.radius_, 0.75, 1e-6);
  CHECK_VEC3_NEAR(f.center_, (Vec3d{1, -2, 3}), 1e-6);
  CHECK_VEC3_NEAR(f.vel_, (Vec3d{0, 0, 2}), 1e-6);
  CHECK(m.input_motion_type_ == LINEAR_MOTION);
  CHECK_NEAR(m.const_vel_, 2.0, 1e-6);
}

UNIT_TEST(dump_op) {
  Ball ball = make_ball();
  Driver_params motion;
  std::stringstream stream;
  ball.dump_op(motion, 0, stream);
  const std::string text = stream.str();
  CHECK(contains(text, "- register_ball:"));
  CHECK(contains(text, "motion_type: STATIONARY"));
}

UNIT_TEST_MAIN()
