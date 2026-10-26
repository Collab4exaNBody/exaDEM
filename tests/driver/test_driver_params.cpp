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
using exaDEM::unit_test::is_finite;
using exaDEM::unit_test::is_unit;
using exanb::Vec3d;

namespace {
constexpr double tol = 1e-12;

Driver_params decode(const std::string& yaml) { return YAML::Load(yaml).as<Driver_params>(); }

bool try_decode(const std::string& yaml) {
  Driver_params p;
  return YAML::convert<Driver_params>::decode(YAML::Load(yaml), p);
}
}  // namespace

// ============================================================
// === Motion window and forces
// ============================================================

UNIT_TEST(default_params) {
  Driver_params p;
  CHECK_VEC3_NEAR(p.motion_vector_, (Vec3d{0, 0, 0}), 0.0);
  CHECK(p.motion_start_threshold_ == 0.0);
  CHECK(p.motion_end_threshold_ == 1e300);
  CHECK(p.input_motion_type_ == STATIONARY);
}

UNIT_TEST(motion_triggered) {
  Driver_params p;
  p.motion_start_threshold_ = 1.0;
  p.motion_end_threshold_ = 2.0;
  CHECK(!p.is_motion_triggered(0.5));
  CHECK(p.is_motion_triggered(1.0));  // bounds are inclusive
  CHECK(p.is_motion_triggered(1.5));
  CHECK(p.is_motion_triggered(2.0));
  CHECK(!p.is_motion_triggered(2.5));
  CHECK(p.is_motion_triggered(uint64_t(150), 0.01));  // timestep * dt = 1.5
  CHECK(!p.is_motion_triggered(uint64_t(50), 0.01));
}

UNIT_TEST(update_forces) {
  Driver_params p;
  p.motion_vector_ = {0, 0, 1};
  p.const_force_ = 5.0;

  Vec3d f = {1, 2, 3};
  p.update_forces(LINEAR_FORCE_MOTION, f);
  CHECK_VEC3_NEAR(f, (Vec3d{0, 0, 8}), tol);  // projected on the motion vector + constant force

  f = {1, 2, 3};
  p.update_forces(PARTICLE, f);
  CHECK_VEC3_NEAR(f, (Vec3d{1, 2, 3}), 0.0);  // free body: unchanged

  for (MotionType m : {STATIONARY, LINEAR_MOTION, SHAKER, TABULATED}) {
    f = {1, 2, 3};
    p.update_forces(m, f);
    CHECK_VEC3_NEAR(f, (Vec3d{0, 0, 0}), 0.0);  // kinematic motions ignore the forces
  }
}

// ============================================================
// === Coherence checks
// ============================================================

UNIT_TEST(coherence_linear_requires_motion_vector) {
  Driver_params p;
  p.const_vel_ = 1.0;
  CHECK(!p.check_motion_coherence(LINEAR_MOTION));
  p.motion_vector_ = {0, 0, 1};
  CHECK(p.check_motion_coherence(LINEAR_MOTION));
}

UNIT_TEST(coherence_normalizes_long_motion_vector) {
  Driver_params p;
  p.const_vel_ = 1.0;
  p.motion_vector_ = {0, 3, 4};
  CHECK(p.check_motion_coherence(LINEAR_MOTION));
  CHECK_VEC3_NEAR(p.motion_vector_, (Vec3d{0, 0.6, 0.8}), tol);
}

UNIT_TEST(coherence_normalizes_short_motion_vector) {
  // a motion vector shorter than 1 must also be normalized, otherwise the velocity is scaled down
  Driver_params p;
  p.const_vel_ = 1.0;
  p.motion_vector_ = {0, 0, 0.5};
  CHECK(p.check_motion_coherence(LINEAR_MOTION));
  CHECK_VEC3_NEAR(p.motion_vector_, (Vec3d{0, 0, 1}), tol);
}

UNIT_TEST(coherence_shaker) {
  Driver_params p;
  CHECK(!p.check_motion_coherence(SHAKER));  // amplitude and omega are not defined
  p.amplitude_ = 0.1;
  CHECK(!p.check_motion_coherence(SHAKER));  // omega is not defined
  p.omega_ = 10;
  p.shaker_dir_ = {0, 0, 2};
  CHECK(p.check_motion_coherence(SHAKER));
  CHECK(is_unit(p.shaker_dir_));
}

UNIT_TEST(coherence_tabulated) {
  Driver_params p;
  CHECK(!p.check_motion_coherence(TABULATED));  // empty
  p.tab_time_ = {0.5, 1.0};
  p.tab_pos_ = {Vec3d{0, 0, 0}, Vec3d{1, 0, 0}};
  CHECK(!p.check_motion_coherence(TABULATED));  // the first time must be 0
  p.tab_time_ = {0.0, 1.0, 2.0};
  CHECK(!p.check_motion_coherence(TABULATED));  // sizes differ
  p.tab_time_ = {0.0, 2.0};
  CHECK(p.check_motion_coherence(TABULATED));
  p.tab_time_ = {0.0, 2.0, 1.0};
  p.tab_pos_ = {Vec3d{0, 0, 0}, Vec3d{1, 0, 0}, Vec3d{2, 0, 0}};
  CHECK(!p.check_motion_coherence(TABULATED));  // not sorted
}

UNIT_TEST(coherence_stationary) {
  Driver_params p;
  CHECK(p.check_motion_coherence(STATIONARY));
}

// ============================================================
// === Shaker, pendulum and expressions
// ============================================================

UNIT_TEST(shaker_signal_and_velocity) {
  Driver_params p;
  p.amplitude_ = 0.2;
  p.omega_ = 3.0;
  p.shaker_dir_ = {1, 0, 0};
  p.motion_start_threshold_ = 1.0;
  CHECK_NEAR(p.shaker_signal(1.0), 0.0, tol);  // the signal starts with the motion
  CHECK_NEAR(p.shaker_signal(1.5), 0.2 * std::sin(1.5), tol);
  CHECK_VEC3_NEAR(p.shaker_velocity(1.5), (Vec3d{0.2 * 3.0 * std::cos(1.5), 0, 0}), tol);
}

UNIT_TEST(pendulum_plane) {
  // anchor above the initial position, swing along x: the surface plane contains the anchor
  // and its normal is orthogonal to the current pendulum arm
  Driver_params p;
  p.amplitude_ = 0.5;
  p.omega_ = 1.0;
  p.pendulum_anchor_point_ = {0, 0, 2};
  p.pendulum_initial_position_ = {0, 0, 1};
  p.pendulum_swing_dir_ = {1, 0, 0};
  const double t = M_PI / 2;
  auto [offset, normal] = p.compute_offset_normal_pendulum_motion(t);
  const Vec3d arm =
      p.pendulum_initial_position_ + p.pendulum_swing_dir_ * p.pendulum_signal(t) - p.pendulum_anchor_point_;
  CHECK(is_unit(normal));
  CHECK_NEAR(exanb::dot(normal, arm), 0.0, tol);
  CHECK_NEAR(offset, exanb::dot(normal, p.pendulum_anchor_point_), tol);
  CHECK_VEC3_NEAR(normal, (Vec3d{2, 0, 1} / std::sqrt(5.0)), tol);
}

UNIT_TEST(pendulum_plane_at_rest) {
  // at the start of the motion the pendulum is at its initial position: the normal must stay defined
  Driver_params p;
  p.amplitude_ = 0.5;
  p.omega_ = 1.0;
  p.pendulum_anchor_point_ = {0, 0, 2};
  p.pendulum_initial_position_ = {0, 0, 1};
  p.pendulum_swing_dir_ = {1, 0, 0};
  auto [offset, normal] = p.compute_offset_normal_pendulum_motion(0.0);
  CHECK(std::isfinite(offset));
  CHECK(is_finite(normal));
}

UNIT_TEST(expressions) {
  Driver_params p;
  p.expr_.expr_use_v_ = true;
  p.expr_.expr_vx_ = "2*t";
  p.expr_.expr_vz_ = "sin(t)";
  p.expr_.expr_use_vrot_ = true;
  p.expr_.expr_vroty_ = "t^2";
  p.expr_.expr_use_mom_ = true;
  p.expr_.expr_momz_ = "-1";
  p.motion_start_threshold_ = 1.0;  // t is the time since the start of the motion
  CHECK_VEC3_NEAR(p.driver_expr_v(3.0), (Vec3d{4, 0, std::sin(2.0)}), tol);
  CHECK_VEC3_NEAR(p.driver_expr_vrot(3.0), (Vec3d{0, 4, 0}), tol);
  CHECK_VEC3_NEAR(p.driver_expr_mom(3.0), (Vec3d{0, 0, -1}), tol);
  CHECK(p.is_expr(EXPRESSION, 3.0));
  CHECK(!p.is_expr(EXPRESSION, 0.5));  // before the start of the motion
  CHECK(!p.is_expr(LINEAR_MOTION, 3.0));
}

// ============================================================
// === YAML
// ============================================================

UNIT_TEST(yaml_linear_motion) {
  const Driver_params p = decode(
      "{ motion_type: LINEAR_MOTION, motion_vector: [0,0,1], const_vel: 2.5, motion_start_threshold: 1, "
      "motion_end_threshold: 3 }");
  CHECK(p.input_motion_type_ == LINEAR_MOTION);
  CHECK_VEC3_NEAR(p.motion_vector_, (Vec3d{0, 0, 1}), 0.0);
  CHECK_NEAR(p.const_vel_, 2.5, 0.0);
  CHECK_NEAR(p.motion_start_threshold_, 1.0, 0.0);
  CHECK_NEAR(p.motion_end_threshold_, 3.0, 0.0);
}

UNIT_TEST(yaml_compressive_and_shaker) {
  const Driver_params c = decode("{ motion_type: COMPRESSIVE_FORCE, sigma: 0.5, damprate: 0.9 }");
  CHECK(c.input_motion_type_ == COMPRESSIVE_FORCE);
  CHECK_NEAR(c.sigma_, 0.5, 0.0);
  CHECK_NEAR(c.damprate_, 0.9, 0.0);

  const Driver_params s = decode("{ motion_type: SHAKER, omega: 10, amplitude: 0.1, shaker_dir: [1,0,0] }");
  CHECK(s.input_motion_type_ == SHAKER);
  CHECK_NEAR(s.omega_, 10.0, 0.0);
  CHECK_NEAR(s.amplitude_, 0.1, 0.0);
  CHECK_VEC3_NEAR(s.shaker_dir_, (Vec3d{1, 0, 0}), 0.0);

  const Driver_params d = decode("{ motion_type: SHAKER, omega: 10, amplitude: 0.1 }");
  CHECK_VEC3_NEAR(d.shaker_dir_, (Vec3d{0, 0, 1}), 0.0);  // default direction
}

UNIT_TEST(yaml_tabulated_and_expression) {
  const Driver_params t = decode("{ motion_type: TABULATED, time: [0, 1], positions: [[0,0,0], [1,2,3]] }");
  CHECK(t.tab_time_.size() == 2);
  CHECK_VEC3_NEAR(t.tab_pos_[1], (Vec3d{1, 2, 3}), 0.0);

  const Driver_params e = decode("{ motion_type: EXPRESSION, expr: { vx: 2*t, vrotz: 1 } }");
  CHECK(e.expr_.expr_use_v_);
  CHECK(e.expr_.expr_use_vrot_);
  CHECK(!e.expr_.expr_use_mom_);
  CHECK(e.expr_.expr_vx_ == "2*t");
  CHECK(e.expr_.expr_vy_ == "0");  // default
}

UNIT_TEST(yaml_missing_fields) {
  CHECK(!try_decode("[1, 2]"));                                                  // not a map
  CHECK(!try_decode("{ const_vel: 1 }"));                                        // no motion type
  CHECK(!try_decode("{ motion_type: LINEAR_MOTION, const_vel: 1 }"));            // no motion vector
  CHECK(!try_decode("{ motion_type: LINEAR_MOTION, motion_vector: [1,0,0] }"));  // no velocity
  CHECK(!try_decode("{ motion_type: COMPRESSIVE_FORCE, sigma: 1 }"));            // no damprate
  CHECK(!try_decode("{ motion_type: TABULATED, time: [0, 1] }"));                // no positions
  CHECK(!try_decode("{ motion_type: SHAKER, amplitude: 1 }"));                   // no omega
  CHECK(try_decode("{ motion_type: STATIONARY }"));
}

UNIT_TEST(dump_and_read_back) {
  Driver_params p;
  p.motion_vector_ = {0, 1, 0};
  p.const_vel_ = 1.5;
  p.motion_start_threshold_ = 0.25;
  std::stringstream stream;
  p.dump_driver_params(LINEAR_MOTION, stream, 0);
  const std::string text = stream.str();
  CHECK(contains(text, "motion_type: LINEAR_MOTION"));
  CHECK(contains(text, "const_vel: 1.5"));

  // the dumped parameters can be read back (restart)
  const Driver_params q = YAML::Load(text)["params"].as<Driver_params>();
  CHECK(q.input_motion_type_ == LINEAR_MOTION);
  CHECK_VEC3_NEAR(q.motion_vector_, p.motion_vector_, 1e-6);
  CHECK_NEAR(q.const_vel_, 1.5, 1e-6);
  CHECK_NEAR(q.motion_start_threshold_, 0.25, 1e-6);
}

UNIT_TEST_MAIN()
