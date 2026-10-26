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
// Tabulated motion (Driver_params::tab_to_position / tab_to_velocity).
// Kept in its own executable: the t = 0 case reads out of bounds and could crash the other tests.
#include "driver_test.hpp"

using namespace exaDEM;
using exanb::Vec3d;

namespace {
constexpr double tol = 1e-12;

// (0,0,0) at t=0 -> (1,0,0) at t=1 -> (1,2,0) at t=2
Driver_params make_table() {
  Driver_params p;
  p.tab_time_ = {0.0, 1.0, 2.0};
  p.tab_pos_ = {Vec3d{0, 0, 0}, Vec3d{1, 0, 0}, Vec3d{1, 2, 0}};
  return p;
}
}  // namespace

UNIT_TEST(interpolation_inside_intervals) {
  const Driver_params p = make_table();
  CHECK_VEC3_NEAR(p.tab_to_position(0.5), (Vec3d{0.5, 0, 0}), tol);
  CHECK_VEC3_NEAR(p.tab_to_velocity(0.5), (Vec3d{1, 0, 0}), tol);
  CHECK_VEC3_NEAR(p.tab_to_position(1.5), (Vec3d{1, 1, 0}), tol);
  CHECK_VEC3_NEAR(p.tab_to_velocity(1.5), (Vec3d{0, 2, 0}), tol);
}

UNIT_TEST(interpolation_on_knots) {
  const Driver_params p = make_table();
  CHECK_VEC3_NEAR(p.tab_to_position(1.0), (Vec3d{1, 0, 0}), tol);
  CHECK_VEC3_NEAR(p.tab_to_position(2.0), (Vec3d{1, 2, 0}), tol);
}

UNIT_TEST(after_the_last_time) {
  const Driver_params p = make_table();
  CHECK_VEC3_NEAR(p.tab_to_position(5.0), (Vec3d{1, 2, 0}), 0.0);  // stays at the last position
  CHECK_VEC3_NEAR(p.tab_to_velocity(5.0), (Vec3d{0, 0, 0}), 0.0);  // and stops
}

UNIT_TEST(at_time_zero) {
  // first call of push_f_v_r_driver (physical_time = 0): must return the first position
  const Driver_params p = make_table();
  CHECK_VEC3_NEAR(p.tab_to_position(0.0), (Vec3d{0, 0, 0}), tol);
  CHECK_VEC3_NEAR(p.tab_to_velocity(0.0), (Vec3d{1, 0, 0}), tol);
}

UNIT_TEST(ball_follows_the_table) {
  Driver_params p = make_table();
  BallFields fields;
  fields.radius_ = 1.0;
  fields.center_ = {9, 9, 9};
  MotionType m = TABULATED;
  Ball ball(fields, m);
  ball.push_f_v_r(p, 1.5, 1e-3);
  CHECK_VEC3_NEAR(ball.fields_.center_, (Vec3d{1, 1, 0}), tol);
  CHECK_VEC3_NEAR(ball.fields_.vel_, (Vec3d{0, 2, 0}), tol);
}

UNIT_TEST_MAIN()
