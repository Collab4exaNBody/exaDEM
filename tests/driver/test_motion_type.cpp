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

#include <vector>

using namespace exaDEM;

namespace {
const std::vector<MotionType> all_motion_types = {
    STATIONARY, LINEAR_MOTION, COMPRESSIVE_FORCE, LINEAR_FORCE_MOTION, PARTICLE, LINEAR_COMPRESSIVE_MOTION,
    TABULATED,  SHAKER,        PENDULUM_MOTION,   EXPRESSION};
}

// ============================================================
// === Motion types
// ============================================================

UNIT_TEST(motion_type_string_round_trip) {
  for (MotionType m : all_motion_types) {
    CHECK(string_to_motion_type(motion_type_to_string(m)) == m);
  }
  CHECK(motion_type_to_string(LINEAR_MOTION) == "LINEAR_MOTION");
  CHECK(string_to_motion_type("NOT_A_MOTION") == UNKNOWN);
  CHECK(string_to_motion_type("linear_motion") == UNKNOWN);  // case sensitive
  CHECK(motion_type_to_string(UNKNOWN) == "UNKNOWN");
}

UNIT_TEST(motion_type_predicates) {
  CHECK(is_stationary(STATIONARY));
  CHECK(!is_stationary(LINEAR_MOTION));
  CHECK(is_tabulated(TABULATED));
  CHECK(is_shaker(SHAKER));
  CHECK(is_pendulum(PENDULUM_MOTION));
  CHECK(is_expr(EXPRESSION));

  for (MotionType m : all_motion_types) {
    const bool linear = (m == LINEAR_MOTION || m == LINEAR_FORCE_MOTION || m == LINEAR_COMPRESSIVE_MOTION);
    const bool compressive = (m == COMPRESSIVE_FORCE || m == LINEAR_COMPRESSIVE_MOTION);
    const bool force_motion = (m == PARTICLE || m == LINEAR_FORCE_MOTION);
    CHECK(is_linear(m) == linear);
    CHECK(is_compressive(m) == compressive);
    CHECK(is_force_motion(m) == force_motion);
    CHECK(need_forces(m) == (compressive || force_motion));  // forces are required to integrate the motion
  }
}

UNIT_TEST(valid_motion_type) {
  const std::vector<MotionType> valid = {STATIONARY, SHAKER};
  CHECK(is_valid_motion_type(STATIONARY, valid));
  CHECK(is_valid_motion_type(SHAKER, valid));
  CHECK(!is_valid_motion_type(LINEAR_MOTION, valid));  // prints a warning
}

// ============================================================
// === Driver types
// ============================================================

UNIT_TEST(driver_type_names) {
  CHECK(get_type("CYLINDER") == DRIVER_TYPE::CYLINDER);
  CHECK(get_type("SURFACE") == DRIVER_TYPE::SURFACE);
  CHECK(get_type("BALL") == DRIVER_TYPE::BALL);
  CHECK(get_type("RSHAPE") == DRIVER_TYPE::RSHAPE);
  CHECK(print(DRIVER_TYPE::BALL) == "Ball");
  CHECK(print(DRIVER_TYPE::UNDEFINED) == "Undefined Driver");
}

UNIT_TEST(driver_type_traits) {
  static_assert(get_type<Cylinder>() == DRIVER_TYPE::CYLINDER);
  static_assert(get_type<Surface>() == DRIVER_TYPE::SURFACE);
  static_assert(get_type<Ball>() == DRIVER_TYPE::BALL);
  static_assert(get_type<RShapeDriver>() == DRIVER_TYPE::RSHAPE);
  static_assert(get_type<UndefinedDriver>() == DRIVER_TYPE::UNDEFINED);
  // the driver type is the index of the driver vector in Drivers::data_
  static_assert(DRIVER_TYPE::CYLINDER == 0 && DRIVER_TYPE::SURFACE == 1 && DRIVER_TYPE::BALL == 2 &&
                DRIVER_TYPE::RSHAPE == 3 && DRIVER_TYPE_SIZE == 4);
  CHECK(str2int("BALL") != str2int("CYLINDER"));
}

UNIT_TEST_MAIN()
