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
// gravity_force.hpp is not self-contained (it uses exaDEM::RShapeDriver and MotionType),
// headers are included in the same order as in src/forcefield/gravity_force.cu.
#include <exanb/core/grid.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <onika/scg/operator.h>

#include <exaDEM/drivers.hpp>
#include <exaDEM/forcefield/gravity_force.hpp>

#include "unit_test.hpp"

using exaDEM::GravityForceFunctor;
using exanb::Vec3d;

UNIT_TEST(default_gravity) {
  GravityForceFunctor func;
  CHECK_VEC3_NEAR(func.g_, (Vec3d{0.0, 0.0, -9.807}), 0.0);
}

UNIT_TEST(gravity_adds_m_g) {
  // same values as the former tests/test_gravity.cpp
  const GravityForceFunctor func{Vec3d{0.0, 0.0, -9.807}};
  const double mass = 2.0;
  double fx = 1.0, fy = 10.0, fz = 100.0;
  func(mass, fx, fy, fz);
  CHECK_NEAR(fx, 1.0, 1e-14);
  CHECK_NEAR(fy, 10.0, 1e-14);
  CHECK_NEAR(fz, 100.0 - 2.0 * 9.807, 1e-12);
}

UNIT_TEST(gravity_any_direction) {
  const GravityForceFunctor func{Vec3d{1.0, -2.0, 3.0}};
  double fx = 0.0, fy = 0.0, fz = 0.0;
  func(0.5, fx, fy, fz);
  CHECK_NEAR(fx, 0.5, 1e-14);
  CHECK_NEAR(fy, -1.0, 1e-14);
  CHECK_NEAR(fz, 1.5, 1e-14);
}

UNIT_TEST(gravity_accumulates) {
  // forces are added, not overwritten
  const GravityForceFunctor func{Vec3d{0.0, 0.0, -10.0}};
  double fx = 0.0, fy = 0.0, fz = 0.0;
  for (int i = 0; i < 3; i++) {
    func(1.0, fx, fy, fz);
  }
  CHECK_NEAR(fz, -30.0, 1e-12);
}

UNIT_TEST(gravity_zero_mass) {
  const GravityForceFunctor func;
  double fx = 1.0, fy = 2.0, fz = 3.0;
  func(0.0, fx, fy, fz);
  CHECK_NEAR(fx, 1.0, 0.0);
  CHECK_NEAR(fy, 2.0, 0.0);
  CHECK_NEAR(fz, 3.0, 0.0);
}

UNIT_TEST_MAIN()
