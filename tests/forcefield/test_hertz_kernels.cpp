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

#include <exaDEM/forcefield/hertz_kernels.hpp>

#include "unit_test.hpp"

using namespace exaDEM;
using exanb::Vec3d;

namespace {
constexpr double tol = 1e-12;
}

UNIT_TEST(hertz_elastic_force_value) {
  // 4/3 E* sqrt(R*) delta^(3/2) with E* = 2, R* = 4, delta = 0.25 (dn = -0.25): 4/3 * 2 * 2 * 0.125
  CHECK_NEAR(hertz_elastic_force(2.0, 4.0, -0.25), (4.0 / 3.0) * 2.0 * 2.0 * 0.125, tol);
}

UNIT_TEST(hertz_elastic_force_is_zero_without_overlap) {
  CHECK_NEAR(hertz_elastic_force(2.0, 4.0, 0.0), 0.0, 0.0);
  CHECK_NEAR(hertz_elastic_force(2.0, 4.0, 0.5), 0.0, 0.0);
}

UNIT_TEST(hertz_normal_force_is_never_attractive) {
  CHECK_NEAR(hertz_normal_force(2.0, 4.0, 0.9, 1.0, -0.25, -100.0), 0.0, 0.0);
  CHECK(hertz_normal_force(2.0, 4.0, 0.9, 1.0, -0.25, 1.0) > hertz_elastic_force(2.0, 4.0, -0.25));
}

UNIT_TEST(hertz_normal_force_damping_uses_the_tangent_stiffness) {
  // k = dFe/d(delta) = 2 E* sqrt(R* delta) = 2 * 2 * sqrt(4 * 0.25) = 4
  const double k = 4.0;
  const double expected = hertz_elastic_force(2.0, 4.0, -0.25) + compute_damp(0.5, k, 1.5) * 0.1;
  CHECK_NEAR(hertz_normal_force(2.0, 4.0, 0.5, 1.5, -0.25, 0.1), expected, tol);
}

UNIT_TEST(hertz_normal_force_cohesive) {
  const Vec3d n = {-1, 0, 0};
  const double fe = hertz_elastic_force(2.0, 4.0, -0.25);
  CHECK_VEC3_NEAR(compute_hertz_normal_force<false>(2.0, 3.0, 4.0, 0.0, 1.0, -0.25, 0.0, n), (Vec3d{-fe, 0, 0}), tol);
  CHECK_VEC3_NEAR(compute_hertz_normal_force<true>(2.0, 3.0, 4.0, 0.0, 1.0, -0.25, 0.0, n), (Vec3d{-(fe - 3.0), 0, 0}),
                  tol);
  // never lower than -fc
  CHECK_VEC3_NEAR(compute_hertz_normal_force<true>(2.0, 3.0, 4.0, 0.9, 1.0, -0.25, -100.0, n), (Vec3d{3.0, 0, 0}), tol);
}

UNIT_TEST(hertz_threshold_from_force) {
  CHECK_NEAR(compute_threshold_ft_from_force(0.5, 10.0), 5.0, tol);
  CHECK_NEAR(compute_threshold_ft_from_force(0.5, -10.0), 5.0, tol);
}

UNIT_TEST_MAIN()
