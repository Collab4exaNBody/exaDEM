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
#include <exaDEM/barostat/cell_barostat.hpp>

#include "unit_test.hpp"

using exaDEM::cell_barostat_epilog_axis_accel;

// ============================================================
// === cell_barostat_epilog_axis_accel
// ============================================================

UNIT_TEST(epilog_accel_zero_at_target_stress_undamped) {
  // stress == sig and damping == 0 -> no driving force, no damping force
  const double ah = cell_barostat_epilog_axis_accel(/*h_dimensionless=*/1.0, /*bounds_size=*/10.0, /*volume=*/1000.0,
                                                     /*sig=*/500.0, /*stress=*/500.0, /*vh=*/0.0, /*mass=*/1.0,
                                                     /*damping=*/0.0);
  CHECK_NEAR(ah, 0.0, 1e-14);
}

UNIT_TEST(epilog_accel_over_compressed_cell_expands) {
  // stress above target (compressive-positive convention): the cell must expand (ah > 0)
  const double ah = cell_barostat_epilog_axis_accel(1.0, 10.0, 1000.0, /*sig=*/500.0, /*stress=*/600.0, 0.0, 1.0,
                                                     0.0);
  CHECK(ah > 0.0);
}

UNIT_TEST(epilog_accel_under_compressed_cell_shrinks) {
  // stress below target: the cell must shrink (ah < 0)
  const double ah = cell_barostat_epilog_axis_accel(1.0, 10.0, 1000.0, /*sig=*/500.0, /*stress=*/400.0, 0.0, 1.0,
                                                     0.0);
  CHECK(ah < 0.0);
}

UNIT_TEST(epilog_accel_damping_opposes_cell_velocity) {
  // at the target stress, a purely damped response opposes vh regardless of its sign
  const double ah_pos_vh = cell_barostat_epilog_axis_accel(1.0, 10.0, 1000.0, 500.0, 500.0, /*vh=*/2.0, 1.0,
                                                           /*damping=*/0.5);
  const double ah_neg_vh = cell_barostat_epilog_axis_accel(1.0, 10.0, 1000.0, 500.0, 500.0, /*vh=*/-2.0, 1.0, 0.5);
  CHECK_NEAR(ah_pos_vh, -1.0, 1e-14);  // -damping*vh/mass = -0.5*2/1
  CHECK_NEAR(ah_neg_vh, 1.0, 1e-14);
}

UNIT_TEST(epilog_accel_matches_hand_computed_value) {
  // h_absolute = h_dimensionless * bounds_size = 2.0 * 3.0 = 6.0
  // driving_force = (volume / h_absolute) * (stress - sig) = (120/6) * (16-10) = 120
  // damping_force = damping * vh = 0.5 * 2.0 = 1.0
  // ah = (driving_force - damping_force) / mass = (120 - 1.0) / 4.0 = 29.75
  const double ah = cell_barostat_epilog_axis_accel(/*h_dimensionless=*/2.0, /*bounds_size=*/3.0, /*volume=*/120.0,
                                                     /*sig=*/10.0, /*stress=*/16.0, /*vh=*/2.0, /*mass=*/4.0,
                                                     /*damping=*/0.5);
  CHECK_NEAR(ah, 29.75, 1e-12);
}

UNIT_TEST(epilog_accel_scales_inversely_with_absolute_cell_length) {
  // same dimensionless xform entry, twice the bounds_size -> twice the absolute
  // length -> half the driving force for the same (stress - sig)
  const double ah_small = cell_barostat_epilog_axis_accel(1.0, 10.0, 1000.0, 500.0, 600.0, 0.0, 1.0, 0.0);
  const double ah_large = cell_barostat_epilog_axis_accel(1.0, 20.0, 1000.0, 500.0, 600.0, 0.0, 1.0, 0.0);
  CHECK_NEAR(ah_large, 0.5 * ah_small, 1e-12);
}

UNIT_TEST(epilog_accel_scales_inversely_with_mass) {
  const double ah_m1 = cell_barostat_epilog_axis_accel(1.0, 10.0, 1000.0, 500.0, 600.0, 1.0, 1.0, 1.0);
  const double ah_m2 = cell_barostat_epilog_axis_accel(1.0, 10.0, 1000.0, 500.0, 600.0, 1.0, 2.0, 1.0);
  CHECK_NEAR(ah_m2, 0.5 * ah_m1, 1e-12);
}

UNIT_TEST_MAIN()
