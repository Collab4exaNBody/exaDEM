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
#include <limits>

#include "unit_test.hpp"

using exaDEM::cell_barostat_min_absolute_scale;
using exaDEM::cell_barostat_prolog_axis;
using exaDEM::cell_barostat_scale_collapsed;
using exaDEM::cell_barostat_scale_valid;

// ============================================================
// === cell_barostat_prolog_axis
// ============================================================

UNIT_TEST(prolog_axis_at_rest_is_unchanged) {
  double vh = 0.0;
  const double h_new = cell_barostat_prolog_axis(1.0, vh, /*ah=*/0.0, /*dt=*/0.1);
  CHECK_NEAR(h_new, 1.0, 1e-14);
  CHECK_NEAR(vh, 0.0, 1e-14);
}

UNIT_TEST(prolog_axis_matches_hand_computed_values) {
  // h_new = h_old + dt*vh + 0.5*dt^2*ah ; vh_new = vh + 0.5*dt*ah
  double vh = 2.0;
  const double h_old = 1.0;
  const double ah = 4.0;
  const double dt = 0.5;
  const double h_new = cell_barostat_prolog_axis(h_old, vh, ah, dt);
  CHECK_NEAR(h_new, 1.0 + 0.5 * 2.0 + 0.5 * 0.25 * 4.0, 1e-14);  // = 2.5
  CHECK_NEAR(vh, 2.0 + 0.5 * 0.5 * 4.0, 1e-14);                  // = 3.0
}

UNIT_TEST(prolog_axis_vh_is_updated_in_place_not_overwritten) {
  // the drift uses the *old* vh, the half-kick is then added on top of it
  double vh = 1.0;
  cell_barostat_prolog_axis(10.0, vh, /*ah=*/2.0, /*dt=*/1.0);
  CHECK_NEAR(vh, 1.0 + 0.5 * 1.0 * 2.0, 1e-14);  // = 2.0, not just 0.5*ah
}

UNIT_TEST(prolog_axis_negative_ah_shrinks_cell) {
  double vh = 0.0;
  const double h_new = cell_barostat_prolog_axis(5.0, vh, /*ah=*/-2.0, /*dt=*/1.0);
  CHECK(h_new < 5.0);
  CHECK(vh < 0.0);
}

// ============================================================
// === cell_barostat_scale_valid
// ============================================================

UNIT_TEST(scale_valid_all_positive_finite) { CHECK(cell_barostat_scale_valid(1.0, 2.0, 3.0)); }

UNIT_TEST(scale_valid_rejects_zero_or_negative) {
  CHECK(!cell_barostat_scale_valid(0.0, 1.0, 1.0));
  CHECK(!cell_barostat_scale_valid(1.0, -1.0, 1.0));
  CHECK(!cell_barostat_scale_valid(1.0, 1.0, -1.0e-9));
}

UNIT_TEST(scale_valid_rejects_non_finite) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  CHECK(!cell_barostat_scale_valid(nan, 1.0, 1.0));
  CHECK(!cell_barostat_scale_valid(1.0, inf, 1.0));
  CHECK(!cell_barostat_scale_valid(1.0, 1.0, -inf));
}

// ============================================================
// === cell_barostat_scale_collapsed
// ============================================================

UNIT_TEST(scale_collapsed_false_for_normal_scales) { CHECK(!cell_barostat_scale_collapsed(1.0, 1.0, 1.0)); }

UNIT_TEST(scale_collapsed_true_when_any_axis_too_small) {
  const double tiny = cell_barostat_min_absolute_scale * 0.1;
  CHECK(cell_barostat_scale_collapsed(tiny, 1.0, 1.0));
  CHECK(cell_barostat_scale_collapsed(1.0, tiny, 1.0));
  CHECK(cell_barostat_scale_collapsed(1.0, 1.0, -tiny));  // uses fabs, sign doesn't matter
}

UNIT_TEST(scale_collapsed_boundary_is_strict) {
  // exactly at the threshold is NOT collapsed (strict <), just below it is
  CHECK(!cell_barostat_scale_collapsed(cell_barostat_min_absolute_scale, 1.0, 1.0));
  CHECK(cell_barostat_scale_collapsed(cell_barostat_min_absolute_scale * 0.999, 1.0, 1.0));
}

UNIT_TEST_MAIN()
