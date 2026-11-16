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
#pragma once

#include <cmath>

namespace exaDEM {

// Minimum absolute diagonal xform scale below which the periodic cell is
// considered to have collapsed (see cell_barostat_prolog's documentation).
constexpr double cell_barostat_min_absolute_scale = 1.e-6;

// First half (drift + first half-kick) of one periodic cell axis, same design
// as Rockable's PeriodicCell. vh is updated in place; the caller runs this
// once per periodic axis, before compute_force.
inline double cell_barostat_prolog_axis(double h_old, double& vh, double ah, double dt) {
  const double h_new = h_old + dt * vh + 0.5 * dt * dt * ah;
  vh += 0.5 * dt * ah;
  return h_new;
}

// True once all three diagonal xform entries are finite and strictly positive.
inline bool cell_barostat_scale_valid(double hx, double hy, double hz) {
  return std::isfinite(hx) && std::isfinite(hy) && std::isfinite(hz) && hx > 0.0 && hy > 0.0 && hz > 0.0;
}

// True once any diagonal xform entry has collapsed below cell_barostat_min_absolute_scale.
inline bool cell_barostat_scale_collapsed(double hx, double hy, double hz) {
  return std::fabs(hx) < cell_barostat_min_absolute_scale || std::fabs(hy) < cell_barostat_min_absolute_scale ||
         std::fabs(hz) < cell_barostat_min_absolute_scale;
}

// Cell acceleration for one periodic axis (Rockable's PeriodicCell epilog),
// with h_dimensionless * bounds_size the *absolute* cell length along that axis:
//   ah = ( (volume / h_absolute) * (stress - sig) - damping * vh ) / mass
// (compressive stress is positive in this codebase's virial convention, so a
// positive cell_sig pulls the cell towards more compression.)
inline double cell_barostat_epilog_axis_accel(double h_dimensionless, double bounds_size, double volume, double sig,
                                              double stress, double vh, double mass, double damping) {
  const double h_absolute = h_dimensionless * bounds_size;
  const double driving_force = (volume / h_absolute) * (stress - sig);
  const double damping_force = damping * vh;
  return (driving_force - damping_force) / mass;
}

}  // namespace exaDEM
