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

#include <exaDEM/forcefield/common_kernels.hpp>

namespace exaDEM {

ONIKA_HOST_DEVICE_FUNC
inline double hertz_elastic_force(const double estar, const double reff, const double dn) {
  if (dn >= 0.0) return 0.0;
  const double delta = -dn;
  return (4.0 / 3.0) * estar * sqrt(reff * delta) * delta;
}

// Elastic part + viscous damping of the Hertz normal force, never attractive.
// dn interpenetration
// vn relative velocity
ONIKA_HOST_DEVICE_FUNC
inline double hertz_normal_force(const double estar, const double reff, const double damp_rate, const double meff,
                                 const double dn, const double vn) {
  if (dn >= 0.0) return 0.0;
  const double delta = -dn;
  const double fne = hertz_elastic_force(estar, reff, dn);
  const double kn_tangent = 2.0 * estar * sqrt(reff * delta);
  const double fnv = compute_damp(damp_rate, kn_tangent, meff) * vn;
  return (fne + fnv > 0) ? fne + fnv : 0;
}

// Same as compute_normal_force for the Hertz law
template <bool cohesive>
ONIKA_HOST_DEVICE_FUNC inline Vec3d compute_hertz_normal_force(const double estar, const double fc, const double reff,
                                                               const double damp_rate, const double meff,
                                                               const double dn, const double vn, const Vec3d& n) {
  const double fn_norm = hertz_normal_force(estar, reff, damp_rate, meff, dn, vn);
  if constexpr (cohesive) {
    const double fn_coh = fn_norm - fc;
    return (fn_coh < (-fc) ? -fc : fn_coh) * n;
  } else {
    return fn_norm * n;
  }
}

// Coulomb threshold computed from the elastic normal force
ONIKA_HOST_DEVICE_FUNC
inline double compute_threshold_ft_from_force(const double mu, const double fne) { return std::fabs(mu * fne); }
}  // namespace exaDEM
