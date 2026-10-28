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

#include <exaDEM/forcefield/contact_force_common.hpp>
#include <exaDEM/forcefield/hertz_kernels.hpp>
#include <exaDEM/forcefield/hertz_parameters.hpp>

namespace exaDEM {

/**
 * @brief Hertz contact law (spheres), see HertzParams and hertz_kernels.hpp.
 *
 * Fn = max(0, 4/3 E* sqrt(R*) |dn|^(3/2) + c vn) (- fc if cohesive), and a tangential force with a constant kt.
 * The Coulomb limit of the friction uses the elastic Hertz force.
 */
template <CohesiveLawType CohesiveLaw>
ONIKA_HOST_DEVICE_FUNC inline void hertz_contact_force_core(const double dn,
                                                            const Vec3d& n,  // -normal
                                                            const double dt, const HertzParams& hp, const double meff,
                                                            const double reff,
                                                            Vec3d& ft,  // tangential force between particle i and j
                                                            const Vec3d& contact_position,
                                                            const Vec3d& pos_i,   // positions i
                                                            const Vec3d& vel_i,   // velocities i
                                                            Vec3d& f_i,           // forces i
                                                            Vec3d& mom_i,         // moments i
                                                            const Vec3d& vrot_i,  // angular velocities i
                                                            const Vec3d& pos_j,   // positions j
                                                            const Vec3d& vel_j,   // velocities j
                                                            const Vec3d& vrot_j   // angular velocities j
) {
  constexpr bool cohesive = (CohesiveLaw == CohesiveLawType::Cohesive);
  constexpr bool dmt = (CohesiveLaw == CohesiveLawType::DMT);

  // === Cohesive law : before contact
  if constexpr (cohesive) {
    if (dn >= 0)  // dn <= hp.dncut_ if contact
    {
      const double fn_value = (hp.fc_ / hp.dncut_) * dn - hp.fc_;
      f_i += fn_value * n;
      ft = {0, 0, 0};
      return;
    }
  }

  // === Relative velocity (j relative to i)
  auto vel = compute_relative_velocity(contact_position, pos_i, vel_i, vrot_i, pos_j, vel_j, vrot_j);
  const double vn = exanb::dot(vel, n);

  // === Compute Contact forces
  // - Normal force (Hertz elastic contact + viscous damping)
  const Vec3d fn = compute_hertz_normal_force<cohesive>(hp.estar_, hp.fc_, reff, hp.damp_rate_, meff, dn, vn, n);

  // - Tangential force (friction)
  ft += exaDEM::compute_tangential_force(hp.kt_, dt, vn, n, vel);

  // - Fit tangential force
  double threshold_ft;
  if constexpr (!cohesive) {
    threshold_ft = exaDEM::compute_threshold_ft_from_force(hp.mu_, exaDEM::hertz_elastic_force(hp.estar_, reff, dn));
  } else {
    threshold_ft = exaDEM::compute_threshold_ft_with_cohesive_force(hp.mu_, exanb::dot(fn, n), hp.fc_);
  }
  exaDEM::fit_tangential_force(threshold_ft, ft);

  // === Sum forces
  f_i = fn + ft;

  // === Adhesion Laws : DMT adhesive force
  if constexpr (dmt) {
    apply_dmt_force(hp.gamma_, reff, dn, n, f_i);
  }

  // === update moments
  update_rolling_moment(hp.kr_, dt, contact_position, pos_i, vrot_i, vrot_j, fn, mom_i);
}
}  // namespace exaDEM
