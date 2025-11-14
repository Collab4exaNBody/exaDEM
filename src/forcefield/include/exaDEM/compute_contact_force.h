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

#include <onika/math/basic_types.h>
#include <exaDEM/common_compute_kernels.h>
#include <exaDEM/contact_force_parameters.h>

namespace exaDEM
{
  ONIKA_HOST_DEVICE_FUNC inline void reset(Vec3d &in) { in = Vec3d{0.0, 0.0, 0.0}; }

  ONIKA_HOST_DEVICE_FUNC inline void cohesive_force_core(const double dn, const Vec3d &n, const double dncut, const double fc, Vec3d &f)
  {
    if (dncut == 0)
      return;

    if (dn <= dncut)
    {
      const double fn_value = (fc / dncut) * dn - fc;
      const Vec3d fn = fn_value * n;

      // === update forces
      f.x += fn.x;
      f.y += fn.y;
      f.z += fn.z;
    }
  }

  template<bool cohesive>
    ONIKA_HOST_DEVICE_FUNC inline void contact_force_core(const double dn,
        const Vec3d &n, // -normal
        const double dt,
        const ContactParams &hkp,
        const double meff,
        Vec3d &ft, // tangential force between particle i and j
        const Vec3d &contact_position,
        const Vec3d &pos_i,  // positions i
        const Vec3d &vel_i,  // positions i
        Vec3d &f_i,          // forces i
        Vec3d &mom_i,        // moments i
        const Vec3d &vrot_i, // angular velocities i
        const Vec3d &pos_j,  // positions j
        const Vec3d &vel_j,  // positions j
        const Vec3d &vrot_j  // angular velocities j
        );

  template<>
    ONIKA_HOST_DEVICE_FUNC inline void contact_force_core<false>(const double dn,
        const Vec3d &n, // -normal
        const double dt,
        const ContactParams &hkp,
        const double meff,
        Vec3d &ft, // tangential force between particle i and j
        const Vec3d &contact_position,
        const Vec3d &pos_i,  // positions i
        const Vec3d &vel_i,  // positions i
        Vec3d &f_i,          // forces i
        Vec3d &mom_i,        // moments i
        const Vec3d &vrot_i, // angular velocities i
        const Vec3d &pos_j,  // positions j
        const Vec3d &vel_j,  // positions j
        const Vec3d &vrot_j  // angular velocities j
        )
    {
      const double damp = compute_damp(hkp.damp_rate, hkp.kn, meff);

      // === Relative velocity (j relative to i)
      auto vel = compute_relative_velocity(contact_position, pos_i, vel_i, vrot_i, pos_j, vel_j, vrot_j);

      // compute relative velocity
      const double vn = exanb::dot(vel, n);

      // === Normal force (elatic contact + viscous damping)
      const Vec3d fn = compute_normal_force(hkp.kn, damp, dn, vn, n); // fc ==> cohesive force

      // === Tangential force (friction)
      ft += exaDEM::compute_tangential_force(hkp.kt, dt, vn, n, vel);
      // ft     += exaDEM::compute_tangential_force(kt, dt, vn, n, vel);

      // fit tangential force
      auto threshold_ft = exaDEM::compute_threshold_ft(hkp.mu, hkp.kn, dn);
      exaDEM::fit_tangential_force(threshold_ft, ft);

      // === sum forces
      f_i = fn + ft;

      // === update moments
      mom_i += hkp.kr * (vrot_j - vrot_i) * dt;

      ///*
      // test
      Vec3d branch = contact_position - pos_i;
      double r = (exanb::dot(branch, vrot_i)) / (exanb::dot(vrot_i, vrot_i));
      branch -= r * vrot_i;

      constexpr double mur = 0;
      double threshold_mom = std::abs(mur * exanb::norm(branch) * exanb::norm(fn)); // even without fabs, the value should
                                                                                    // be positive
      double mom_square = exanb::dot(mom_i, mom_i);
      if (mom_square > 0.0 && mom_square > threshold_mom * threshold_mom)
        mom_i = mom_i * (threshold_mom / sqrt(mom_square));
      //*/
    }


  template<> // cohesive
    ONIKA_HOST_DEVICE_FUNC inline void contact_force_core<true>(const double dn,
        const Vec3d &n, // -normal
        const double dt,
        const ContactParams &hkp,
        const double meff,
        Vec3d &ft, // tangential force between particle i and j
        const Vec3d &contact_position,
        const Vec3d &pos_i,  // positions i
        const Vec3d &vel_i,  // positions i
        Vec3d &f_i,          // forces i
        Vec3d &mom_i,        // moments i
        const Vec3d &vrot_i, // angular velocities i
        const Vec3d &pos_j,  // positions j
        const Vec3d &vel_j,  // positions j
        const Vec3d &vrot_j  // angular velocities j
        )
    {
      if( dn >= 0 ) // dn <= hkp.dncut if contact
      {
        const double fn_value = (hkp.fc / hkp.dncut) * dn - hkp.fc;
        f_i += fn_value * n;
        ft = {0, 0, 0};
        return;
      }

      const double damp = compute_damp(hkp.damp_rate, hkp.kn, meff);

      // === Relative velocity (j relative to i)
      auto vel = compute_relative_velocity(contact_position, pos_i, vel_i, vrot_i, pos_j, vel_j, vrot_j);

      // compute relative velocity
      const double vn = exanb::dot(vel, n);

      // === Normal force (elatic contact + viscous damping)
      const double fn = exaDEM::compute_normal_force_value_with_cohesive_force(hkp.kn, hkp.fc, damp, dn, vn);

      // === Tangential force (friction)
      ft += exaDEM::compute_tangential_force(hkp.kt, dt, vn, n, vel);

      // fit tangential force
      auto threshold_ft = exaDEM::compute_threshold_ft_with_cohesive_force(hkp.mu, fn, hkp.fc);
      exaDEM::fit_tangential_force(threshold_ft, ft);

      // === sum forces
      f_i = fn * n + ft;

      // === update moments
      mom_i += hkp.kr * (vrot_j - vrot_i) * dt;

      ///*
      // test
      Vec3d branch = contact_position - pos_i;
      double r = (exanb::dot(branch, vrot_i)) / (exanb::dot(vrot_i, vrot_i));
      branch -= r * vrot_i;

      constexpr double mur = 0;
      double threshold_mom = std::abs(mur * exanb::norm(branch) * fn); // even without fabs, the value should
                                                                                    // be positive
      double mom_square = exanb::dot(mom_i, mom_i);
      if (mom_square > 0.0 && mom_square > threshold_mom * threshold_mom)
        mom_i = mom_i * (threshold_mom / sqrt(mom_square));
      //*/
    }


  ONIKA_HOST_DEVICE_FUNC inline Vec3d compute_moments(const Vec3d &contact_position,
      const Vec3d &p, // position
      const Vec3d &f, // forces
      const Vec3d &m) // I.mom
  {
    const auto Ci = (contact_position - p);
    const auto Pimoment = exanb::cross(Ci, f) + m;
    return Pimoment;
  }
} // namespace exaDEM
