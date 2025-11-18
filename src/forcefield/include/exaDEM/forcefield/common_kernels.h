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
#include <onika/cuda/cuda.h>

namespace exaDEM
{
  using exanb::Vec3d;

  ONIKA_HOST_DEVICE_FUNC 
    inline double compute_damp(
        const double a_damp_rate, 
        const double a_kn, 
        const double a_meff)
    {
      const double ret = a_damp_rate * 2.0 * sqrt(a_kn * a_meff);
      return ret;
    }

  ONIKA_HOST_DEVICE_FUNC 
    inline Vec3d compute_relative_velocity(
        const Vec3d &contact_position, 
        const Vec3d &pos_i, 
        const Vec3d &vel_i, 
        const Vec3d &vrot_i, 
        const Vec3d &pos_j, 
        const Vec3d &vel_j, 
        const Vec3d &vrot_j)
    {
      const auto contribution_i = vel_i - exanb::cross(contact_position - pos_i, vrot_i);
      const auto contribution_j = vel_j - exanb::cross(contact_position - pos_j, vrot_j);
      const auto ret = contribution_j - contribution_i;
      return ret;
    }

  // === Normal force (elatic contact + viscous damping)
  // a_dn interpenetration
  // a_n normal direction (normalized)
  // a_vn relative velocity
  ONIKA_HOST_DEVICE_FUNC 
    inline double normal_force(
        const double kn, 
        const double damp, 
        const double dn, 
        const double vn)
    {
      const double fne = -kn * dn;  // elastic contact
      const double fnv = damp * vn; // viscous damping
      return fnv + fne;
    }

  // === Normal force (elatic contact + viscous damping)
  // a_dn interpenetration
  // a_n normal direction (normalized)
  // a_vn relative velocity
  ONIKA_HOST_DEVICE_FUNC 
    inline Vec3d compute_normal_force(
        const double kn, 
        const double damp, 
        const double dn, 
        const double vn, 
        const Vec3d &n)
    {
      return normal_force(kn, damp, dn, vn) * n;
    }

  // === Tangential force (friction)
  ONIKA_HOST_DEVICE_FUNC 
    inline Vec3d compute_tangential_force(
        const double dt, 
        const double vn, 
        const Vec3d &n, 
        const Vec3d &vel)
    {
      const Vec3d vt = vel - vn * n;
      const Vec3d ft = (dt * vt);
      return ft;
    }

  ONIKA_HOST_DEVICE_FUNC 
    inline Vec3d compute_tangential_force(
        const double kt, 
        const double dt, 
        const double vn, 
        const Vec3d &n, 
        const Vec3d &vel)
    {
      return kt * compute_tangential_force(dt, vn, n, vel);
    }

  ONIKA_HOST_DEVICE_FUNC 
    inline Vec3d contribution_stick_tangential_force(
        const double damp, 
        const double vn, 
        const Vec3d &n, 
        const Vec3d &vel)
    {
      // Tangential viscosity ===================================
      // This term should be added only on the elastic part of ft
      // So it is somehow wrong because the viscosity is cumulated... Be carreful!
      return damp * (vel - vn * n); 
    }

  ONIKA_HOST_DEVICE_FUNC 
    inline double compute_effective_mass(
        const double a_mi, 
        const double a_mj)
    {
      const double ret = (a_mi * a_mj) / (a_mi + a_mj);
      return ret;
    }

  //
  ONIKA_HOST_DEVICE_FUNC 
    inline double compute_normal_force_value_with_cohesive_force(
        const double a_kn, 
        const double a_fc, 
        const double a_damp, 
        const double a_dn, 
        const double a_vn)
    {
      const double fne = -a_kn * a_dn;  // elastic contact
      const double fnv = a_damp * a_vn; // viscous damping
      const double fn = fnv + fne - a_fc;
      const double ret = fn < (-a_fc) ? -a_fc : fn;
      return ret;
    }

  ONIKA_HOST_DEVICE_FUNC 
    inline double compute_threshold_ft(
        const double a_mu, 
        const double a_kn, 
        const double a_dn)
    {
      // === recompute fne
      const double fne = a_kn * a_dn; //(remove -)
      const double threshold_ft = std::fabs(a_mu * fne);
      return threshold_ft;
    }

  ONIKA_HOST_DEVICE_FUNC 
    inline double compute_threshold_ft_with_cohesive_force(
        const double a_mu, 
        const double a_fn, 
        const double a_fc)
    {
      const double threshold_ft = std::fabs(a_mu * (a_fn + a_fc));
      return threshold_ft;
    }

  ONIKA_HOST_DEVICE_FUNC 
    inline void fit_tangential_force(
        const double threshold_ft, 
        Vec3d &a_ft)
    {
      double ft_square = exanb::dot(a_ft, a_ft);
      if (ft_square > 0.0 && ft_square > threshold_ft * threshold_ft)
        a_ft *= (threshold_ft / sqrt(ft_square));
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
}; // namespace exaDEM
