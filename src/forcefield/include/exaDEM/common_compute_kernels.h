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
        const double damp_rate, 
        const double kn, 
        const double meff)
    {
      const double ret = damp_rate * 2.0 * sqrt(kn * meff);
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
  // dn interpenetration
  // n normal direction (normalized)
  // vn relative velocity
  template<bool cohesive>
    ONIKA_HOST_DEVICE_FUNC 
    inline Vec3d compute_normal_force(
        const double kn, 
        const double fc, 
        const double damp, 
        const double dn, 
        const double vn, 
        const Vec3d &n);

  template<> 
    ONIKA_HOST_DEVICE_FUNC 
    inline Vec3d compute_normal_force<false>(
        const double kn, 
        const double fc, 
        const double damp, 
        const double dn, 
        const double vn, 
        const Vec3d &n)
    {
      const double fne = -kn * dn;  // elastic contact
      const double fnv = damp * vn; // viscous damping
      const double fn = fnv + fne;
      const auto ret = fn * n;
      return ret;
    }

  //
  template<> 
    ONIKA_HOST_DEVICE_FUNC 
    inline Vec3d compute_normal_force<true>(
        const double kn, 
        const double fc, 
        const double damp, 
        const double dn, 
        const double vn, 
        const Vec3d &n)
    {
      const double fne = -kn * dn;  // elastic contact
      const double fnv = damp * vn; // viscous damping
      const double fn_norm = fnv + fne - fc;
      const double fn = fn_norm < (-fc) ? -fc : fn_norm;
      const auto   ret = fn * n;

      return ret;
    }

  // === Tangential force (friction)
  ONIKA_HOST_DEVICE_FUNC 
    inline Vec3d compute_tangential_force(
        const double kt, 
        const double dt, 
        const double vn, 
        const Vec3d &n, 
        const Vec3d &vel)
    {
      const Vec3d vt = vel - vn * n;
      const Vec3d ft = kt * (dt * vt);
      return ft;
    }

  ONIKA_HOST_DEVICE_FUNC 
    inline double compute_effective_mass(
        const double mi, 
        const double mj)
    {
      const double ret = (mi * mj) / (mi + mj);
      return ret;
    }


  ONIKA_HOST_DEVICE_FUNC 
    inline double compute_effective_radius(
        const double ri, 
        const double rj)
    {
      const double ret = (ri * rj) / (ri + rj);
      return ret;
    }

  ONIKA_HOST_DEVICE_FUNC 
    inline double compute_threshold_ft(
        const double mu, 
        const double kn, 
        const double dn)
    {
      // === recompute fne
      const double fne = kn * dn; //(remove -)
      const double threshold_ft = std::fabs(mu * fne);
      return threshold_ft;
    }

  ONIKA_HOST_DEVICE_FUNC 
    inline double compute_threshold_ft_with_cohesive_force(
        const double mu, 
        const double fn, 
        const double fc)
    {
      const double threshold_ft = std::fabs(mu * (fn + fc));
      return threshold_ft;
    }

  ONIKA_HOST_DEVICE_FUNC 
    inline void fit_tangential_force(
        const double threshold_ft, 
        Vec3d &ft)
    {
      double ft_square = exanb::dot(ft, ft);
      if (ft_square > 0.0 && ft_square > threshold_ft * threshold_ft)
        ft *= (threshold_ft / sqrt(ft_square));
    }
}; // namespace exaDEM
