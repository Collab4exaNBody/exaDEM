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

#include <exaDEM/forcefield/common_kernels.hpp>
#include <exaDEM/forcefield/inner_bond_parameters.hpp>

namespace exaDEM {
ONIKA_HOST_DEVICE_FUNC
inline void force_law_core(const double dn,
                           const Vec3d& n,  // -normal
                           const double dn0, const double weight, const double dt, const InnerBondParams& ibp,
                           const double meff, double& En,
                           Vec3d& tds,  // cumulative tangential displacement
                           double& Et,
                           Vec3d& ft,  // tangential force between particle i and j
                           const Vec3d& contact_position,
                           const Vec3d& pos_i,   // positions i
                           const Vec3d& vel_i,   // velocities i
                           Vec3d& f_i,           // forces i
                           const Vec3d& vrot_i,  // angular velocities i
                           const Vec3d& pos_j,   // positions j
                           const Vec3d& vel_j,   // velocities j
                           const Vec3d& vrot_j   // angular velocities j
) {
  // === Compute damping coefficient
  const double damp = compute_damp(ibp.damp_rate_, ibp.kn_, meff);

  // === Relative velocity (j relative to i)
  auto vel = compute_relative_velocity(contact_position, pos_i, vel_i, vrot_i, pos_j, vel_j, vrot_j);
  const double vn = exanb::dot(vel, n);

  // === Normal force (elastic contact + viscous damping)
  const double delta = dn - dn0;
  const double kn_w = ibp.kn_ * weight;
  const double kt_w = ibp.kt_ * weight;
  const double fne = -kn_w * delta;
  const double fnv = damp * vn;
  const double fn = fne + fnv;
  const Vec3d vfn = fn * n;  // vector fn

  // === Tangential force (friction)
  const Vec3d ds = compute_tangential_force(dt, vn, n, vel);
  tds += ds;
  ft = kt_w * tds;

  // === sum forces
  f_i = vfn + ft;

  // === Compute energies (branchless to avoid warp divergence on GPU)
  En = static_cast<double>(fne <= 0) * (0.5 * kn_w * delta * delta);  // 0 in compression, tension otherwise
  Et = 0.5 * kt_w * dot(tds, tds);  // 0.5 * kt * norm2(vt * dt); with vt = (vel - (vn * n))
}
}  // namespace exaDEM
