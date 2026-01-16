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
                           const double dn0, const double dt, const InnerBondParams& ibp, const double meff, double& En,
                           double& Et,
                           Vec3d& vft,  // tangential force between particle i and j
                           const Vec3d& contact_position,
                           const Vec3d& pos_i,   // positions i
                           const Vec3d& vel_i,   // positions i
                           Vec3d& f_i,           // forces i
                           const Vec3d& vrot_i,  // angular velocities i
                           const Vec3d& pos_j,   // positions j
                           const Vec3d& vel_j,   // positions j
                           const Vec3d& vrot_j   // angular velocities j
) {
  // === Compute damping coefficient
  const double damp = compute_damp(ibp.damp_rate, ibp.kn, meff);

  // === Relative velocity (j relative to i)
  auto vel = compute_relative_velocity(contact_position, pos_i, vel_i, vrot_i, pos_j, vel_j, vrot_j);

  // compute relative velocity
  const double vn = exanb::dot(vel, n);
  // === Normal force (elatic contact + viscous damping)
  double fne = -ibp.kn * (dn - dn0);
  double fnv = damp * vn;
  double fn = fne + fnv;
  // double fn = normal_force(ibp.kn, damp, dn - dn0, vn);
  const Vec3d vfn = fn * n;

  // === Tangential force (friction)
  const Vec3d ft = compute_tangential_force(dt, vn, n, vel);
  vft += ibp.kt * ft;
  vft += exaDEM::contribution_stick_tangential_force(damp, vn, n, vel);

  // === sum forces
  f_i = vfn + vft;

  if (fne > 0) {
    En = 0;
  } else {
    En = 0.5 * ibp.kn * (dn - dn0) * (dn - dn0);
  }
  Et = 0.5 * ibp.kt * dot(ft, ft);  // 0.5 * kt * norm2(vt * dt); with  vt = (vel - (vn * n));
}
}  // namespace exaDEM
