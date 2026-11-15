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

#include <exanb/core/grid.h>
#include <exanb/compute/compute_cell_particles.h>
#include <onika/math/basic_types_operators.h>
#include <onika/cuda/cuda.h>
#include <exaDEM/basic_type.hpp>

namespace exaDEM {
struct PushToAngularAccelerationFunctor {
  ONIKA_HOST_DEVICE_FUNC
  inline void operator()(const Quaternion& Q, const Vec3d& mom, const Vec3d& vrot, Vec3d& arot,
                         const Vec3d& inertia) const {
    Quaternion Qinv = get_conjugated(Q);
    const auto omega = Qinv * vrot;  // Express omega in the body framework
    const auto M = Qinv * mom;       // Express torque in the body framework
    Vec3d domega{(M.x - (inertia.z - inertia.y) * omega.y * omega.z) / inertia.x,
                 (M.y - (inertia.x - inertia.z) * omega.z * omega.x) / inertia.y,
                 (M.z - (inertia.y - inertia.x) * omega.x * omega.y) / inertia.z};
    arot = Q * domega;  // Express arot in the global framework
  }
};
}  // namespace exaDEM

namespace exanb {
template <>
struct ComputeCellParticlesTraits<exaDEM::PushToAngularAccelerationFunctor> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace exanb
