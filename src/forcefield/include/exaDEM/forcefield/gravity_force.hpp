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
#include <onika/parallel/parallel_for.h>
#include <exanb/compute/compute_cell_particles.h>

namespace exaDEM {
struct GravityForceDriverFunctor {
  exanb::Vec3d g = {0.0, 0.0, -9.807};
  template <typename T>
  inline void operator()(T& drv) const {
    if constexpr (std::is_same_v<std::remove_cv_t<T>, exaDEM::Stl_mesh>) {
      if(drv.motion_type == MotionType::PARTICLE) {
        drv.forces += g * drv.mass;
      }
    }
  }
};

struct GravityForceFunctor {
  exanb::Vec3d g = {0.0, 0.0, -9.807};
  ONIKA_HOST_DEVICE_FUNC inline
      void operator()(double mass, double& fx, double& fy, double& fz) const {
        fx += g.x * mass;
        fy += g.y * mass;
        fz += g.z * mass;
      }
};
}  // namespace exaDEM

namespace onika {
namespace parallel {
template <>
struct ParallelForFunctorTraits<exaDEM::GravityForceDriverFunctor> {
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika

namespace exanb {
template <>
struct ComputeCellParticlesTraits<exaDEM::GravityForceFunctor> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace exanb
