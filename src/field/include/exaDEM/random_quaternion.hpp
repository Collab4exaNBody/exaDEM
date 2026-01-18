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
#include <exanb/compute/compute_cell_particles.h>
#include <exanb/grid_cell_particles/particle_region.h>
#include <onika/math/quaternion_operators.h>

namespace exaDEM {
struct RandomQuaternionFunctor {
  const ParticleRegionCSGShallowCopy region; /**< Shallow copy of a particle region. */

  inline void operator()(exanb::Quaternion& q) const {
    randomize(q);
  }

  inline void operator()(double rx, double ry, double rz, const uint64_t id, exanb::Quaternion& q) const {
    Vec3d r = {rx, ry, rz};
    if (region.contains(r, id)) {
      randomize(q);
    }
  }
};
}  // namespace exaDEM

namespace exanb {
template <>
struct ComputeCellParticlesTraits<exaDEM::RandomQuaternionFunctor> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = false;
};
}  // namespace exanb
