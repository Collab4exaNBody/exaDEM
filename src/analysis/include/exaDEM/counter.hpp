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
#include <onika/flat_tuple.h>
#include <exanb/grid_cell_particles/particle_region.h>

namespace exaDEM {
struct ReduceParticleCounterTypeFunctor {
  const ParticleRegionCSGShallowCopy region;
  const uint16_t filter_type;
  ONIKA_HOST_DEVICE_FUNC inline void operator()(int& local, const double rx, const double ry, const double rz,
                                                const uint16_t type, reduce_thread_local_t = {}) const {
    Vec3d r = {rx, ry, rz};
    if (region.contains(r)) {
      if (type == filter_type) {
        local++;
      }
    }
  }

  ONIKA_HOST_DEVICE_FUNC inline void operator()(int& global, int local, reduce_thread_block_t) const {
    ONIKA_CU_ATOMIC_ADD(global, local);
  }

  ONIKA_HOST_DEVICE_FUNC inline void operator()(int& global, int local, reduce_global_t) const {
    ONIKA_CU_ATOMIC_ADD(global, local);
  }
};
}  // namespace exaDEM

namespace exanb {
template <>
struct ReduceCellParticlesTraits<exaDEM::ReduceParticleCounterTypeFunctor> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool RequiresCellParticleIndex = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace exanb
