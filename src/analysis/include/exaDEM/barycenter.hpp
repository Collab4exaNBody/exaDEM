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

#include <exanb/grid_cell_particles/particle_region.h>
#include <onika/flat_tuple.h>
#include <onika/math/basic_types.h>

namespace exaDEM {
struct ParticleBarycenterTypeValue {
  int count_;         // the number of particles of a type
  Vec3d barycenter_;  // sum of particle centers
  inline ONIKA_HOST_DEVICE_FUNC void operator+=(Vec3d& r) {
    count_++;
    barycenter_ += r;
  }
};

struct ReduceParticleBarycenterTypeFunctor {
  const ParticleRegionCSGShallowCopy region_;
  const uint16_t filter_type_;
  ONIKA_HOST_DEVICE_FUNC inline void operator()(ParticleBarycenterTypeValue& local, const double rx, const double ry,
                                                const double rz, const uint16_t type,
                                                reduce_thread_local_t = {}) const {
    Vec3d r = {rx, ry, rz};
    if (region_.contains(r)) {
      if (type == filter_type_) {
        local += r;
      }
    }
  }

  ONIKA_HOST_DEVICE_FUNC inline void operator()(ParticleBarycenterTypeValue& global,
                                                const ParticleBarycenterTypeValue& local, reduce_thread_block_t) const {
    ONIKA_CU_ATOMIC_ADD(global.count_, local.count_);
    ONIKA_CU_ATOMIC_ADD(global.barycenter_.x, local.barycenter_.x);
    ONIKA_CU_ATOMIC_ADD(global.barycenter_.y, local.barycenter_.y);
    ONIKA_CU_ATOMIC_ADD(global.barycenter_.z, local.barycenter_.z);
  }

  ONIKA_HOST_DEVICE_FUNC inline void operator()(ParticleBarycenterTypeValue& global,
                                                const ParticleBarycenterTypeValue& local, reduce_global_t) const {
    ONIKA_CU_ATOMIC_ADD(global.count_, local.count_);
    ONIKA_CU_ATOMIC_ADD(global.barycenter_.x, local.barycenter_.x);
    ONIKA_CU_ATOMIC_ADD(global.barycenter_.y, local.barycenter_.y);
    ONIKA_CU_ATOMIC_ADD(global.barycenter_.z, local.barycenter_.z);
  }
};
}  // namespace exaDEM

namespace exanb {
template <>
struct ReduceCellParticlesTraits<exaDEM::ReduceParticleBarycenterTypeFunctor> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool RequiresCellParticleIndex = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace exanb
