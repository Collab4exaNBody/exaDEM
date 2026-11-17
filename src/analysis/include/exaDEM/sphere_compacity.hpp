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

#include <onika/flat_tuple.h>
#include <onika/math/basic_types.h>

namespace exaDEM {
struct SphereVolumeValue {
  static constexpr double pi = 3.14159265358979323846;
  unsigned long long count_;  // number of spheres
  double sum_radius_;         // sum of radii
  double sum_volume_;         // sum of (4/3) pi r^3
  inline ONIKA_HOST_DEVICE_FUNC void operator+=(double r) {
    count_++;
    sum_radius_ += r;
    sum_volume_ += (4.0 / 3.0) * pi * r * r * r;
  }
};

struct ReduceSphereVolumeFunctor {
  ONIKA_HOST_DEVICE_FUNC inline void operator()(SphereVolumeValue& local, const double radius,
                                                exanb::reduce_thread_local_t = {}) const {
    local += radius;
  }

  ONIKA_HOST_DEVICE_FUNC inline void operator()(SphereVolumeValue& global, const SphereVolumeValue& local,
                                                exanb::reduce_thread_block_t) const {
    ONIKA_CU_ATOMIC_ADD(global.count_, local.count_);
    ONIKA_CU_ATOMIC_ADD(global.sum_radius_, local.sum_radius_);
    ONIKA_CU_ATOMIC_ADD(global.sum_volume_, local.sum_volume_);
  }

  ONIKA_HOST_DEVICE_FUNC inline void operator()(SphereVolumeValue& global, const SphereVolumeValue& local,
                                                exanb::reduce_global_t) const {
    ONIKA_CU_ATOMIC_ADD(global.count_, local.count_);
    ONIKA_CU_ATOMIC_ADD(global.sum_radius_, local.sum_radius_);
    ONIKA_CU_ATOMIC_ADD(global.sum_volume_, local.sum_volume_);
  }
};
}  // namespace exaDEM

namespace exanb {
template <>
struct ReduceCellParticlesTraits<exaDEM::ReduceSphereVolumeFunctor> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool RequiresCellParticleIndex = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace exanb
