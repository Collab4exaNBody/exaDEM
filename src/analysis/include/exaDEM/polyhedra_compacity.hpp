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

#include <onika/cuda/cuda.h>
#include <onika/cuda/stl_adaptors.h>

#include <exanb/compute/reduce_cell_particles.h>

#include <exaDEM/shape.hpp>

namespace exaDEM {
struct PolyhedraVolumeValue {
  unsigned long long count_;  // number of polyhedra
  double sum_volume_;         // sum of the polyhedron volumes (shape volume * h^3)
};

struct ReducePolyhedraVolumeFunctor {
  onika::cuda::span<shape> shps_;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(PolyhedraVolumeValue& local, const uint16_t type, const double h,
                                                exanb::reduce_thread_local_t = {}) const {
    local.count_++;
    local.sum_volume_ += shps_[type].compute_volume(h);
  }

  ONIKA_HOST_DEVICE_FUNC inline void operator()(PolyhedraVolumeValue& global, const PolyhedraVolumeValue& local,
                                                exanb::reduce_thread_block_t) const {
    ONIKA_CU_ATOMIC_ADD(global.count_, local.count_);
    ONIKA_CU_ATOMIC_ADD(global.sum_volume_, local.sum_volume_);
  }

  ONIKA_HOST_DEVICE_FUNC inline void operator()(PolyhedraVolumeValue& global, const PolyhedraVolumeValue& local,
                                                exanb::reduce_global_t) const {
    ONIKA_CU_ATOMIC_ADD(global.count_, local.count_);
    ONIKA_CU_ATOMIC_ADD(global.sum_volume_, local.sum_volume_);
  }
};
}  // namespace exaDEM

namespace exanb {
template <>
struct ReduceCellParticlesTraits<exaDEM::ReducePolyhedraVolumeFunctor> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool RequiresCellParticleIndex = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace exanb
