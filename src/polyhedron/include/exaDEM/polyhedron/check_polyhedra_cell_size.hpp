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
#include <exanb/compute/reduce_cell_particles.h>
#include <exaDEM/atomic.h>

#include <cstdint>

namespace exaDEM {
/** Reduces the largest polyhedron radius, h * rcut(shape), homothety included. */
struct ReducePolyhedraMaxRadiusFunctor {
  const double* shape_rcut_;  // max rcut per shape type, host/device accessible

  ONIKA_HOST_DEVICE_FUNC inline void operator()(double& local, const uint16_t type, const double h,
                                                exanb::reduce_thread_local_t = {}) const {
    const double r = h * shape_rcut_[type];
    local = (r > local) ? r : local;
  }
  ONIKA_HOST_DEVICE_FUNC inline void operator()(double& global, const double& local,
                                                exanb::reduce_thread_block_t) const {
    atomic_max(global, local);
  }
  ONIKA_HOST_DEVICE_FUNC inline void operator()(double& global, const double& local, exanb::reduce_global_t) const {
    atomic_max(global, local);
  }
};
}  // namespace exaDEM

namespace exanb {
template <>
struct ReduceCellParticlesTraits<exaDEM::ReducePolyhedraMaxRadiusFunctor> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool RequiresCellParticleIndex = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace exanb
