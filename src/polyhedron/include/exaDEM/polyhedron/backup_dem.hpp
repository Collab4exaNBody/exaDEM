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
#include <exanb/compute/reduce_cell_particles.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_math.h>
#include <onika/memory/allocator.h>

#include <cmath>
#include <cstdint>
#include <exaDEM/shapes.hpp>
#include <vector>

namespace exaDEM {
struct DEMBackupData {
  using DEMBackupVectorData = onika::memory::CudaMMVector<double>;
  using DEMBackupVectorIdx = onika::memory::CudaMMVector<uint32_t>;
  DEMBackupVectorData data_;
  DEMBackupVectorIdx index_map_;
  static constexpr size_t components = 7;
};

template <typename TMPLC>
void setup_dem_backup(DEMBackupData& backup_dem, TMPLC& cells, const IJK dims) {
  /// CPU
  backup_dem.index_map_.resize(dims.i * dims.j * dims.k);
  uint32_t shift = 0;
  uint32_t* index_map = onika::cuda::vector_data(backup_dem.index_map_);

  GRID_FOR_BEGIN(dims, i, _) {
    index_map[i] = DEMBackupData::components * shift;
    const size_t n_particles = cells[i].size();
    shift += n_particles;
  }
  GRID_FOR_END
  backup_dem.data_.resize(DEMBackupData::components * shift);
}

template <bool defbox>
struct ReduceMaxVertexDisplacementFunctor {
  const double* backup_data_ = nullptr;
  const uint32_t* backup_cell_idx_ = nullptr;
  const double threshold_sqr_ = 0.0;
  const shape* shps_;
  Mat3d xform_;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(unsigned long long int& count_over_dist2, IJK cell_loc, size_t cell,
                                                size_t j, double rx, double ry, double rz, uint32_t type,
                                                double homothety, const exanb::Quaternion& orientation,
                                                reduce_thread_local_t = {}) const {
    const double* __restrict__ rb = backup_data_ + backup_cell_idx_[cell];

    Vec3d new_center = {rx, ry, rz};
    if constexpr (defbox) {
      new_center = xform_ * new_center;
    }

    // try another data layout
    size_t p = DEMBackupData::components * j;
    Quaternion old_orientation = {rb[p + 3], rb[p + 4], rb[p + 5], rb[p + 6]};
    Vec3d old_center = {rb[p], rb[p + 1], rb[p + 2]};  // xform * pos at old timestep

    const auto& shp = shps_[type];
    const int nv = shp.get_number_of_vertices();
    for (int v = 0; v < nv; v++) {
      const Vec3d old_vertex = shp.get_vertex(v, old_center, homothety, old_orientation);
      const Vec3d new_vertex = shp.get_vertex(v, new_center, homothety, orientation);
      const Vec3d dr = new_vertex - old_vertex;
      if (exanb::dot(dr, dr) >= threshold_sqr_) {
        ++count_over_dist2;
      }
    }
  }
  ONIKA_HOST_DEVICE_FUNC inline void operator()(unsigned long long int& count_over_dist2, unsigned long long int value,
                                                reduce_thread_block_t) const {
    ONIKA_CU_BLOCK_ATOMIC_ADD(count_over_dist2, value);
  }

  ONIKA_HOST_DEVICE_FUNC inline void operator()(unsigned long long int& count_over_dist2, unsigned long long int value,
                                                reduce_global_t) const {
    if (value > 0) {
      ONIKA_CU_ATOMIC_ADD(count_over_dist2, value);
    }
  }
};
}  // namespace exaDEM

namespace exanb {
template <bool defbox>
struct ReduceCellParticlesTraits<exaDEM::ReduceMaxVertexDisplacementFunctor<defbox>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool RequiresCellParticleIndex = true;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace exanb
