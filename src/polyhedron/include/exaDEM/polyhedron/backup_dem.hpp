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
#include <onika/memory/allocator.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_math.h>
#include <exanb/compute/reduce_cell_particles.h>

#include <vector>
#include <cstdint>
#include <cmath>
#include <exaDEM/shapes.hpp>

namespace exaDEM {
struct DEMBackupData {
/*
  using CellDEMBackupVector = onika::memory::CudaMMVector<double>;
  using DEMBackupVector = onika::memory::CudaMMVector<CellDEMBackupVector>;
  DEMBackupVector m_data;
  */
  using DEMBackupVectorData = onika::memory::CudaMMVector<double>;
  using DEMBackupVectorIdx = onika::memory::CudaMMVector<uint32_t>;
  DEMBackupVectorData m_data;
  DEMBackupVectorIdx m_index_map;
  static constexpr size_t components = 7;
};

  template<typename TMPLC>
void setup_dem_backup(DEMBackupData& backup_dem, TMPLC& cells, const IJK dims)
{
  /// CPU
  backup_dem.m_index_map.resize(dims.i*dims.j*dims.k);
  uint32_t shift = 0;
  uint32_t* index_map = onika::cuda::vector_data(backup_dem.m_index_map);

  GRID_FOR_BEGIN(dims, i, _) { 
    index_map[i] = DEMBackupData::components*shift;
    const size_t n_particles = cells[i].size();
    shift += n_particles;
  } GRID_FOR_END
  backup_dem.m_data.resize(DEMBackupData::components*shift);
}

template <bool defbox>
struct ReduceMaxVertexDisplacementFunctor {
  const double* m_backup_data = nullptr;
  const uint32_t* m_backup_cell_idx = nullptr;
  const double m_threshold_sqr = 0.0;
  const shape* shps;
  Mat3d m_xform;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(unsigned long long int& count_over_dist2, IJK cell_loc, size_t cell,
      size_t j, double rx, double ry, double rz, uint32_t type,
      double homothety, const exanb::Quaternion& orientation,
      reduce_thread_local_t = {}) const {
    const double* __restrict__ rb = m_backup_data + m_backup_cell_idx[cell];

    Vec3d new_center = {rx, ry, rz};
    if constexpr (defbox) {
      new_center = m_xform * new_center;
    }

    // try another data layout
    size_t p = DEMBackupData::components*j;
    Quaternion old_orientation = {rb[p + 3], rb[p + 4], rb[p + 5], rb[p + 6]};
    Vec3d old_center = {rb[p], rb[p + 1], rb[p + 2]};  // xform * pos at old timestep

    const auto& shp = shps[type];
    const int nv = shp.get_number_of_vertices();
    for (int v = 0; v < nv; v++) {
      const Vec3d old_vertex = shp.get_vertex(v, old_center, homothety, old_orientation);
      const Vec3d new_vertex = shp.get_vertex(v, new_center, homothety, orientation);
      const Vec3d dr = new_vertex - old_vertex;
      if (exanb::dot(dr, dr) >= m_threshold_sqr) {
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
