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

#include <vector>
#include <cstdint>
#include <cmath>
#include <onika/memory/allocator.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_math.h>
#include <exanb/compute/reduce_cell_particles.h>
#include <exaDEM/shapes.hpp>

namespace exaDEM
{

  using namespace exanb;
  struct DEMBackupData
  {
    using CellDEMBackupVector = onika::memory::CudaMMVector<double>;
    using DEMBackupVector = onika::memory::CudaMMVector<CellDEMBackupVector>;
    DEMBackupVector m_data;
    Mat3d m_xform;
  };

  struct ReduceMaxVertexDisplacementFunctor
  {
    const DEMBackupData::CellDEMBackupVector *m_backup_data = nullptr;
    const double m_threshold_sqr = 0.0;
    const shape *shps;

    ONIKA_HOST_DEVICE_FUNC inline void operator()(unsigned long long int &count_over_dist2, IJK cell_loc, size_t cell, size_t j, double rx, double ry, double rz, uint32_t type, const exanb::Quaternion &orientation, reduce_thread_local_t = {}) const
    {
      const double *__restrict__ rb = onika::cuda::vector_data(m_backup_data[cell]);
      Vec3d new_center = {rx, ry, rz};
#ifdef TRY_SOA
      //try another data layout
      const size_t size = onika::cuda::vector_size(m_backup_data[cell]) / 7;
      Quaternion old_orientation = {rb[j + size * 3], rb[j + size * 4], rb[j + size * 5], rb[j + size * 6]};
      Vec3d old_center = {rb[j], rb[j + size * 1], rb[j + size * 2]};
#else
      Quaternion old_orientation = {rb[j * 7 + 3], rb[j * 7 + 4], rb[j * 7 + 5], rb[j * 7 + 6]};
      Vec3d old_center = {rb[j * 7 + 0], rb[j * 7 + 1], rb[j * 7 + 2]};
#endif
      const auto &shp = shps[type];
      const int nv = shp.get_number_of_vertices();
      for (int v = 0; v < nv; v++)
      {
        const Vec3d old_vertex = shp.get_vertex(v, old_center, old_orientation);
        const Vec3d new_vertex = shp.get_vertex(v, new_center, orientation);
        const Vec3d dr = new_vertex - old_vertex;
        if (exanb::dot(dr, dr) >= m_threshold_sqr)
        {
          ++count_over_dist2;
        }
      }
    }
    ONIKA_HOST_DEVICE_FUNC inline void operator()(unsigned long long int &count_over_dist2, unsigned long long int value, reduce_thread_block_t) const 
    {
      ONIKA_CU_BLOCK_ATOMIC_ADD(count_over_dist2, value); 
    }

    ONIKA_HOST_DEVICE_FUNC inline void operator()(unsigned long long int &count_over_dist2, unsigned long long int value, reduce_global_t) const 
    {
      if(value > 0)
      { 
        ONIKA_CU_ATOMIC_ADD(count_over_dist2, value); 
      }
    }
  };
} // namespace exaDEM

namespace exanb
{
  template <> struct ReduceCellParticlesTraits<exaDEM::ReduceMaxVertexDisplacementFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool RequiresCellParticleIndex = true;
    static inline constexpr bool CudaCompatible = true;
  };
} // namespace exanb
