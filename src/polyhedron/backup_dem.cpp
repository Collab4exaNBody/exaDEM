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
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/grid.h>
#include <onika/math/basic_types.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid_fields.h>
#include <exanb/core/domain.h>
#include <exaDEM/polyhedron/backup_dem.hpp>

namespace exaDEM {
template <typename GridT>
struct DEMBackupNode : public OperatorNode {
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(Domain, domain, INPUT, REQUIRED);
  ADD_SLOT(DEMBackupData, backup_dem, INPUT_OUTPUT);

  inline void execute() final {
    IJK dims = grid->dimension();
    auto cells = grid->cells();
    const ssize_t gl = grid->ghost_layers();
    auto& bd = *backup_dem;

    const bool defbox = !domain->xform_is_identity();
    Mat3d m_xform = domain->xform();
    setup_dem_backup(bd, cells, dims);

    double* data = bd.m_data.data();
    uint32_t* idxs = bd.m_index_map.data();

#   pragma omp parallel
    {
      GRID_OMP_FOR_BEGIN(dims - 2 * gl, _, loc_no_gl) {
        const IJK loc = loc_no_gl + gl;
        const size_t i = grid_ijk_to_index(dims, loc);
        const size_t n_particles = cells[i].size();
        double* rb = data + idxs[i];
        const auto* __restrict__ rx = cells[i][field::rx];
        const auto* __restrict__ ry = cells[i][field::ry];
        const auto* __restrict__ rz = cells[i][field::rz];
        const auto* __restrict__ orient = cells[i][field::orient];

#       pragma omp simd
        for (size_t j = 0; j < n_particles; j++) {
          const size_t p = j*DEMBackupData::components;
          if (defbox) {
            Vec3d r = m_xform * Vec3d{rx[j], ry[j], rz[j]};
            rb[p] = r.x;
            rb[p + 1] = r.y;
            rb[p + 2] = r.z;
          } else {
            rb[p] = rx[j];
            rb[p + 1] = ry[j];
            rb[p + 2] = rz[j];
          }
          rb[p + 3] = orient[j].w;
          rb[p + 4] = orient[j].x;
          rb[p + 5] = orient[j].y;
          rb[p + 6] = orient[j].z;
        }
      }
      GRID_OMP_FOR_END
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(backup_dem) {
  OperatorNodeFactory::instance()->register_factory("backup_dem", make_grid_variant_operator<DEMBackupNode>);
}
}  // namespace exaDEM
