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
#include <onika/math/basic_types.h>
#include <onika/math/basic_types_operators.h>
#include <onika/math/basic_types_stream.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>
#include <exanb/core/domain.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>

#include <iomanip>
#include <vector>
#include <mpi.h>

#include <exaDEM/drivers.hpp>

namespace exaDEM {
using namespace exanb;

template <class GridT>
class UpdateGridRShapeOperator : public OperatorNode {
  using ComputeFields = FieldSet<field::_rx, field::_ry, field::_rz>;
  static constexpr ComputeFields compute_field_set{};
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD, DocString{"MPI communicator for parallel processing."});
  ADD_SLOT(GridT, grid, INPUT, REQUIRED, DocString{"Grid used for computations."});
  ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, REQUIRED, DocString{"List of Drivers"});
  ADD_SLOT(double, rcut_max, INPUT, REQUIRED, DocString{"rcut_max"});
  ADD_SLOT(bool, force_reset, INPUT, REQUIRED, DocString{"Force to rebuild grid for rshape meshes."});
  ADD_SLOT(std::vector<Vec3d>, grid_rshape_buffer, PRIVATE);

 public:
  inline std::string documentation() const final {
    return R"EOF(
      Update the list of information for each cell regarding the vertex,
      edge, and face indices in contact with the cell in an RShape."
    )EOF";
  }

  inline void execute() final {
    const auto& g = *grid;
    const size_t n_cells = g.number_of_cells();
    const IJK dims = g.dimension();
    const double Rmax = *rcut_max;
    bool ForceResetRShapeGrid = *force_reset;
    auto& gsb = *grid_rshape_buffer;

    for (size_t id = 0; id < drivers->get_size(); id++) {
      if (drivers->type(id) == DRIVER_TYPE::RSHAPE) {
        exaDEM::RShapeDriver& mesh = drivers->get_typed_driver<exaDEM::RShapeDriver>(id);
        auto& grid_rshape = mesh.grid_indexes;
        auto& mutexes = mesh.grid_mutexes;

        if (!ForceResetRShapeGrid) {
          if (mesh.stationary() && grid_rshape.size() == n_cells) {
            // The grid is already built and didn't change
            continue;
          }
        }

        gsb.resize(mesh.shp.get_number_of_vertices());  // we just need to get the upper size.
        mesh.shp.compute_prepro_obb(gsb.data(), mesh.center, mesh.quat);
        bool resize = grid_rshape.size() != n_cells;
        if (resize) {
          grid_rshape.resize(n_cells);
          mutexes.resize(n_cells);
        }

        if (resize) {
#pragma omp parallel for
          for (size_t i = 0; i < n_cells; i++) {
            omp_init_lock(&mutexes[i]);
          }
        }

#pragma omp parallel for
        for (size_t i = 0; i < n_cells; i++) {
          grid_rshape[i].clean();
        }

        auto& obb_v = mesh.shp.m_obb_vertices;
        auto& obb_e = mesh.shp.m_obb_edges;
        auto& obb_f = mesh.shp.m_obb_faces;

#pragma omp parallel
        {
#pragma omp for nowait
          for (size_t vid = 0; vid < obb_v.size(); vid++) {
            auto obb = obb_v[vid];
            obb.enlarge(Rmax);
            AABB aabb = conv_to_aabb(obb);
            IJK max = g.locate_cell(aabb.bmax);
            IJK min = g.locate_cell(aabb.bmin);
            for (int x = min.i; x <= max.i; x++) {
              for (int y = min.j; y <= max.j; y++) {
                for (int z = min.k; z <= max.k; z++) {
                  IJK next = {x, y, z};
                  if (g.contains(next)) {
                    AABB cell_aabb = g.cell_bounds(next);
                    OBB cell_obb = conv_to_obb(cell_aabb);
                    if (obb.intersect(cell_obb)) {
                      size_t cell_next_id = grid_ijk_to_index(dims, next);
                      omp_set_lock(&mutexes[cell_next_id]);
                      grid_rshape[cell_next_id].vertices.push_back(vid);
                      omp_unset_lock(&mutexes[cell_next_id]);
                    }
                  }
                }
              }
            }
          }

          // add edges
#pragma omp for nowait
          for (size_t eid = 0; eid < obb_e.size(); eid++) {
            auto obb = obb_e[eid];
            obb.enlarge(Rmax);
            AABB aabb = conv_to_aabb(obb);
            IJK max = g.locate_cell(aabb.bmax);
            IJK min = g.locate_cell(aabb.bmin);
            for (int x = min.i; x <= max.i; x++) {
              for (int y = min.j; y <= max.j; y++) {
                for (int z = min.k; z <= max.k; z++) {
                  IJK next = {x, y, z};
                  if (g.contains(next)) {
                    AABB cell_aabb = g.cell_bounds(next);
                    OBB cell_obb = conv_to_obb(cell_aabb);
                    if (obb.intersect(cell_obb)) {
                      size_t cell_next_id = grid_ijk_to_index(dims, next);
                      omp_set_lock(&mutexes[cell_next_id]);
                      grid_rshape[cell_next_id].edges.push_back(eid);
                      omp_unset_lock(&mutexes[cell_next_id]);
                    }
                  }
                }
              }
            }
          }

#pragma omp for nowait
          for (size_t fid = 0; fid < obb_f.size(); fid++) {
            auto obb = obb_f[fid];
            obb.enlarge(Rmax);
            AABB aabb = conv_to_aabb(obb);
            IJK max = g.locate_cell(aabb.bmax);
            IJK min = g.locate_cell(aabb.bmin);
            for (int x = min.i; x <= max.i; x++) {
              for (int y = min.j; y <= max.j; y++) {
                for (int z = min.k; z <= max.k; z++) {
                  IJK next = {x, y, z};
                  if (g.contains(next)) {
                    AABB cell_aabb = g.cell_bounds(next);
                    OBB cell_obb = conv_to_obb(cell_aabb);
                    if (obb.intersect(cell_obb)) {
                      size_t cell_next_id = grid_ijk_to_index(dims, next);
                      omp_set_lock(&mutexes[cell_next_id]);
                      grid_rshape[cell_next_id].faces.push_back(fid);
                      omp_unset_lock(&mutexes[cell_next_id]);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
};

// === register factories ==
template <class GridT>
using UpdateGridRShapeOperatorTemplate = UpdateGridRShapeOperator<GridT>;
ONIKA_AUTORUN_INIT(grid_rshape_mesh) {
  OperatorNodeFactory::instance()->register_factory("grid_rshape_driver",
                                                    make_grid_variant_operator<UpdateGridRShapeOperatorTemplate>);
}
}  // namespace exaDEM
