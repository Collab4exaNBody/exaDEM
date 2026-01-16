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

#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/particle_neighbors/chunk_neighbors_apply.h>

#include <mpi.h>

#include <exaDEM/traversal.hpp>
#include <exaDEM/drivers.hpp>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/interaction_manager.hpp>
#include <exaDEM/interaction/migration_test.hpp>
#include <exaDEM/shapes.hpp>
#include <exaDEM/shape_detection.hpp>
#include <exaDEM/shape_detection_driver.hpp>
#include <exaDEM/sphere/nbh_sphere.hpp>

namespace exaDEM {
template <typename GridT>
class UpdateContactInteractionSphere : public OperatorNode {
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(exanb::GridChunkNeighbors, chunk_neighbors, INPUT, OPTIONAL, DocString{"Neighbor list"});
  ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, REQUIRED, DocString{"List of Drivers"});
  ADD_SLOT(double, rcut_inc, INPUT, REQUIRED,
           DocString{"value added to the search distance to update neighbor list less frequently. in physical space"});
  ADD_SLOT(bool, symetric, INPUT, REQUIRED, DocString{"Activate the use of symetric feature (contact law)"});
  ADD_SLOT(GridCellParticleInteraction, ges, INPUT_OUTPUT, DocString{"Interaction list"});
  ADD_SLOT(Traversal, traversal_real, INPUT, REQUIRED, DocString{"list of non empty cells within the current grid"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator add a stl mesh to the drivers list.

        YAML example:

          - nbh_sphere:
             symetric: true)EOF";
  }

  inline void execute() final {
    auto& g = *grid;
    const auto cells = g.cells();
    const size_t n_cells = g.number_of_cells();
    const IJK dims = g.dimension();
    auto& interactions = ges->m_data;
    double rVerlet = *rcut_inc;
    bool sym = *symetric;
    // if grid structure (dimensions) changed, we invalidate thie whole data
    if (interactions.size() != n_cells) {
      ldbg << "number of cells has changed, reset friction data" << std::endl;
      interactions.clear();
      interactions.resize(n_cells);
    }
    assert(interactions.size() == n_cells);

    if (!chunk_neighbors.has_value()) {
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < n_cells; i++) interactions[i].initialize(0);
      return;
    }

    auto [cell_ptr, cell_size] = traversal_real->info();

#pragma omp parallel
    {
      // local storage per thread
      PlaceholderInteraction item;
      InteractionManager manager;
#pragma omp for schedule(dynamic)
      for (size_t ci = 0; ci < cell_size; ci++) {
        item.pair.ghost = InteractionPair::NotGhost;
        item.pair.swap = false;
        size_t cell_a = cell_ptr[ci];
        IJK loc_a = grid_index_to_ijk(dims, cell_a);

        const unsigned int n_particles = cells[cell_a].size();
        CellExtraDynamicDataStorageT<PlaceholderInteraction>& storage = interactions[cell_a];

        assert(interaction_test::check_extra_interaction_storage_consistency(
            storage.number_of_particles(), storage.m_info.data(), storage.m_data.data()));

        if (n_particles == 0) {
          storage.initialize(0);
          continue;
        }

        // Extract history before reset it
        const size_t data_size = storage.m_data.size();
        PlaceholderInteraction* __restrict__ data_ptr = storage.m_data.data();
        extract_history(manager.hist, data_ptr, data_size);
        std::sort(manager.hist.begin(), manager.hist.end());
        manager.reset(n_particles);

        // Reset storage, interaction history was stored in the manager
        storage.initialize(n_particles);
        auto& info_particles = storage.m_info;

        // Get data pointers
        const uint64_t* __restrict__ id_a = cells[cell_a][field::id];
        ONIKA_ASSUME_ALIGNED(id_a);
        const double* __restrict__ rx = cells[cell_a][field::rx];
        ONIKA_ASSUME_ALIGNED(rx);
        const double* __restrict__ ry = cells[cell_a][field::ry];
        ONIKA_ASSUME_ALIGNED(ry);
        const double* __restrict__ rz = cells[cell_a][field::rz];
        ONIKA_ASSUME_ALIGNED(rz);
        const double* __restrict__ rad = cells[cell_a][field::radius];
        ONIKA_ASSUME_ALIGNED(rad);

        // Fill particle ids in the interaction storage
        for (size_t it = 0; it < n_particles; it++) {
          info_particles[it].pid = id_a[it];
        }

        item.clear_placeholder();
        auto& pi = item.i();       // particle i (id, cell, pos, sub)
        auto& pd = item.driver();  // particle driver (id, cell, pos, sub)
        pi.cell = cell_a;

        // First, interaction between a sphere and a driver
        if (drivers.has_value()) {
          auto& drvs = *drivers;
          // By default, if the interaction is between a particle and a driver
          // Data about the particle j is set to -1
          // Except for id_j that contains the driver id
          pd.id = decltype(pd.id)(-1);
          pd.cell = decltype(pd.cell)(-1);
          pd.p = decltype(pd.p)(-1);

          for (size_t drvs_idx = 0; drvs_idx < drvs.get_size(); drvs_idx++) {
            pd.id = drvs_idx;  // we store the driver idx
            DRIVER_TYPE type = drvs.type(drvs_idx);

            if (type == DRIVER_TYPE::UNDEFINED) {
              continue;
            }

            if (type == DRIVER_TYPE::CYLINDER) {
              item.pair.type = InteractionTypeId::VertexCylinder;
              pd.id = drvs_idx;
              Cylinder& driver = drvs.get_typed_driver<Cylinder>(drvs_idx);
              for (size_t p = 0; p < n_particles; p++) {
                const Vec3d r = {rx[p], ry[p], rz[p]};
                const double rVerletMax = rad[p] + rVerlet;
                if (driver.filter(rVerletMax, r)) {
                  pi.p = p;
                  pi.id = id_a[p];
                  manager.add_item(item);
                }
              }
            } else if (type == DRIVER_TYPE::SURFACE) {
              item.pair.type = InteractionTypeId::VertexSurface;
              pd.id = drvs_idx;
              Surface& driver = drvs.get_typed_driver<Surface>(drvs_idx);
              for (size_t p = 0; p < n_particles; p++) {
                const Vec3d r = {rx[p], ry[p], rz[p]};
                const double rVerletMax = rad[p] + rVerlet;
                if (driver.filter(rVerletMax, r)) {
                  pi.p = p;
                  pi.id = id_a[p];
                  manager.add_item(item);
                }
              }
            } else if (type == DRIVER_TYPE::BALL) {
              item.pair.type = InteractionTypeId::VertexBall;
              pd.id = drvs_idx;
              Ball& driver = drvs.get_typed_driver<Ball>(drvs_idx);
              for (size_t p = 0; p < n_particles; p++) {
                const Vec3d r = {rx[p], ry[p], rz[p]};
                const double rVerletMax = rad[p] + rVerlet;
                if (driver.filter(rVerletMax, r)) {
                  pi.p = p;
                  pi.id = id_a[p];
                  manager.add_item(item);
                }
              }
            } else if (type == DRIVER_TYPE::STL_MESH) {
              auto& driver = drvs.get_typed_driver<Stl_mesh>(drvs_idx);
              for (size_t p = 0; p < n_particles; p++) {
                // a sphere can have multiple interactions with a stl mesh
                auto items =
                    detection_sphere_driver(driver, cell_a, p, id_a[p], drvs_idx, rx[p], ry[p], rz[p], rad[p], rVerlet);
                for (auto& it : items) {
                  manager.add_item(it);
                }
              }
            }
          }
        }

        item.pair.type = InteractionTypeId::VertexVertex;

        if (sym) {
          // Second, we add interactions between two spheres.
          apply_cell_particle_neighbors(
              *grid, *chunk_neighbors, cell_a, loc_a, std::false_type() /* not symetric */,
              [&g, &manager, &cells, cell_a, &item, id_a](int p_a, size_t cell_b, unsigned int p_b,
                                                          size_t p_nbh_index) {
                // default value of the interaction studied (A or i -> B or j)
                const uint64_t id_nbh = cells[cell_b][field::id][p_b];
                if (id_a[p_a] >= id_nbh) {
                  if (!g.is_ghost_cell(cell_b)) {
                    return;
                  }
                }

                auto& pi = item.i();  // particle i (id, cell id, particle position, sub vertex)
                auto& pj = item.j();  // particle i (id, cell id, particle position, sub vertex)

                // Add interactions
                pi.id = id_a[p_a];
                pi.p = p_a;
                pj.id = id_nbh;
                pj.p = p_b;
                pj.cell = cell_b;
                manager.add_item(item);
              });
        } else {
          // Second, we add interactions between two spheres.
          apply_cell_particle_neighbors(
              *grid, *chunk_neighbors, cell_a, loc_a, std::false_type() /* not symetric */,
              [&g, &manager, &cells, cell_a, &item, id_a](int p_a, size_t cell_b, unsigned int p_b,
                                                          size_t p_nbh_index) {
                // default value of the interaction studied (A or i -> B or j)
                const uint64_t id_nbh = cells[cell_b][field::id][p_b];
                item.pair.ghost = InteractionPair::NotGhost;
                item.pair.swap = false;

                if (id_a[p_a] >= id_nbh) {
                  return;
                }
                if (g.is_ghost_cell(cell_b)) {
                  item.pair.ghost = InteractionPair::OwnerGhost;
                }

                auto& pi = item.i();  // particle i (id, cell id, particle position, sub vertex)
                auto& pj = item.j();  // particle i (id, cell id, particle position, sub vertex)

                // Add interactions
                pi.id = id_a[p_a];
                pi.p = p_a;
                pj.id = id_nbh;
                pj.p = p_b;
                pj.cell = cell_b;
                manager.add_item(item);
              });
        }

        manager.update_extra_storage<true>(storage);

        assert(interaction_test::check_extra_interaction_storage_consistency(
            storage.number_of_particles(), storage.m_info.data(), storage.m_data.data()));

        assert(migration_test::check_info_value(storage.m_info.data(), storage.m_info.size(), 1e6));
      }  // GRID_OMP_FOR_END
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(nbh_sphere) {
  OperatorNodeFactory::instance()->register_factory("nbh_sphere",
                                                    make_grid_variant_operator<UpdateContactInteractionSphere>);
}
}  // namespace exaDEM
