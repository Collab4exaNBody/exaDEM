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
#include <exaDEM/traversal.h>
#include <cassert>
#include <exaDEM/polyhedron/vertices.hpp>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/interaction_manager.hpp>
#include <exaDEM/interaction/migration_test.hpp>
#include <exaDEM/shapes.hpp>


namespace exaDEM {
template <typename GridT, class = AssertGridHasFields<GridT>>
class UpdatePersistentInteractionsOperator : public OperatorNode {
  using ComputeFields = FieldSet<>;
  static constexpr ComputeFields compute_field_set{};

  ADD_SLOT(GridT, grid,
           INPUT_OUTPUT, REQUIRED);
  ADD_SLOT(Domain, domain,
           INPUT, REQUIRED);
  ADD_SLOT(GridCellParticleInteraction, ges,
           INPUT_OUTPUT,
           DocString{"Interaction list"});
  ADD_SLOT(Traversal, traversal_real,
           INPUT, REQUIRED,
           DocString{"list of non empty cells within the current grid"});
  ADD_SLOT(double, rcut_max,
           INPUT, REQUIRED,
           DocString{"."});

 public:
  inline std::string documentation() const final {
    return R"EOF(
       )EOF";
  }

  inline std::tuple<bool, uint16_t> locate_particle_id(
      const uint64_t* ids,
      size_t n,
      uint64_t id) {
    for (uint16_t i=0 ; i < n ; i++) {
      if (ids[i] == id) {
        return {true, i};
      }
    }
    return {false, 0};
  }

  inline void execute() final {
    constexpr uint64_t InvalidId = -1;
    const double Rmax = 2 * (*rcut_max);
    auto& g = *grid;
    const auto cells = g.cells();
    const IJK dims = g.dimension();
    auto &cell_interactions = ges->m_data;

    auto [cell_ptr, cell_size] = traversal_real->info();

#   pragma omp parallel
    {
#       pragma omp for schedule(dynamic)
      for (size_t ci = 0; ci < cell_size; ci++) {
        size_t cell_a = cell_ptr[ci];

        CellExtraDynamicDataStorageT<PlaceholderInteraction> &storage = cell_interactions[cell_a];

        if (storage.m_data.size() == 0) {
          continue;
        }

        const unsigned int n_particles = cells[cell_a].size();
        PlaceholderInteraction* const interactions = storage.m_data.data();

        // Get data pointers
        const uint64_t *__restrict__ id_a = cells[cell_a][field::id];
        ONIKA_ASSUME_ALIGNED(id_a);
        const double *__restrict__ rx_a = cells[cell_a][field::rx];
        ONIKA_ASSUME_ALIGNED(rx_a);
        const double *__restrict__ ry_a = cells[cell_a][field::ry];
        ONIKA_ASSUME_ALIGNED(ry_a);
        const double *__restrict__ rz_a = cells[cell_a][field::rz];
        ONIKA_ASSUME_ALIGNED(rz_a);

        if (storage.m_data.size() == 0) {
          continue;
        }

        // note that size > 0 (otherwise -> continue)
        [[maybe_unused]] uint64_t previous_a_id = InvalidId;
        [[maybe_unused]] uint64_t previous_b_id = InvalidId;
        for (size_t i=0 ; i < storage.m_data.size() ; i++) {
          // get the current interaction.
          PlaceholderInteraction& item =  interactions[i];
          // verify that this interaction is a persistent interaction.
          if (!item.persistent()) {
            continue;
          }
          // get members
          auto& particle_loc_a = item.pair.owner();
          auto& particle_loc_b = item.pair.partner();

          uint64_t id = particle_loc_a.id;

          // if( shift > 0 && previous_a_id == particle_loc_a.id )
          //    if( i > 0 && previous_a_id == particle_loc_a.id )
          //    {
          //        particle_loc_a.cell = interactions[i - 1].pair.owner().cell;
          //        particle_loc_a.p    = interactions[i - 1].pair.owner().p;
          //      }
          //      else
          {
            auto [find, p_a] = locate_particle_id(id_a, n_particles, id);
            particle_loc_a.cell  = cell_a;
            particle_loc_a.p = p_a;

            // check error
            if (!find) {
              color_log::error("update_persistent_interactions",
                               "The particle with the particle id "
                               + std::to_string(id)
                               + " is not located in the cell "
                               + std::to_string(cell_a));
            }
          }

          // reuse data
          //    if( previous_b_id == particle_loc_b.id )
          //    {
          //      assert(i>0);
          //      particle_loc_b.cell = interactions[i - 1].pair.partner().cell;
          //    particle_loc_b.p    = interactions[i - 1].pair.partner().p;
          //      }
          //      else
          {
            // looking for both cell and p values in current and other cells
            Vec3d r = {rx_a[particle_loc_a.p], ry_a[particle_loc_a.p], rz_a[particle_loc_a.p]};
            AABB cover_particle = { r - Rmax, r + Rmax};
            IJK max = g.locate_cell(cover_particle.bmax);
            IJK min = g.locate_cell(cover_particle.bmin);
            bool do_continue = true;
            for (int x = min.i; x <= max.i && do_continue; x++) {
              for (int y = min.j; y <= max.j && do_continue; y++) {
                for (int z = min.k; z <= max.k && do_continue; z++) {
                  IJK next = {x, y, z};
                  if (g.contains(next)) {
                    size_t cell_b = grid_ijk_to_index(dims, next);
                    const unsigned int nb = cells[cell_b].size();
                    const uint64_t *__restrict__ id_b = cells[cell_b][field::id];
                    ONIKA_ASSUME_ALIGNED(id_b);
                    for (uint16_t p_b=0 ; p_b < nb ; p_b++) {
                      if (id_b[p_b] == particle_loc_b.id) {
                        particle_loc_b.cell = cell_b;
                        particle_loc_b.p    = p_b;
                        item.pair.ghost = g.is_ghost_cell(cell_b) ? InteractionPair::OwnerGhost : InteractionPair::NotGhost;
                        do_continue = false;
                        break;
                      }
                    }
                  }
                }
              }
            }

            if (do_continue) {
              color_log::error("update_persistent_interactions",
                               "The particle b with the particle id "
                               + std::to_string(particle_loc_b.id)
                               + " has not been found");
            }
          }
          // update previous_ab_id
          previous_a_id = particle_loc_a.id;
          previous_b_id = particle_loc_b.id;
        }
        //          }
    }  //    GRID_OMP_FOR_END
  }
}
};

// === register factories ===
ONIKA_AUTORUN_INIT(nbh_polyhedron) {
  OperatorNodeFactory::instance()->register_factory(
      "update_persistent_interactions",
      make_grid_variant_operator<UpdatePersistentInteractionsOperator>);
}
}  // namespace exaDEM
