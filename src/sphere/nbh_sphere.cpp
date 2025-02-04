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
//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/particle_neighbors/chunk_neighbors_apply.h>
#include <mpi.h>
#include <memory>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/interaction_manager.hpp>
#include <exaDEM/interaction/migration_test.hpp>
#include <exaDEM/drivers.h>
#include <exaDEM/shapes.hpp>
#include <exaDEM/shape_detection.hpp>
#include <exaDEM/shape_detection_driver.hpp>
#include <exaDEM/traversal.hpp>
#include <exaDEM/traversal.hpp>
#include <exaDEM/nbh_sphere.hpp>

namespace exaDEM
{

  using namespace exanb;

  template <typename GridT> class UpdateContactInteractionSphere : public OperatorNode
  {
    ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
    ADD_SLOT(exanb::GridChunkNeighbors, chunk_neighbors, INPUT, OPTIONAL, DocString{"Neighbor list"});
    ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, DocString{"List of Drivers"});
    ADD_SLOT(double, rcut_inc, INPUT, DocString{"value added to the search distance to update neighbor list less frequently. in physical space"});
    ADD_SLOT(bool, symetric, INPUT, REQUIRED, DocString{"Activate the use of symetric feature (contact law)"});
    ADD_SLOT(GridCellParticleInteraction, ges, INPUT_OUTPUT, DocString{"Interaction list"});
    ADD_SLOT(Traversal, traversal_real, INPUT, DocString{"list of non empty cells within the current grid"});

    public:
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator add a stl mesh to the drivers list.
        )EOF";
    }

    inline void execute() override final
    {
      auto &g = *grid;
      const auto cells = g.cells();
      const size_t n_cells = g.number_of_cells(); // nbh.size();
      const IJK dims = g.dimension();
      auto &interactions = ges->m_data;
      double rVerlet = *rcut_inc;
      bool sym = *symetric;
      // if grid structure (dimensions) changed, we invalidate thie whole data
      if (interactions.size() != n_cells)
      {
        ldbg << "number of cells has changed, reset friction data" << std::endl;
        interactions.clear();
        interactions.resize(n_cells);
      }
      assert(interactions.size() == n_cells);

      if (!chunk_neighbors.has_value())
      {
#       pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_cells; i++)
          interactions[i].initialize(0);
        return;
      }

      auto [cell_ptr, cell_size] = traversal_real->info();

#     pragma omp parallel
      {
        // local storage per thread
        Interaction item;
        interaction_manager manager;
#       pragma omp for schedule(dynamic)
        for (size_t ci = 0; ci < cell_size; ci++)
        {
          size_t cell_a = cell_ptr[ci];
          IJK loc_a = grid_index_to_ijk(dims, cell_a);

          const unsigned int n_particles = cells[cell_a].size();
          CellExtraDynamicDataStorageT<Interaction> &storage = interactions[cell_a];

          assert(interaction_test::check_extra_interaction_storage_consistency(storage.number_of_particles(), storage.m_info.data(), storage.m_data.data()));

          if (n_particles == 0)
          {
            storage.initialize(0);
            continue;
          }

          // Extract history before reset it
          const size_t data_size = storage.m_data.size();
          Interaction *__restrict__ data_ptr = storage.m_data.data();
          extract_history(manager.hist, data_ptr, data_size);
          std::sort(manager.hist.begin(), manager.hist.end());
          manager.reset(n_particles);

          // Reset storage, interaction history was stored in the manager
          storage.initialize(n_particles);

          item.type = 0; // === Vertex - Vertex


        } //    GRID_OMP_FOR_END
      }
    }
  };

  template <class GridT> using UpdateContactInteractionSphereTmpl = UpdateContactInteractionSphere<GridT>;

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("nbh_sphere", make_grid_variant_operator<UpdateContactInteractionSphereTmpl>); }
} // namespace exaDEM
