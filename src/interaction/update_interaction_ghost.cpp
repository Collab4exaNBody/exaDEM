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
#include <mpi.h>

// exaNBody
#include <exanb/mpi/ghosts_comm_scheme.h>

// exaDEM
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/traversal.hpp>
#include <exaDEM/interaction/interaction_ghost_manager.hpp>

namespace exaDEM {
template <typename GridT, class = AssertGridHasFields<GridT>>
class UpdateInteractionGhost : public OperatorNode {
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(GhostCommunicationScheme, ghost_comm_scheme, INPUT, REQUIRED);
  ADD_SLOT(GridCellParticleInteraction, ges, INPUT_OUTPUT, DocString{"Interaction list"});
  ADD_SLOT(InteractionGhostManager, interaction_ghost_manager, INPUT_OUTPUT, DocString{""});

 public:
  inline std::string documentation() const final {
    return R"EOF(
       This operator ...
       )EOF";
  }

  inline void execute() final {
    // Get slots
    auto& interaction_cells = ges->m_data;
    InteractionGhostManager& manager = *interaction_ghost_manager;
    auto& g = *grid;
    auto cells = g.cells();

    // Reset Interaction within the grid ghost layer
#pragma omp parallel for
    for (size_t i = 0; i < g.number_of_cells(); i++) {
      if (!g.is_ghost_cell(i)) {
        continue;
      }
      const unsigned int n_particles = cells[i].size();
      CellExtraDynamicDataStorageT<PlaceholderInteraction>& storage = interaction_cells[i];
      storage.initialize(n_particles);
    }

    // MPI comms are done here
    manager.setup(*ghost_comm_scheme, *mpi, interaction_cells, g);

    // Fill ghost layers with
    manager.copy_interaction(g, interaction_cells);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(update_interaction_ghost) {
  OperatorNodeFactory::instance()->register_factory("update_interaction_ghost",
                                                    make_grid_variant_operator<UpdateInteractionGhost>);
}
}  // namespace exaDEM
