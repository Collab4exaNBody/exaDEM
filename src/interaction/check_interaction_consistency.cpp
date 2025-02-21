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

#include <memory>

#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exanb/extra_storage/migration_test.hpp>
#include <exaDEM/interaction/migration_test.hpp>

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT>> class CheckInteractionConsistency : public OperatorNode
  {
    ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
    ADD_SLOT(GridCellParticleInteraction, ges, INPUT, DocString{"Interaction list"});

  public:
    inline std::string documentation() const override final
    {
      return R"EOF(
          "This opertor checks if a interaction related to a particle contains its particle id. (i.e. , I_id(i,j), id == item.id_i || item.id_j)"
                )EOF";
    }

    inline void execute() override final
    {
      if (grid->number_of_cells() == 0)
      {
        return;
      }
      auto &cell_interactions = ges->m_data;
      for (size_t current_cell = 0; current_cell < cell_interactions.size(); current_cell++)
      {
        auto storage = cell_interactions[current_cell];
        size_t n_particles_stored = storage.number_of_particles();
        auto *info_ptr = storage.m_info.data();
        auto *data_ptr = storage.m_data.data();
        [[maybe_unused]] bool is_okay = interaction_test::check_extra_interaction_storage_consistency(n_particles_stored, info_ptr, data_ptr);
        assert(is_okay && "CheckInteractionConsistency");
      }
    }
  };

  template <class GridT> using CheckInteractionConsistencyTmpl = CheckInteractionConsistency<GridT>;

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("check_interaction_consistency", make_grid_variant_operator<CheckInteractionConsistencyTmpl>); }
} // namespace exaDEM
