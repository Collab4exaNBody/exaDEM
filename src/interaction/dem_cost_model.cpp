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
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/fields.h>
#include <exanb/core/domain.h>
#include <exanb/core/cell_costs.h>

#include <vector>
#include <algorithm>
#include <limits>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>

namespace exaDEM
{
  using namespace exanb;

  // simple cost model where the cost of a cell is the number of particles in it
  //
  template <class GridT> class DEMCostModel : public OperatorNode
  {
    ADD_SLOT(GridT, grid, INPUT);
    ADD_SLOT(CellCosts, cell_costs, OUTPUT);
    ADD_SLOT(GridCellParticleInteraction, ges, INPUT_OUTPUT, DocString{"Interaction list"});

  public:
    inline void execute() override final
    {
      GridT &grid = *(this->grid);
      CellCosts &cell_costs = *(this->cell_costs);
      const int ghost_layers = grid.ghost_layers();
      std::vector<double> cost_weights;

      // Warning: for correctness we only account for inner cells (not ghost cells)
      IJK grid_dims = grid.dimension();
      auto cells = grid.cells();
      cell_costs.m_block.start = grid.offset() + ghost_layers;
      cell_costs.m_block.end = (grid.offset() + grid_dims) - ghost_layers;

      assert(cell_costs.m_block.end.i >= cell_costs.m_block.start.i);
      assert(cell_costs.m_block.end.j >= cell_costs.m_block.start.j);
      assert(cell_costs.m_block.end.k >= cell_costs.m_block.start.k);

      IJK dims = dimension(cell_costs.m_block);
      const size_t n_cells = grid_cell_count(dims);
      cell_costs.m_costs.resize(n_cells, 0.);
      auto &interactions = ges->m_data;
      bool skip_interaction = interactions.size() == 0 ? true : false;

#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN (dims, i, loc, schedule(static))
        {
          const size_t cell_i = grid_ijk_to_index(grid_dims, loc + ghost_layers);
          const size_t N = cells[cell_i].size();
          double cost = 3 * N + 1;
          if (!skip_interaction)
          {
            CellExtraDynamicDataStorageT<Interaction> &storage = interactions[cell_i];
            auto *__restrict__ data = storage.m_data.data();
            size_t size = storage.m_data.size();
            for (size_t i = 0; i < size; i++)
            {
              auto &type = data[i].type;
              if (type == 0)
                cost += 1; // vertex - vertex
              if (type == 1)
                cost += 3; // vertex - edge
              if (type == 2)
                cost += 5; // vertex - face
              if (type == 3)
                cost += 4; // edge - edge
              if (type == 4)
                cost += 1; // cylinder
              if (type == 5)
                cost += 1; // wall
              if (type == 6)
                cost += 1; // balls
              if (type == 7)
                cost += 1; // vertex - vertex
              if (type == 8)
                cost += 3; // vertex - edge
              if (type == 9)
                cost += 5; // vertex - face
              if (type == 10)
                cost += 4; // edge - edge
              if (type == 11)
                cost += 3; // edge - vertex
              if (type == 12)
                cost += 5; // face - vertex
            }
          }
          cell_costs.m_costs[i] = cost;
        }
        GRID_OMP_FOR_END
      }
    }
  };

  template <class GridT> using DEMCostModelTmpl = DEMCostModel<GridT>;

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("dem_cost_model", make_grid_variant_operator<DEMCostModelTmpl>); }
} // namespace exaDEM
