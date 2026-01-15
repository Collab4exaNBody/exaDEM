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
#include <exaDEM/mutexes.h>
#include <memory>

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT>> class UpdateMutexes : public OperatorNode
  {
    using ComputeFields = FieldSet<>;
    static constexpr ComputeFields compute_field_set{};

    ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
    ADD_SLOT(mutexes, locks, INPUT_OUTPUT);

  public:
    inline std::string documentation() const override final
    {
      return R"EOF(
				        )EOF";
    }

    inline void execute() override final
    {
      const auto &cells = grid->cells();
      const int n_cells = grid->number_of_cells();
      mutexes &cell_locks = *locks;
      locks->resize(n_cells);
#     pragma omp parallel for
      for (int c = 0; c < n_cells; c++)
      {
        auto &current_locks = cell_locks.get_mutexes(c);
        current_locks.resize(cells[c].size());
      }
      cell_locks.initialize();
    }
  };

  template <class GridT> using UpdateMutexesTmpl = UpdateMutexes<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(update_mutexes) { OperatorNodeFactory::instance()->register_factory("update_mutexes", make_grid_variant_operator<UpdateMutexesTmpl>); }
} // namespace exaDEM
