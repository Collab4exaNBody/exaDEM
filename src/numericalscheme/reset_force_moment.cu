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
#include <exanb/compute/compute_cell_particles.h>
#include <exaDEM/set_fields.h>
#include <exaDEM/traversal.h>

namespace exaDEM
{
  using namespace exanb;
  template <typename GridT, class = AssertGridHasFields<GridT, field::_fx, field::_fy, field::_fz, field::_mom>> class ResetForceMomentNode : public OperatorNode
  {
    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    ADD_SLOT(Traversal, traversal_real, INPUT, DocString{"list of non empty cells within the current grid"});

    static inline constexpr FieldSet<field::_fx, field::_fy, field::_fz, field::_mom> compute_field_set = {};

  public:
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator resets two grid fields : moments and forces.
        )EOF";
    }

    inline void execute() override final
    {
      auto [cell_ptr, cell_size] = traversal_real->info();
      SetFunctor<double, double, double, Vec3d> func = {{double(0.0), double(0.0), double(0.0), Vec3d{0.0, 0.0, 0.0}}};
      compute_cell_particles(*grid, false, func, compute_field_set, parallel_execution_context(), cell_ptr, cell_size);
    }
  };

  template <class GridT> using ResetForceMomentNodeTmpl = ResetForceMomentNode<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(reset_force_moment) { OperatorNodeFactory::instance()->register_factory("reset_force_moment", make_grid_variant_operator<ResetForceMomentNodeTmpl>); }

} // namespace exaDEM
