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
#include <exaDEM/set_fields.hpp>

namespace exaDEM {
template <typename GridT, class = AssertGridHasFields<GridT, field::_type>>
class ResetStressTensor : public OperatorNode {
  using ComputeFields = FieldSet<field::_stress>;
  using ComputeRegionFields = FieldSet<field::_rx, field::_ry, field::_rz, field::_id, field::_stress>;
  static constexpr ComputeFields compute_field_set{};
  static constexpr ComputeRegionFields compute_region_field_set{};
  ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
  ADD_SLOT(ParticleRegions, particle_regions, INPUT, OPTIONAL);
  ADD_SLOT(ParticleRegionCSG, region, INPUT, OPTIONAL);

  // -----------------------------------------------
  // ----------- Operator documentation ------------
  inline std::string documentation() const final {
    return R"EOF(
        This operator resets the stress tensor of all particles. 
 
        YAML example [no option]:

          - reset_stress
        )EOF";
  }

 public:
  inline void execute() final {
    if (region.has_value()) {
      if (!particle_regions.has_value()) {
        fatal_error() << "Region is defined, but particle_regions has no value" << std::endl;
      }

      if (region->m_nb_operands == 0) {
        ldbg << "rebuild CSG from expr " << region->m_user_expr << std::endl;
        region->build_from_expression_string(particle_regions->data(), particle_regions->size());
      }

      ParticleRegionCSGShallowCopy prcsg = *region;
      SetRegionFunctor<exanb::Mat3d> func = {prcsg, exanb::make_zero_matrix()};
      compute_cell_particles(*grid, false, func, compute_region_field_set, parallel_execution_context());
    } else {
      SetFunctor<exanb::Mat3d> func = {exanb::make_zero_matrix()};
      compute_cell_particles(*grid, false, func, compute_field_set, parallel_execution_context());
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(reset_stress) {
  OperatorNodeFactory::instance()->register_factory("reset_stress", make_grid_variant_operator<ResetStressTensor>);
}

}  // namespace exaDEM
