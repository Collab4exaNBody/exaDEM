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
#include <exaDEM/set_fields.h>

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT, field::_type>> class SetParticleType : public OperatorNode
  {
    using ComputeFields = FieldSet<field::_type>;
    using ComputeRegionFields = FieldSet<field::_rx, field::_ry, field::_rz, field::_id, field::_type>;
    static constexpr ComputeFields compute_field_set{};
    static constexpr ComputeRegionFields compute_region_field_set{};
    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    ADD_SLOT(uint32_t, type, INPUT, REQUIRED, DocString{"type value applied to all particles"});
    ADD_SLOT(ParticleRegions, particle_regions, INPUT, OPTIONAL);
    ADD_SLOT(ParticleRegionCSG, region, INPUT, OPTIONAL);

    // -----------------------------------------------
    // ----------- Operator documentation ------------
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator fills type id to all particles. 
        )EOF";
    }

  public:
    inline void execute() override final
    {
      uint32_t t = *type;
      if (region.has_value())
      {
        if (!particle_regions.has_value())
        {
          fatal_error() << "Region is defined, but particle_regions has no value" << std::endl;
        }

        if (region->m_nb_operands == 0)
        {
          ldbg << "rebuild CSG from expr " << region->m_user_expr << std::endl;
          region->build_from_expression_string(particle_regions->data(), particle_regions->size());
        }

        ParticleRegionCSGShallowCopy prcsg = *region;
        SetRegionFunctor<uint32_t> func = {prcsg, t};
        compute_cell_particles(*grid, false, func, compute_region_field_set, parallel_execution_context());
      }
      else
      {
        SetFunctor<uint32_t> func = {t};
        compute_cell_particles(*grid, false, func, compute_field_set, parallel_execution_context());
      }
    }
  };

  template <class GridT> using SetParticleTypeTmpl = SetParticleType<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(set_type) { OperatorNodeFactory::instance()->register_factory("set_type", make_grid_variant_operator<SetParticleTypeTmpl>); }

} // namespace exaDEM
