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
#include <exaDEM/traversal.hpp>
#include <exanb/compute/compute_cell_particles.h>
#include <exanb/grid_cell_particles/particle_region.h>
#include <exaDEM/set_fields.hpp>

namespace exaDEM {
template <typename GridT, class = AssertGridHasFields<GridT, field::_fx, field::_fy, field::_fz, field::_mom>>
class FreezeFieldsNode : public OperatorNode{
# define FREEZE_TMPL double, double, double, Vec3d, double, double, double, Vec3d 
  ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
  ADD_SLOT(Traversal, traversal_real, INPUT, REQUIRED, DocString{"list of non empty cells within the current grid"});
  ADD_SLOT(ParticleRegions, particle_regions, INPUT, OPTIONAL);
  ADD_SLOT(ParticleRegionCSG, region, INPUT, OPTIONAL);

  using ComputeFields = field_accessor_tuple_from_field_set_t<
                        FieldSet<field::_fx, field::_fy, field::_fz, field::_mom,
                                 field::_vx, field::_vy, field::_vz, field::_vrot>>;
  using ComputeRegionFields = field_accessor_tuple_from_field_set_t<
                        FieldSet<field::_rx, field::_ry, field::_rz, field::_id,
                                 field::_fx, field::_fy, field::_fz, field::_mom,
                                 field::_vx, field::_vy, field::_vz, field::_vrot>>;

  static constexpr ComputeFields compute_field_set{};
  static constexpr ComputeRegionFields compute_region_field_set{};

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator resets two grid fields : moments and forces.
        )EOF";
  }

  inline void execute() final {
    const ComputeCellParticlesOptions ccpo = traversal_real->get_compute_cell_particles_options();
    if (region.has_value()) {
      if (!particle_regions.has_value()) {
        fatal_error() << "Region is defined, but particle_regions has no value" << std::endl;
      }
      if (region->m_nb_operands == 0) {
        ldbg << "rebuild CSG from expr " << region->m_user_expr << std::endl;
        region->build_from_expression_string(particle_regions->data(), particle_regions->size());
      }
      ParticleRegionCSGShallowCopy prcsg = *region;
      SetRegionFunctor<FREEZE_TMPL> func = {prcsg, 
        {0.0, 0.0, 0.0, Vec3d{0.0, 0.0, 0.0}, 0.0, 0.0, 0.0, Vec3d{0.0, 0.0, 0.0}}};
      compute_cell_particles(*grid, false, func, compute_region_field_set, parallel_execution_context(), ccpo);
    } else {
      SetFunctor<FREEZE_TMPL> func = {0.0, 0.0, 0.0,
                                      Vec3d{0.0, 0.0, 0.0},
                                      0.0, 0.0, 0.0,
                                      Vec3d{0.0, 0.0, 0.0}};
      compute_cell_particles(*grid, false, func, compute_field_set, parallel_execution_context(), ccpo);
    }
	}
# undef FREEZE_TMPL
};
// === register factories ===
ONIKA_AUTORUN_INIT(freeze_particles) {
  OperatorNodeFactory::instance()->register_factory("freeze_particles",
                                                    make_grid_variant_operator<FreezeFieldsNode>);
  OperatorNodeFactory::instance()->register_factory("pas_libere_pas_delivre",
                                                    make_grid_variant_operator<FreezeFieldsNode>);
}
}  // namespace exaDEM
