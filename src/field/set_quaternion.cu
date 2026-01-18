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

#include <random>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/compute/compute_cell_particles.h>
#include <exanb/grid_cell_particles/particle_region.h>
#include <exaDEM/set_fields.hpp>
#include <exaDEM/random_quaternion.hpp>

namespace exaDEM {

template <typename GridT, class = AssertGridHasFields<GridT, field::_orient>>
class SetQuaternion : public OperatorNode {
  static constexpr Quaternion default_quaternion = {0.0, 0.0, 0.0, 1.0};
  using ComputeFields = FieldSet<field::_orient>;
  using ComputeRegionFields = FieldSet<field::_rx, field::_ry, field::_rz, field::_id, field::_orient>;
  static constexpr ComputeFields compute_field_set{};
  static constexpr ComputeRegionFields compute_region_field_set{};

  ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
  ADD_SLOT(bool, random, INPUT, false, DocString{"This option generates random orientations for each particle"});
  ADD_SLOT(Quaternion, quat, INPUT, default_quaternion, DocString{"Quaternion value for all particles"});
  ADD_SLOT(ParticleRegions, particle_regions, INPUT, OPTIONAL);
  ADD_SLOT(ParticleRegionCSG, region, INPUT, OPTIONAL);

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator sets the orientation value for every particles. Random option is available.

        YAML example:

          - set_quaternion:
             quat: [1,0,0,0]
             random: true

        )EOF";
  }

  inline void execute() final {
    bool is_random = *random;
    set_gpu_enabled(false);

    if (is_random) {
      if (region.has_value()) {
        if (!particle_regions.has_value()) {
          fatal_error() << "Region is defined, but particle_regions has no value" << std::endl;
        }

        if (region->m_nb_operands == 0) {
          ldbg << "rebuild CSG from expr " << region->m_user_expr << std::endl;
          region->build_from_expression_string(particle_regions->data(), particle_regions->size());
        }
        ParticleRegionCSGShallowCopy prcsg = *region;
        RandomQuaternionFunctor func = {prcsg};

        compute_cell_particles(*grid, false, func, compute_region_field_set, parallel_execution_context());
      } else {
        RandomQuaternionFunctor func = {};
        compute_cell_particles(*grid, false, func, compute_field_set, parallel_execution_context());
      }
    } else {
      if (region.has_value()) {
        if (!particle_regions.has_value()) {
          fatal_error() << "Region is defined, but particle_regions has no value" << std::endl;
        }

        if (region->m_nb_operands == 0) {
          ldbg << "rebuild CSG from expr " << region->m_user_expr << std::endl;
          region->build_from_expression_string(particle_regions->data(), particle_regions->size());
        }

        ParticleRegionCSGShallowCopy prcsg = *region;
        SetRegionFunctor<Quaternion> func = {prcsg, *quat};
        compute_cell_particles(*grid, false, func, compute_region_field_set, parallel_execution_context());
      } else {
        SetFunctor<Quaternion> func = {*quat};
        compute_cell_particles(*grid, false, func, compute_field_set, parallel_execution_context());
      }
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(set_quaternion) {
  OperatorNodeFactory::instance()->register_factory("set_quaternion", make_grid_variant_operator<SetQuaternion>);
}

}  // namespace exaDEM
