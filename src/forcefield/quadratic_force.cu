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
#include <exaDEM/traversal.hpp>
#include <exaDEM/forcefield/quadratic_force.hpp>

namespace exaDEM {

template <typename GridT,
          class = AssertGridHasFields<GridT, field::_fx, field::_fy, field::_fz, field::_vx, field::_vy, field::_vz>>
class QuadraticForce : public OperatorNode {
  // attributes processed during computation
  using ComputeFields = field_accessor_tuple_from_field_set_t<
      FieldSet<field::_fx, field::_fy, field::_fz, field::_vx, field::_vy, field::_vz>>;
  static constexpr ComputeFields compute_field_set{};

  ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
  ADD_SLOT(double, cx, INPUT, REQUIRED, DocString{"aerodynamic coefficient."});
  ADD_SLOT(double, mu, INPUT, REQUIRED, DocString{"drag coefficient. air = 0.000015"});
  ADD_SLOT(Traversal, traversal_real, INPUT, REQUIRED, DocString{"list of non empty cells within the current grid"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator computes friction forces related to air or fluide.

        YAML example:
 
          - quadratic_force:
             cx: 0.38
             mu: 0.0000015
        )EOF";
  }

  inline void execute() final {
    const ComputeCellParticlesOptions ccpo = traversal_real->get_compute_cell_particles_options();
    QuadraticForceFunctor func{(*cx) * (*mu)};
    compute_cell_particles(*grid, false, func, compute_field_set, parallel_execution_context(), ccpo);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(quadratic_force) {
  OperatorNodeFactory::instance()->register_factory("quadratic_force", make_grid_variant_operator<QuadraticForce>);
}
}  // namespace exaDEM
