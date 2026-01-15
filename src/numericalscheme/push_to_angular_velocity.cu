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
#include <memory>

#include <exaDEM/traversal.h>
#include <exaDEM/angular_velocity.h>

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT, field::_vrot, field::_arot>> class PushToAngularVelocity : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = field_accessor_tuple_from_field_set_t<FieldSet<field::_vrot, field::_arot>>;
    static constexpr ComputeFields compute_field_set{};

    ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
    ADD_SLOT(double, dt, INPUT, REQUIRED, DocString{"dt is the time increment of the timeloop"});
    ADD_SLOT(Traversal, traversal_real, INPUT, REQUIRED, DocString{"list of non empty cells within the current grid"});

  public:
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator computes particle angular velocitiy values from angular velocities and angular accelerations. 
        )EOF";
    }

    inline void execute() override final
    {
      const double dt = *(this->dt);
      const double dt_2 = 0.5 * dt;
      const ComputeCellParticlesOptions ccpo = traversal_real->get_compute_cell_particles_options();
      PushToAngularVelocityFunctor func{dt_2};
      compute_cell_particles(*grid, false, func, compute_field_set, parallel_execution_context(), ccpo);
    }
  };

  template <class GridT> using PushToAngularVelocityTmpl = PushToAngularVelocity<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(push_to_angular_velocity) { OperatorNodeFactory::instance()->register_factory("push_to_angular_velocity", make_grid_variant_operator<PushToAngularVelocityTmpl>); }

} // namespace exaDEM
