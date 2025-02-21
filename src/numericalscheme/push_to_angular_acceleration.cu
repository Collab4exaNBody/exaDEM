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
#include <exanb/core/quaternion_operators.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/compute/compute_cell_particles.h>
#include <memory>

#include <exaDEM/traversal.hpp>
#include <exaDEM/angular_acceleration.h>

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT, field::_orient, field::_mom, field::_vrot, field::_arot, field::_inertia>> class PushToAngularAcceleration : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet<field::_orient, field::_mom, field::_vrot, field::_arot, field::_inertia>;
    static constexpr ComputeFields compute_field_set{};

    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    ADD_SLOT(Traversal, traversal_real, INPUT, DocString{"list of non empty cells within the current grid"});

  public:
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator computes the new values of angular acceleration from moments, orientations, angular velocities, angular accelerations and inertia.
        )EOF";
    }

    inline void execute() override final
    {
      auto [cell_ptr, cell_size] = traversal_real->info();
      PushToAngularAccelerationFunctor func{};
      compute_cell_particles(*grid, false, func, compute_field_set, parallel_execution_context(), cell_ptr, cell_size);
    }
  };

  template <class GridT> using PushToAngularAccelerationTmpl = PushToAngularAcceleration<GridT>;

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("push_to_angular_acceleration", make_grid_variant_operator<PushToAngularAccelerationTmpl>); }
} // namespace exaDEM
