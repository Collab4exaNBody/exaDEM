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
#include <onika/math/quaternion_operators.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/compute/compute_cell_particles.h>
#include <memory>

#include <exaDEM/push_to_quaternion.h>
#include <exaDEM/traversal.hpp>

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT, field::_orient, field::_vrot, field::_arot>> class PushToQuaternion : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet<field::_orient, field::_vrot, field::_arot>;
    static constexpr ComputeFields compute_field_set{};

    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    ADD_SLOT(double, dt, INPUT, DocString{"dt is the time increment of the timeloop"});
    ADD_SLOT(Traversal, traversal_real, INPUT, DocString{"list of non empty cells within the current grid"});

  public:
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator computes particleorientation from angular velocities and angular accelerations.
        )EOF";
    }

    inline void execute() override final
    {
      const double dt = *(this->dt);
      const double dt_2 = 0.5 * dt;
      const double dt2_2 = dt_2 * dt;
      PushToQuaternionFunctor func{dt, dt_2, dt2_2};
      auto [cell_ptr, cell_size] = traversal_real->info();
      compute_cell_particles(*grid, false, func, compute_field_set, parallel_execution_context(), cell_ptr, cell_size);
    }
  };

  template <class GridT> using PushToQuaternionTmpl = PushToQuaternion<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(push_to_quaternion) { OperatorNodeFactory::instance()->register_factory("push_to_quaternion", make_grid_variant_operator<PushToQuaternionTmpl>); }
} // namespace exaDEM
