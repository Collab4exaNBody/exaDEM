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
//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <memory>
#include <exaDEM/traversal.hpp>
#include <exaDEM/gravity_force.h>

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT, field::_mass, field::_fx, field::_fy, field::_fz>> class GravityForce : public OperatorNode
  {
    static constexpr Vec3d default_gravity = {0.0, 0.0, -9.807};
    // attributes processed during computation
    using ComputeFields = FieldSet<field::_mass, field::_fx, field::_fy, field::_fz>;
    static constexpr ComputeFields compute_field_set{};

    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    ADD_SLOT(Traversal, traversal_real, INPUT_OUTPUT, DocString{"list of non empty cells within the current grid"});
    ADD_SLOT(Vec3d, gravity, INPUT, default_gravity, DocString{"define the gravity constant in function of the gravity axis, default value are x axis = 0, y axis = 0 and z axis = -9.807"});

  public:
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator computes forces related to the gravity.
        )EOF";
    }

    inline void execute() override final
    {
      auto [cell_ptr, cell_size] = traversal_real->info();
      GravityForceFunctor func{*gravity};
      compute_cell_particles(*grid, false, func, compute_field_set, parallel_execution_context(), cell_ptr, cell_size);
    }
  };

  template <class GridT> using GravityForceTmpl = GravityForce<GridT>;

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("gravity_force", make_grid_variant_operator<GravityForceTmpl>); }

} // namespace exaDEM
