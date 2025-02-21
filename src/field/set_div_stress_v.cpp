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
#include <exanb/compute/compute_cell_particles.h>

#include <exaDEM/shapes.hpp>
#include <include/exaDEM/div_field_volume.hpp>

namespace exaDEM
{
  using namespace exanb;
  template <typename GridT, class = AssertGridHasFields<GridT, field::_radius, field::_stress>> class DivStressV : public OperatorNode
  {
    using ComputeFieldType = FieldSet<field::_type, field::_stress>;
    using ComputeFieldRadius = FieldSet<field::_radius, field::_stress>;
    static constexpr ComputeFieldType compute_field_set_type{};
    static constexpr ComputeFieldRadius compute_field_set_radius{};

    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    ADD_SLOT(shapes, shapes_collection, INPUT, OPTIONAL, DocString{"Collection of shapes"});

    public:
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator divides the stress tensor field by the volume (spheres or polyhedra).
        )EOF";
    }

    inline void execute() override final
    {
      if( shapes_collection.has_value()) // polyhedra
      {
        poly_div_field_volume<exanb::Mat3d> func = {shapes_collection->data()};
        compute_cell_particles(*grid, false, func, compute_field_set_type, parallel_execution_context());
      }
      else // spheres
      {
        sphere_div_field_volume<exanb::Mat3d> func = {};
        compute_cell_particles(*grid, false, func, compute_field_set_radius, parallel_execution_context());
      }
    }
  };

  template <class GridT> using DivStressVTmpl = DivStressV<GridT>;

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("set_div_stress_v", make_grid_variant_operator<DivStressVTmpl>); }

} // namespace exaDEM
