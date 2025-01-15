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
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/compute/compute_cell_particles.h>
#include <exanb/core/grid.h>
#include <memory>
#include <random>
#include <exaDEM/shapes.hpp>
#include <exaDEM/compute_vertices.hpp>
#include <exaDEM/traversal.hpp>

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT, field::_type, field::_homothety, field::_orient, field::_vertices>> class PolyhedraComputeVertices : public OperatorNode
  {
    using ComputeFields = FieldSet<field::_type, field::_rx, field::_ry, field::_rz, field::_homothety, field::_orient, field::_vertices>;
    static constexpr ComputeFields compute_field_set{};
    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    ADD_SLOT(shapes, shapes_collection, INPUT_OUTPUT, DocString{"Collection of shapes"});
    ADD_SLOT(Traversal, traversal_all, INPUT_OUTPUT, DocString{"list of non empty cells [REAL] within the current grid"});

    // -----------------------------------------------
    // ----------- Operator documentation ------------
    inline std::string documentation() const override final
    {
      return R"EOF(
        )EOF";
    }

  public:
    inline void execute() override final
    {
      const shape *shps = shapes_collection->data();
      PolyhedraComputeVerticesFunctor func{shps};
      
      size_t* cell_ptr = nullptr;
      size_t cell_size = size_t(-1);
      
      if(traversal_all->iterator)
      {
      	std::tie(cell_ptr, cell_size) = traversal_all->info();
      }
      
      compute_cell_particles(*grid, true, func, compute_field_set, parallel_execution_context(), cell_ptr, cell_size);
    }
  };

  template <class GridT> using PolyhedraComputeVerticesTmpl = PolyhedraComputeVertices<GridT>;

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("compute_vertices", make_grid_variant_operator<PolyhedraComputeVerticesTmpl>); }

} // namespace exaDEM
