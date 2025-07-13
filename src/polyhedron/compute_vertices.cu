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
#include <exanb/compute/compute_cell_particles.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>

#include <memory>
#include <random>
#include <exaDEM/vertices.hpp>
#include <exaDEM/shapes.hpp>
#include <exaDEM/compute_vertices.hpp>
#include <exaDEM/traversal.h>
#include <iostream>

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT, field::_type, field::_homothety, field::_orient>> class PolyhedraComputeVertices : public OperatorNode
  {
    using ComputeFields = FieldSet<field::_type, field::_rx, field::_ry, field::_rz, field::_homothety, field::_orient>;
    static constexpr ComputeFields compute_field_set{};
    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    ADD_SLOT(CellVertexField, cvf, INPUT_OUTPUT, DocString{"Store vertex positions for every polyhedron"});
    ADD_SLOT(Domain , domain, INPUT , REQUIRED );
    ADD_SLOT(shapes, shapes_collection, INPUT_OUTPUT, DocString{"Collection of shapes"});
    ADD_SLOT(Traversal, traversal_all, INPUT, DocString{"list of non empty cells [ALL] within the current grid"});
    ADD_SLOT(bool, resize_vertex, INPUT, true, DocString{"enable to resize the data storage used for vertices"});
    ADD_SLOT(bool, minimize_memory_footprint, INPUT, false, DocString{"enable to resize the data storage using only the maximum of vertices according to the particle shapes into a cell. This option is useful if there are some particles with a very high number of particles."});

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
      bool is_def_box = !domain->xform_is_identity();
      auto& vertex_fields = *cvf;
      const auto cells = grid->cells();
      const size_t n_cells = grid->number_of_cells(); // nbh.size();

      size_t* cell_ptr = nullptr;
      size_t cell_size = 0;

      if(traversal_all->iterator)
      {
        std::tie(cell_ptr, cell_size) = traversal_all->info();
      }

      if(*resize_vertex || *minimize_memory_footprint)
      {
        vertex_fields.resize(n_cells);
        if( !(*minimize_memory_footprint) )
        {
          int max_number_of_vertices = shps->get_number_of_vertices();
#pragma omp parallel for schedule(guided)
          for(size_t cell_id = 0 ; cell_id < n_cells ; cell_id++)
          {
            size_t np = cells[cell_id].size();       
            vertex_fields.resize(cell_id, np, max_number_of_vertices); 
          }
        }
        else
        {
#pragma omp parallel for schedule(guided)
          for(size_t cell_id = 0 ; cell_id < n_cells ; cell_id++)
          {
            const auto *__restrict__ type = cells[cell_id][field::type];
            size_t np = cells[cell_id].size();  
            int max_number_of_vertices = 0;
            for(size_t p = 0 ; p < np ; p++)
            {
              max_number_of_vertices = std::max(max_number_of_vertices, shps[type[p]].get_number_of_vertices());
            }
            vertex_fields.resize(cell_id, np, max_number_of_vertices); 
          }
        }
      }

      if( is_def_box )
      {
        PolyhedraComputeVerticesFunctor<true> func{shps, vertex_fields.data(), domain->xform()};
        compute_cell_particles(*grid, true, func, compute_field_set, parallel_execution_context(), cell_ptr, cell_size);
      }
      else
      {
        PolyhedraComputeVerticesFunctor<false> func{shps, vertex_fields.data(), domain->xform() };
        compute_cell_particles(*grid, true, func, compute_field_set, parallel_execution_context(), cell_ptr, cell_size);
      }
    }
  };

  template <class GridT> using PolyhedraComputeVerticesTmpl = PolyhedraComputeVertices<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(compute_vertices) { OperatorNodeFactory::instance()->register_factory("compute_vertices", make_grid_variant_operator<PolyhedraComputeVerticesTmpl>); }
} // namespace exaDEM
