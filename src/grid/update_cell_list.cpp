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
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exaDEM/cell_list_wrapper.hpp>
#include <memory>

namespace exaDEM
{
	using namespace exanb;

	template<typename GridT
		, class = AssertGridHasFields< GridT >
		>
		class UpdateCellList : public OperatorNode
		{
			using ComputeFields = FieldSet<>;
			static constexpr ComputeFields compute_field_set {};
      template <typename T> using VectorT =  onika::memory::CudaMMVector<T>;

			ADD_SLOT( GridT           , grid      , INPUT        , REQUIRED );
			ADD_SLOT( CellListWrapper , cell_list , INPUT_OUTPUT , DocString{"list of non empty cells within the current grid"});


			public:

			inline std::string documentation() const override final
			{
				return R"EOF( This operator update the list of non-empty cells. This operator should be called as long as a particle move from a cell to another cell.
				        )EOF";
			}

			inline void execute () override final
			{
				const auto& cells    = grid->cells();
        IJK dims = grid->dimension();
        const ssize_t gl = grid->ghost_layers();
        auto& cl = cell_list->m_data;

        // reset the cell list
        cl.clear();

        // iterate over "real" cells
        GRID_OMP_FOR_BEGIN( dims-2*gl, _, loc_no_gl )
        {
          const IJK loc = loc_no_gl + gl;
          const size_t i = grid_ijk_to_index( dims , loc );
	        const size_t n_particles = cells[i].size();
          if( n_particles > 0 ) cl.push_back(i);
        }
        GRID_OMP_FOR_END
			}
		};

	template<class GridT> using UpdateCellListTmpl = UpdateCellList<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "update_cell_list", make_grid_variant_operator< UpdateCellListTmpl > );
	}
}

