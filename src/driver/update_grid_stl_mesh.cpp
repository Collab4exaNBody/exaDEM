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
#include <exanb/core/domain.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_operators.h>
#include <exanb/core/basic_types_stream.h>
#include <onika/memory/allocator.h> // for ONIKA_ASSUME_ALIGNED macro
#include <exanb/compute/compute_pair_optional_args.h>
#include <vector>
#include <iomanip>

#include <exanb/compute/compute_cell_particles.h>
#include <exaDEM/face.h>
#include <exaDEM/stl_mesh.h>
#include <exaDEM/drivers.h>

#include <mpi.h>

namespace exaDEM
{
	using namespace exanb;
	template<	class GridT, class = AssertGridHasFields< GridT >> class UpdateGridSTLMeshOperator : public OperatorNode
	{
		using ComputeFields = FieldSet< field::_rx ,field::_ry ,field::_rz>;
		static constexpr ComputeFields compute_field_set {};
		ADD_SLOT( MPI_Comm , mpi      , INPUT , MPI_COMM_WORLD , DocString{"MPI communicator for parallel processing."});
		ADD_SLOT( GridT    , grid     , INPUT_OUTPUT , DocString{"Grid used for computations."} );
		ADD_SLOT( Drivers     , drivers         , INPUT_OUTPUT,            DocString{"List of Drivers"});
		ADD_SLOT( double     , rcut_max         , INPUT, REQUIRED,           DocString{"rcut_max"});

		public:
		inline std::string documentation() const override final
		{
			return R"EOF( 
    	    			)EOF";
		}


		// could be speedup by an OBB TREE
		void update_indexes ( list_of_elements& list, OBB& bx, shape& shp)
		{
			// first vertices
			auto& obb_v = shp.m_obb_vertices;
			auto& obb_e = shp.m_obb_edges;
			auto& obb_f = shp.m_obb_faces;
			list.vertices.clear();
			list.faces.clear();
			list.edges.clear();
			for(size_t it = 0; it < obb_v.size() ; it++)
			{
				if ( bx.intersect( obb_v[it] ) ) list.vertices.push_back( it );
				//list.vertices.push_back( it );
			}

			for(size_t it = 0; it < obb_e.size() ; it++)
			{
				if ( bx.intersect( obb_e[it] ) ) list.edges.push_back( it );
				//list.edges.push_back( it );
			}

			for(size_t it = 0; it < obb_f.size() ; it++)
			{
				if ( bx.intersect( obb_f[it] ) ) list.faces.push_back( it );
				//list.faces.push_back( it );
			}
		}

		inline void execute () override final
		{
			const size_t n_cells = grid->number_of_cells(); // nbh.size();
			const IJK dims = grid->dimension();
			const int gl = grid->ghost_layers();
			const double csize = grid->cell_size();
			const double Rmax = *rcut_max;

			for(size_t id = 0 ; id < drivers->get_size() ; id++)
			{
				if ( drivers->type(id) == DRIVER_TYPE::STL_MESH)
				{
					exaDEM::Stl_mesh& mesh = std::get<exaDEM::Stl_mesh> (drivers->data(id)); 
					auto& grid_stl = mesh.grid_indexes;
					grid_stl.clear();
					grid_stl.resize(n_cells);

#     pragma omp parallel
					{
						OBB bx;
            bx.extent = {0.5 * csize + Rmax, 0.5 * csize + Rmax, 0.5 * csize + Rmax};
						GRID_OMP_FOR_BEGIN(dims-2*gl,_,block_loc, schedule(dynamic))
						{
							IJK loc_a = block_loc + gl;
							size_t cell_a = grid_ijk_to_index( dims , loc_a );
							auto cb = grid->cell_bounds(loc_a);
							auto center = (cb.bmin + cb.bmax) / 2;
							bx.center = { center.x , center.y, center.z};
							update_indexes ( grid_stl[cell_a], bx, mesh.shp);							
						}
						GRID_OMP_FOR_END
					}
					//mesh.grid_indexes_summary(); for debug
				}
			}
		};
	};

	// this helps older versions of gcc handle the unnamed default second template parameter
	template <class GridT> using UpdateGridSTLMeshOperatorTemplate = UpdateGridSTLMeshOperator<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "update_grid_stl_mesh", make_grid_variant_operator< UpdateGridSTLMeshOperatorTemplate > );
	}
}
