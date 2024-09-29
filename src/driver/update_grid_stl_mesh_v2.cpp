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
	template<	class GridT, class = AssertGridHasFields< GridT >> class UpdateGridSTLMeshOperatorV2 : public OperatorNode
	{
		using ComputeFields = FieldSet< field::_rx ,field::_ry ,field::_rz>;
		static constexpr ComputeFields compute_field_set {};
		ADD_SLOT( MPI_Comm , mpi      , INPUT        , MPI_COMM_WORLD , DocString{"MPI communicator for parallel processing."});
		ADD_SLOT( GridT    , grid     , INPUT_OUTPUT , DocString{"Grid used for computations."} );
		ADD_SLOT( Drivers  , drivers  , INPUT_OUTPUT , DocString{"List of Drivers"});
		ADD_SLOT( double   , rcut_max , INPUT        , REQUIRED , DocString{"rcut_max"});

		public:
		inline std::string documentation() const override final
		{
			return R"EOF( 
    	    			)EOF";
		}

		inline void execute () override final
		{
			const auto& g = *grid;
			const size_t n_cells = g.number_of_cells(); // nbh.size();
			const IJK dims = g.dimension();
			const double Rmax = *rcut_max;

			for(size_t id = 0 ; id < drivers->get_size() ; id++)
			{
				if ( drivers->type(id) == DRIVER_TYPE::STL_MESH)
				{
					exaDEM::Stl_mesh& mesh = std::get<exaDEM::Stl_mesh> (drivers->data(id)); 
          mesh.shp.pre_compute_obb_vertices(mesh.center, mesh.quat);
          mesh.shp.pre_compute_obb_edges(mesh.center, mesh.quat);
          mesh.shp.pre_compute_obb_faces(mesh.center, mesh.quat);
					auto& grid_stl = mesh.grid_indexes;
					grid_stl.clear();
					grid_stl.resize(n_cells);

					auto& obb_v = mesh.shp.m_obb_vertices;

					for(size_t vid = 0 ; vid < obb_v.size() ; vid++)
					{
						auto obb = obb_v[vid];
						obb.enlarge(Rmax);
						AABB aabb = conv_to_aabb(obb);
						IJK max = g.locate_cell(aabb.bmax);
						IJK min = g.locate_cell(aabb.bmin);
						for(int x = min.i ; x <= max.i ; x++)
							for(int y = min.j ; y <= max.j ; y++)
								for(int z = min.k ; z <= max.k ; z++)
								{
									IJK next = {x,y,z};
									if(g.contains(next))
									{ 
										AABB cell_aabb = g.cell_bounds(next);
                    OBB cell_obb = conv_to_obb(cell_aabb);
							  		if(obb.intersect(cell_obb))
										{
                      size_t cell_next_id = grid_ijk_to_index( dims , next );
											grid_stl[cell_next_id].vertices.push_back( vid );
										}
									}
								}
					}

					/** add edges */
					auto& obb_e = mesh.shp.m_obb_edges;

					for(size_t eid = 0 ; eid < obb_e.size() ; eid++)
					{
						auto obb = obb_e[eid];
						obb.enlarge(Rmax);
						AABB aabb = conv_to_aabb(obb);
						IJK max = g.locate_cell(aabb.bmax);
						IJK min = g.locate_cell(aabb.bmin);
						for(int x = min.i ; x <= max.i ; x++)
							for(int y = min.j ; y <= max.j ; y++)
								for(int z = min.k ; z <= max.k ; z++)
								{
									IJK next = {x,y,z};
									if(g.contains(next))
									{ 
									  AABB cell_aabb = g.cell_bounds(next);
                    OBB cell_obb = conv_to_obb(cell_aabb);
										if(obb.intersect(cell_obb))
										{
                      size_t cell_next_id = grid_ijk_to_index( dims , next );
											grid_stl[cell_next_id].edges.push_back( eid );
										}
									}
								}
					}

					auto& obb_f = mesh.shp.m_obb_faces;
					for(size_t fid = 0 ; fid < obb_f.size() ; fid++)
					{
						auto obb = obb_f[fid];
						obb.enlarge(Rmax);
						AABB aabb = conv_to_aabb(obb);
						IJK max = g.locate_cell(aabb.bmax);
						IJK min = g.locate_cell(aabb.bmin);
						for(int x = min.i ; x <= max.i ; x++)
							for(int y = min.j ; y <= max.j ; y++)
								for(int z = min.k ; z <= max.k ; z++)
								{
									IJK next = {x,y,z};
									if(g.contains(next))
									{ 
										AABB cell_aabb = g.cell_bounds(next);
                    OBB cell_obb = conv_to_obb(cell_aabb);
										if(obb.intersect(cell_obb))
										{
                      size_t cell_next_id = grid_ijk_to_index( dims , next );
											grid_stl[cell_next_id].faces.push_back( fid );
										}
									}
								}
					}
          //mesh.grid_indexes_summary(); //for debug
				}
			}
		}
	};
	// this helps older versions of gcc handle the unnamed default second template parameter
	template <class GridT> using UpdateGridSTLMeshOperatorV2Template = UpdateGridSTLMeshOperatorV2<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "update_grid_stl_mesh_v2", make_grid_variant_operator< UpdateGridSTLMeshOperatorV2Template > );
	}
}

