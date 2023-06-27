#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>

#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/particle_neighbors/chunk_neighbors_apply.h>

#include <memory>

#include <exaDEM/neighbor_friction.h>

namespace exaDEM
{
	using namespace exanb;

	template<typename GridT
		, class = AssertGridHasFields< GridT, field::_radius >
		>
		class InitNeighborFriction : public OperatorNode
		{
			// attributes processed during computation
			using ComputeFields = FieldSet< field::_vrot, field::_arot >;
			static constexpr ComputeFields compute_field_set {};

			ADD_SLOT( GridT  , grid     , INPUT_OUTPUT , REQUIRED );
			ADD_SLOT( exanb::GridChunkNeighbors , chunk_neighbors     , INPUT , OPTIONAL , DocString{"neighbor list"} );
			ADD_SLOT( GridCellParticleNeigborFriction , nbh_friction  , INPUT_OUTPUT , GridCellParticleNeigborFriction{} , DocString{"Neighbor particle friction term"} );

			public:

			inline std::string documentation() const override final
			{
				return R"EOF(
        				This operator initializes friction data strutures.
				        )EOF";
			}

			inline void execute () override final
			{
				const auto cells = grid->cells();
				const size_t n_cells = grid->number_of_cells(); // nbh.size();

				const IJK dims = grid->dimension();
				const int gl = grid->ghost_layers();

				auto & friction = *nbh_friction;
				size_t n_total_nbh=0,n_friction=0;

				friction.m_cell_friction.clear();
				friction.m_cell_friction.resize( n_cells );

				if( ! chunk_neighbors.has_value() ) return;

				ldbg << "init_neighbor_friction: "<< n_cells<<" cells, chunk_neighbors.has_value() = "<<chunk_neighbors.has_value() <<std::endl;

#     pragma omp parallel
				{
					GRID_OMP_FOR_BEGIN(dims-2*gl,_,block_loc, schedule(dynamic) reduction(+:n_total_nbh,n_friction) )
					{
						IJK loc_a = block_loc + gl;
						size_t cell_a = grid_ijk_to_index( dims , loc_a );
						const int n_particles = cells[cell_a].size();

						auto & friction = nbh_friction->m_cell_friction[cell_a];
						friction.initialize( n_particles );
						//friction.m_neighbor_friction.clear();

						int last_p_a = -1;

						size_t local_n_total_nbh=0, local_n_friction=0;
						apply_cell_particle_neighbors(*grid, *chunk_neighbors, cell_a, loc_a, std::false_type() /* not symetric */,
								[cells,cell_a,&friction,&last_p_a,&local_n_total_nbh,&local_n_friction]
								( int p_a, size_t cell_b, unsigned int p_b , size_t p_nbh_index )
								{
								++ local_n_total_nbh;
								if( p_a != last_p_a )
								{
								assert( p_a>=0 && last_p_a<p_a );
								while( last_p_a < p_a ) // ensures we have a begin/end for each particle, even if it hasn't any neighbor
								{
								if( last_p_a != -1 ) friction.end_nbh_write( last_p_a , cells[cell_a][field::id][last_p_a] );
								++ last_p_a;
								friction.begin_nbh_write( last_p_a , cells[cell_a][field::id][last_p_a] );
								}
								assert( last_p_a == p_a );
								assert( size_t(p_a) < friction.number_of_particles() );                
								}
								//assert( friction.m_offset[p_a] + p_nbh_index == friction.m_neighbor_friction.size() );
								Vec3d ra = { cells[cell_a][field::rx][p_a] , cells[cell_a][field::ry][p_a] , cells[cell_a][field::rz][p_a] };
								Vec3d rb = { cells[cell_b][field::rx][p_b] , cells[cell_b][field::ry][p_b] , cells[cell_b][field::rz][p_b] };
								const double radius_a = cells[cell_a][field::radius][p_a];
								const double radius_b = cells[cell_b][field::radius][p_b];
								const double rc2 = radius_a + radius_b;
								double d2 = norm2( rb - ra );
								if( d2 < rc2 )
								{
									++ local_n_friction;
									friction.push_back( { cells[cell_b][field::id][p_b] , {d2,0.,rc2} } );
								}
								else
								{
									friction.push_back( { cells[cell_b][field::id][p_b] , {0.,0.,0.} } );
								}
								});

						// ensures correct encoding for every particles, even those without any neighbors
						while( last_p_a < n_particles ) // ensures we have a begin/end for each particle, even if it hasn't any neighbor
						{
							if( last_p_a != -1 ) friction.end_nbh_write( last_p_a , cells[cell_a][field::id][last_p_a] );
							++ last_p_a;
							if( last_p_a < n_particles ) friction.begin_nbh_write( last_p_a , cells[cell_a][field::id][last_p_a] );
						}

						assert( friction.check_consistency() );
						n_total_nbh += local_n_total_nbh;
						n_friction += local_n_friction;
					}
					GRID_OMP_FOR_END
				}

				ldbg << "init_neighbor_friction: "<< n_total_nbh<<" total neighbors, "<<n_friction<<" friction pairs"<<std::endl;
			}
		};

	template<class GridT> using InitNeighborFrictionTmpl = InitNeighborFriction<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "init_neighbor_friction", make_grid_variant_operator< InitNeighborFrictionTmpl > );
	}

}

