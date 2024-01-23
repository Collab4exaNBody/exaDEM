#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>

#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/particle_neighbors/chunk_neighbors_apply.h>

#include <memory>

#include <exaDEM/neighbor_type.h>
#include <exaDEM/interaction.hpp>

namespace exaDEM
{
	using namespace exanb;

	template<typename GridT
		, class = AssertGridHasFields< GridT, field::_radius >
		>
		class InitNeighborT : public OperatorNode
		{
			// attributes processed during computation
			using ComputeFields = FieldSet< field::_vrot, field::_arot >;
			static constexpr ComputeFields compute_field_set {};
			ADD_SLOT( GridT  , grid     , INPUT_OUTPUT , REQUIRED );
			ADD_SLOT( exanb::GridChunkNeighbors , chunk_neighbors     , INPUT , OPTIONAL , DocString{"neighbor list"} );
			ADD_SLOT( GridCellParticleNeigborT<Interaction> , interactions  , INPUT_OUTPUT , GridCellParticleNeigborT<Interaction>{} , DocString{"Neighbor particle interaction term"} );

			public:

			inline std::string documentation() const override final
			{
				return R"EOF(
        				This operator initializes interaction data strutures.
				        )EOF";
			}

			inline void execute () override final
			{
				const auto cells = grid->cells();
				const size_t n_cells = grid->number_of_cells(); // nbh.size();

				const IJK dims = grid->dimension();
				const int gl = grid->ghost_layers();

				auto & interaction = *interactions;
				size_t n_total_nbh=0,n_interaction=0;

				interaction.m_cell_type.clear();
				interaction.m_cell_type.resize( n_cells );

				if( ! chunk_neighbors.has_value() ) return;

				ldbg << "init_neighbor_interaction: "<< n_cells<<" cells, chunk_neighbors.has_value() = "<<chunk_neighbors.has_value() <<std::endl;

#     pragma omp parallel
				{
					GRID_OMP_FOR_BEGIN(dims-2*gl,_,block_loc, schedule(dynamic) reduction(+:n_total_nbh,n_interaction) )
					{
						IJK loc_a = block_loc + gl;
						size_t cell_a = grid_ijk_to_index( dims , loc_a );
						const int n_particles = cells[cell_a].size();

						auto & interaction = interactions->m_cell_type[cell_a];
						interaction.initialize( n_particles );

						int last_p_a = -1;

						size_t local_n_total_nbh=0, local_n_interaction=0;
						apply_cell_particle_neighbors(*grid, *chunk_neighbors, cell_a, loc_a, std::false_type() /* not symetric */,
								[cells,cell_a,&interaction,&last_p_a,&local_n_total_nbh,&local_n_interaction]
								( int p_a, size_t cell_b, unsigned int p_b , size_t p_nbh_index )
								{
								++ local_n_total_nbh;
								if( p_a != last_p_a )
								{
								assert( p_a>=0 && last_p_a<p_a );
								while( last_p_a < p_a ) // ensures we have a begin/end for each particle, even if it hasn't any neighbor
								{
								if( last_p_a != -1 ) interaction.end_nbh_write( last_p_a , cells[cell_a][field::id][last_p_a] );
								++ last_p_a;
								interaction.begin_nbh_write( last_p_a , cells[cell_a][field::id][last_p_a] );
								}
								assert( last_p_a == p_a );
								assert( size_t(p_a) < interaction.number_of_items() );                
								}
								//assert( interaction.m_offset[p_a] + p_nbh_index == interaction.m_neighbor_interaction.size() );
								Vec3d ra = { cells[cell_a][field::rx][p_a] , cells[cell_a][field::ry][p_a] , cells[cell_a][field::rz][p_a] };
								Vec3d rb = { cells[cell_b][field::rx][p_b] , cells[cell_b][field::ry][p_b] , cells[cell_b][field::rz][p_b] };
								const double radius_a = cells[cell_a][field::radius][p_a];
								const double radius_b = cells[cell_b][field::radius][p_b];
								const double rc2 = radius_a + radius_b;
								double d2 = norm2( rb - ra );
								if( d2 < rc2 )
								{
									++ local_n_interaction;
									const exanb::Vec3d vec_null = {0,0,0};
									const int vidx = 0;
									const int type = 0;
									Interaction item {vec_null, vec_null, cell_a, p_a, vidx, cell_b, p_b, vidx, type} ;
									
									//interaction.push_back( { cells[cell_b][field::id][p_b] , {d2,0.,rc2} } );
									interaction.push_back( {cells[cell_b][field::id][p_b], item} );
								}
								else
								{
									// no interaction
								}
								});

						// ensures correct encoding for every particles, even those without any neighbors
						while( last_p_a < n_particles ) // ensures we have a begin/end for each particle, even if it hasn't any neighbor
						{
							if( last_p_a != -1 ) interaction.end_nbh_write( last_p_a , cells[cell_a][field::id][last_p_a] );
							++ last_p_a;
							if( last_p_a < n_particles ) interaction.begin_nbh_write( last_p_a , cells[cell_a][field::id][last_p_a] );
						}

						assert( interaction.check_consistency() );
						n_total_nbh += local_n_total_nbh;
						n_interaction += local_n_interaction;
					}
					GRID_OMP_FOR_END
				}

				std::cout << "init_interactions: "<< n_total_nbh<<" total interactions, "<<n_interaction<<" interaction pairs"<<std::endl;
				std::abort();
			}
		};

	template<class GridT> using InitNeighborTTmpl = InitNeighborT<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "init_interactions", make_grid_variant_operator< InitNeighborTTmpl > );
	}

}

