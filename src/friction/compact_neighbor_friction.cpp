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
		, class = AssertGridHasFields< GridT, field::_radius 
#                                ifndef NDEBUG
		, field::_id
#                                endif
		>
		>
		class CompactNeighborFriction : public OperatorNode
		{
			// attributes processed during computation
			using ComputeFields = FieldSet< field::_vrot, field::_arot >;
			static constexpr ComputeFields compute_field_set {};

			ADD_SLOT( GridT  , grid     , INPUT_OUTPUT , REQUIRED );
			ADD_SLOT( exanb::GridChunkNeighbors , chunk_neighbors   , INPUT , OPTIONAL , DocString{"neighbor list"} );
			ADD_SLOT( GridCellParticleNeigborFriction , nbh_friction , INPUT_OUTPUT , GridCellParticleNeigborFriction{} , DocString{"Neighbor particle friction term"} );

			public:
			inline std::string documentation() const override final
			{
				return R"EOF(
        				This operator copies / compacts friction data structures.
				        )EOF";
			}

			inline void execute () override final
			{
				const size_t n_cells = grid->number_of_cells(); // nbh.size();

				const IJK dims = grid->dimension();
				const int gl = grid->ghost_layers();

				assert( nbh_friction->m_cell_friction.size() == n_cells );
				ldbg << "CompactNeighborFriction: "<< n_cells<<" cells, chunk_neighbors.has_value() = "<<chunk_neighbors.has_value() << std::endl;

				if( ! chunk_neighbors.has_value() ) return;

				size_t pairs_before=0 , pairs_after=0;
				size_t pairs_min=std::numeric_limits<size_t>::max(), pairs_max=0;

#     pragma omp parallel
				{
					GRID_OMP_FOR_BEGIN(dims-2*gl,_,block_loc, schedule(dynamic) reduction(+:pairs_before,pairs_after) reduction(max:pairs_max) reduction(min:pairs_min) )
					{
						IJK loc_a = block_loc + gl;
						size_t cell_a = grid_ijk_to_index( dims , loc_a );
						auto [before,after] = nbh_friction->m_cell_friction[cell_a].compact_friction_pairs();
						pairs_before += before;
						pairs_after += after;
						size_t n_particles = nbh_friction->m_cell_friction[cell_a].number_of_particles();
						for(size_t p_a=0;p_a<n_particles;p_a++)
						{
							size_t npairs = nbh_friction->m_cell_friction[cell_a].particle_number_of_pairs(p_a);
							pairs_min = std::min(pairs_min,npairs);
							pairs_max = std::max(pairs_max,npairs);
						}  
					}
					GRID_OMP_FOR_END
				}

				ldbg << "CompactNeighborFriction: end : pairs_before="<<pairs_before<<" , pairs_after="<<pairs_after << ", min="<<pairs_min<<", max="<<pairs_max<< std::endl;

#if 0
#     ifndef NDEBUG
				const auto cells = grid->cells();
				/*
					 std::map< uint64_t , std::pair<size_t,size_t> > id_map;
					 GRID_FOR_BEGIN(dims,cell_i,cell_loc)
					 {
					 size_t n_particles = cells[cell_i].size();        
					 for(size_t p_i=0;p_i<n_particles;p_i++)
					 {
					 id_map[ cells[cell_i][field::id][p_i] ] = std::pair<size_t,size_t>{ cell_i , p_i };
					 }
					 }
					 GRID_FOR_END
				 */
				// check symetry of friction values
				std::map< std::pair<uint64_t,uint64_t> , Vec3d > friction_map;
				size_t n_matched_sym_pairs = 0;
				GRID_FOR_BEGIN(dims-2*gl,_,block_loc)
				{
					IJK loc_a = block_loc + gl;
					size_t cell_a = grid_ijk_to_index( dims , loc_a );
					const auto& cell_friction = nbh_friction->m_cell_friction[cell_a];
					assert( cell_friction.number_of_particles() == cells[cell_a].size() );
					size_t n_particles = cell_friction.number_of_particles();
					for(size_t p_a=0;p_a<n_particles;p_a++)
					{
						uint64_t id_a = cell_friction.particle_id(p_a);
						assert( cells[cell_a][field::id][p_a] == id_a );
						size_t n_pairs = cell_friction.particle_number_of_pairs(p_a);
						for(size_t i=0;i<n_pairs;i++)
						{
							const auto & pf = cell_friction.pair_friction(p_a,i);
							std::pair<uint64_t,uint64_t> id_pair = { id_a , pf.m_particle_id };
							if( id_pair.first > id_pair.second ) std::swap( id_pair.first , id_pair.second );
							assert( id_pair.first != id_pair.second );
							auto it = friction_map.find( id_pair );
							if( it == friction_map.end() )
							{
								friction_map.insert( { id_pair , pf.m_friction } );
							}
							else
							{
								if( it->second == pf.m_friction )
								{
									++ n_matched_sym_pairs;
								}
								else
								{
#               pragma omp critical(dbg_mesg)
									lerr << "Symetric friction value mismatch: cell_a="<<cell_a<<", ghost="<<grid->is_ghost_cell(cell_a)<<", p_a="<<p_a<<", id_a="<<id_a<<", pair=("<<id_pair.first<<","<<id_pair.second<<")" <<std::endl;
									std::abort();
								}
							}
						}
					}
				}
				GRID_FOR_END
					ldbg << "CompactNeighborFriction: symetry Ok, matched symetrical pairs = "<<n_matched_sym_pairs <<std::endl;
#     endif
#endif
			}
		};

	template<class GridT> using CompactNeighborFrictionTmpl = CompactNeighborFriction<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "compact_neighbor_friction", make_grid_variant_operator< CompactNeighborFrictionTmpl > );
	}

}

