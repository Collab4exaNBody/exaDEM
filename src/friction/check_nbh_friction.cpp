#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_operators.h>
#include <exanb/core/operator.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/log.h>
#include <exanb/core/cpp_utils.h>

#include <yaml-cpp/yaml.h>
#include <exanb/core/quantity_yaml.h>

#include <exanb/core/config.h> // for MAX_PARTICLE_NEIGHBORS
#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/compute/compute_pair_singlemat.h>

#include <onika/memory/allocator.h> // for DEFAULT_ALIGNMENT

#include <exaDEM/neighbor_friction.h>


namespace exaDEM
{
	using namespace exanb;

	// Functor to check correctness of pair friction data
	template<class GridT>
		struct alignas(onika::memory::DEFAULT_ALIGNMENT) CheckNbhFrictionOp 
		{ 
			GridT & grid;
			GridCellParticleNeigborFriction& nbh_friction;
			bool check_ghost = false;

			template<class CellParticlesT, class FrictionT>
				inline void operator ()
				(      
				 const Vec3d& dr,
				 double d2,
				 const uint64_t id_a,
				 CellParticlesT* cells,
				 size_t cell_b,
				 size_t p_b,
				 FrictionT& friction
				) const
				{

					auto id_b = friction.m_particle_id;
					if( friction.m_particle_id != cells[cell_b][field::id][p_b] )
					{
#         pragma omp critical(dbg_mesg)
						std::cerr<<"Bad neighbor Neighbor id "<<friction.m_particle_id<<", should be "<<cells[cell_b][field::id][p_b]<<", central id "<<id_a<<", nbh C#"<<cell_b<<" P#"<<p_b <<std::endl;
						std::abort();
					}

					if( cell_b >= nbh_friction.m_cell_friction.size() )
					{
#         pragma omp critical(dbg_mesg)
						std::cerr<<"Neighbor cell #"<<cell_b<<", over maximum "<<nbh_friction.m_cell_friction.size() <<std::endl;
						std::abort();
					}

					bool cell_b_is_ghost = grid.is_ghost_cell(cell_b);
					const char * cell_b_ghost_str = ( cell_b_is_ghost ? " (ghost)" : "" );

					if( !cell_b_is_ghost || check_ghost )
					{
						const auto & cell_b_friction = nbh_friction.m_cell_friction[cell_b];
						if( p_b >= cell_b_friction.number_of_particles() )
						{
#           pragma omp critical(dbg_mesg)
							std::cerr<<"Neighbor particle #"<<p_b<<", in cell #"<<cell_b<< cell_b_ghost_str <<", over maximum "<<cell_b_friction.number_of_particles() <<std::endl;
							std::abort();
						}

						if( cell_b_friction.particle_id(p_b) != id_b )
						{
#           pragma omp critical(dbg_mesg)
							std::cerr<<"Symetric particle id "<<cell_b_friction.particle_id(p_b)<<", should be "<<id_b <<std::endl;
							std::abort();          
						}

						size_t n_sym_pairs = cell_b_friction.particle_number_of_pairs( p_b );
						int p_a_sym_idx = -1;
						for(size_t i=0;i<n_sym_pairs;i++)
						{
							if( cell_b_friction.pair_friction(p_b,i).m_particle_id == id_a )
							{
								if( p_a_sym_idx != -1 )
								{
#               pragma omp critical(dbg_mesg)
									std::cerr<<"id_a found in more than one friction_pair in symetric data" <<std::endl;
									std::abort();          
								}
								p_a_sym_idx = i;
							}
						}
						if( p_a_sym_idx == -1 )
						{
#           pragma omp critical(dbg_mesg)
							std::cerr<<"id_a ("<<id_a<<") not found in symetric data of particle C#"<<cell_b<<"P"<<p_b<<" , id_b="<<id_b<< cell_b_ghost_str <<std::endl;
							std::abort();          
						}
						const auto & fb = cell_b_friction.pair_friction(p_b,p_a_sym_idx);
						if( fb.m_friction != friction.m_friction )
						{
							const double friction_max_norm = std::max( norm(friction.m_friction) , norm(fb.m_friction) );
							double friction_l2 = norm2( friction.m_friction - fb.m_friction );
							if( friction_max_norm > 0.0 ) friction_l2 /= friction_max_norm;
							if( friction_l2 > 1.e-16 )
							{
#             pragma omp critical(dbg_mesg)
								std::cerr<<"Symetrical friction mismatch: L2="<<friction_l2 <<"id_a="<<id_a<<", id_b="<<id_b<<", d2="<<d2<<", cell_b="<<cell_b<<cell_b_ghost_str<<", p_b="<<p_b
									<<", fa={"<<friction.m_particle_id<<",("<<friction.m_friction.x<<","<<friction.m_friction.y<<","<<friction.m_friction.z<<")} , fb={"
									<<fb.m_particle_id<<",("<<fb.m_friction.x<<","<<fb.m_friction.y<<","<<fb.m_friction.z<<")}"<<std::endl;
								std::abort();          
							}
						}
					}
				}
		};

}

namespace exanb
{
	// specialize functor traits to allow Cuda execution space
	template<class GridT>
		struct ComputePairTraits< exaDEM::CheckNbhFrictionOp<GridT> >
		{
			static inline constexpr bool RequiresBlockSynchronousCall = false;
			static inline constexpr bool ComputeBufferCompatible = false;
			static inline constexpr bool BufferLessCompatible = true;
			static inline constexpr bool CudaCompatible = false;
		};
}

namespace exaDEM
{

	template<
		class GridT,
					class = AssertGridHasFields< GridT, field::_id >
						>
						class CheckNbhFriction : public OperatorNode
						{
							// ========= I/O slots =======================
							ADD_SLOT( GridT                 , grid              , INPUT_OUTPUT );
							ADD_SLOT( Domain                , domain            , INPUT , REQUIRED );
							ADD_SLOT( double                , rcut              , INPUT , REQUIRED );
							ADD_SLOT( exanb::GridChunkNeighbors    , chunk_neighbors   , INPUT , exanb::GridChunkNeighbors{} , DocString{"neighbor list"} );
							ADD_SLOT( GridCellParticleNeigborFriction , nbh_friction , INPUT_OUTPUT , OPTIONAL, DocString{"Neighbor particle friction term"} );

							// cell particles array type
							using CellParticles = typename GridT::CellParticles;

							// attributes processed during computation
							using ComputeFields = FieldSet< field::_id>;
							static constexpr ComputeFields compute_fields {};

							public:

							inline std::string documentation() const override final
							{
								return R"EOF(
        				This operator checks if the numbers of neighbors for friction data structure are corrects.
				        )EOF";
							}

							// Operator execution
							inline void execute () override final
							{
								assert( chunk_neighbors->number_of_cells() == grid->number_of_cells() );

								if( grid->number_of_cells() == 0 ) { return; }

								const IJK dims = grid->dimension();
								const int gl = grid->ghost_layers();

								bool has_friction = false;
								if( nbh_friction.has_value() ) has_friction = ! nbh_friction->m_cell_friction.empty();
								if( ! has_friction ) return;

								ldbg << "check nbh friction"<< std::endl;
								auto cells = grid->cells();

#     pragma omp parallel
								{
									GRID_OMP_FOR_BEGIN(dims-2*gl,_,block_loc, schedule(dynamic) )
									{
										IJK loc_a = block_loc + gl;
										size_t cell_a = grid_ijk_to_index( dims , loc_a );
										const size_t n_particles = cells[cell_a].size();
										const auto & cell_friction = nbh_friction->m_cell_friction[cell_a];
										if( n_particles != cell_friction.number_of_particles() )
										{
#           pragma omp critical(dbg_mesg)
											lerr << "Bad number of particles in cell #"<<cell_a<<" ("<<loc_a<< ") : " <<cell_friction.number_of_particles()<<" but "<<n_particles<<" expected"<<std::endl;
											std::abort();
										}
										for(size_t p=0;p<n_particles;p++)
										{
											if( cell_friction.particle_id(p) != cells[cell_a][field::id][p] )
											{
#             pragma omp critical(dbg_mesg)
												lerr << "Bad central particle id in cell #"<<cell_a<<", particle #"<<p <<" : found id "<<cell_friction.particle_id(p)<<", should be "<<cells[cell_a][field::id][p] <<std::endl;
												std::abort();              
											}
										}
									}
									GRID_OMP_FOR_END
								}

								ComputePairOptionalLocks<false> cp_locks {};
								exanb::GridChunkNeighborsLightWeightIt<false> nbh_it{ *chunk_neighbors };
								CheckNbhFrictionOp<GridT> check_op { *grid , *nbh_friction };
								ParticleNeighborFrictionIterator cp_friction{ nbh_friction->m_cell_friction.data() };

								if( domain->xform_is_identity() )
								{
									auto optional = make_compute_pair_optional_args( nbh_it, cp_friction, NullXForm{}, cp_locks );
									compute_pair_singlemat( *grid, *rcut, false /*no ghost*/, optional, make_default_pair_buffer(), check_op , compute_fields );
								}
								else
								{
									auto optional = make_compute_pair_optional_args( nbh_it, cp_friction , LinearXForm{ domain->xform() }, cp_locks );
									compute_pair_singlemat( *grid, *rcut, false /*no ghost*/, optional, make_default_pair_buffer(), check_op , compute_fields );
								}
								ldbg << "nbh_friction alignment with chunk_neighbors Ok, symetry Ok\n";
							}

						};

	template<class GridT> using CheckNbhFrictionTmpl = CheckNbhFriction<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{  
		OperatorNodeFactory::instance()->register_factory( "check_nbh_friction" , make_grid_variant_operator< CheckNbhFrictionTmpl > );
	}

}


