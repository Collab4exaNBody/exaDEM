#include <memory>
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/particle_neighbors/chunk_neighbors_apply.h>

#include <exaDEM/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/shapes.hpp>
#include <exaDEM/shape_detection.hpp>


namespace exaDEM
{
	using namespace exanb;

	template<typename GridT
		, class = AssertGridHasFields< GridT >
		>
		class UpdateGridCellInteraction : public OperatorNode
		{
			using ComputeFields = FieldSet<>;
			static constexpr ComputeFields compute_field_set {};

			ADD_SLOT( GridT                       , grid              , INPUT_OUTPUT , REQUIRED );
      ADD_SLOT( exanb::GridChunkNeighbors   , chunk_neighbors   , INPUT        , OPTIONAL , DocString{"Neighbor list"} );
			ADD_SLOT( GridCellParticleInteraction , grid_interaction  , INPUT_OUTPUT , DocString{"Interaction list"} );
    	ADD_SLOT( shapes                      , shapes_collection , INPUT        , DocString{"Collection of shapes"});
    	ADD_SLOT(double                       , rcut_inc          , INPUT_OUTPUT , DocString{"value added to the search distance to update neighbor list less frequently. in physical space"} );


			public:

			inline std::string documentation() const override final
			{
				return R"EOF(
				        )EOF";
			}


			inline void execute () override final
			{
				auto& g = *grid;
				const auto cells = g.cells();
				const size_t n_cells = g.number_of_cells(); // nbh.size();
				const IJK dims = g.dimension();
				const int gl = g.ghost_layers();
				auto & interactions = grid_interaction->m_data;

				// if grid structure (dimensions) changed, we invalidate thie whole data
				if( interactions.size() != n_cells )
				{
					ldbg << "number of cells changed, reset friction data" << std::endl;
					interactions.clear();
					interactions.resize( n_cells );
				}
				assert( interactions.size() == n_cells );

				if( ! chunk_neighbors.has_value() ) 
				{
#pragma omp parallel for schedule(static)
					for(size_t i = 0 ; i < n_cells ; i++) interactions[i].initialize(0);
					return;
				}

				auto & shps = *shapes_collection;
				double rVerlet = *rcut_inc;


				// use OBB for vertex/edge and vertex/faces
				constexpr bool skip_obb = false;

#     pragma omp parallel
				{
					Interaction item;
					std::vector<exaDEM::Interaction> local;
					std::vector<exaDEM::Interaction> local_history;
					GRID_OMP_FOR_BEGIN(dims-2*gl,_,block_loc, schedule(guided) )
					{
						IJK loc_a = block_loc + gl;
						size_t cell_a = grid_ijk_to_index( dims , loc_a );

						const unsigned int n_particles = cells[cell_a].size();
						CellExtraDynamicDataStorageT<Interaction>& storage = interactions[cell_a];
						storage.initialize(n_particles);
						if(n_particles == 0) continue;

						local.clear();

						const auto* __restrict__ id_a = cells[cell_a][ field::id ]; ONIKA_ASSUME_ALIGNED(id_a);
						const auto* __restrict__ rx_a = cells[cell_a][ field::rx ]; ONIKA_ASSUME_ALIGNED(rx_a);
						const auto* __restrict__ ry_a = cells[cell_a][ field::ry ]; ONIKA_ASSUME_ALIGNED(ry_a);
						const auto* __restrict__ rz_a = cells[cell_a][ field::rz ]; ONIKA_ASSUME_ALIGNED(rz_a);
						const auto* __restrict__ t_a = cells[cell_a][ field::type ]; ONIKA_ASSUME_ALIGNED(t_a);
						const auto* __restrict__ orient_a = cells[cell_a][ field::orient ]; ONIKA_ASSUME_ALIGNED(orient_a);

						auto& info_particles = storage.m_info;
						const size_t data_size = storage.m_data.size();
						const Interaction* const __restrict__ data_ptr  = storage.m_data.data();
						extract_history(local_history, data_ptr, data_size);
						std::sort ( local_history.begin(), local_history.end() );
						int last = 0;
						size_t offset = 0;

						auto add_contact = []( std::vector<Interaction>& list, Interaction& item, int sub_i, int sub_j) -> void
						{
							item.sub_i = sub_i;
							item.sub_j = sub_j;
							list.push_back(item);
						};

						auto incr_particle_interactions_info = [&info_particles, &offset] (int p_a)
						{
							offset ++;
							std::get<1> (info_particles [p_a])++;
						};

						// get particle id
						for( size_t it = 0 ; it < n_particles ; it++)
						{
							std::get<2> (info_particles[it]) = id_a[it];
						}

						apply_cell_particle_neighbors(*grid, *chunk_neighbors, cell_a, loc_a, std::false_type() /* not symetric */,
								[&g , cells, &last, &offset, &info_particles, cell_a, &local, &item, &shps, rVerlet, id_a, rx_a, ry_a, rz_a, t_a, orient_a, add_contact, incr_particle_interactions_info]
								( int p_a, size_t cell_b, unsigned int p_b , size_t p_nbh_index ){
/*
									if(p_a != last)
									{
										last = p_a;
										std::get<0> (info_particles[p_a]) = offset; 
									}*/

									// default value of the interaction studied (A or i -> B or j)
									const uint64_t id_nbh = cells[cell_b][field::id][p_b];

									if( id_a[p_a] >= id_nbh)
									{
									if ( !g.is_ghost_cell(cell_b) ) return;
									}

									const uint8_t type_nbh = cells[cell_b][field::type][p_b];
									const Quaternion orient_nbh = cells[cell_b][field::orient][p_b];
									const double rx_nbh = cells[cell_b][field::rx][p_b];
									const double ry_nbh = cells[cell_b][field::ry][p_b];
									const double rz_nbh = cells[cell_b][field::rz][p_b];

									// prev
									const shape* shp = shps[t_a[p_a]];
									const shape* shp_nbh = shps[type_nbh];
									OBB obb_i = shp->obb;
									OBB obb_j = shp_nbh->obb;
									const Quaternion& orient = orient_a[p_a];
									const double rx = rx_a[p_a];
									const double ry = ry_a[p_a];
									const double rz = rz_a[p_a];
									quat conv_orient_i = quat{vec3r{orient.x, orient.y, orient.z}, orient.w};
									quat conv_orient_j = quat{vec3r{orient_nbh.x, orient_nbh.y, orient_nbh.z}, orient_nbh.w};
									obb_i.rotate(conv_orient_i);
									obb_j.rotate(conv_orient_j);
									obb_i.translate(vec3r{rx, ry, rz});
									obb_j.translate(vec3r{rx_nbh, ry_nbh, rz_nbh});

									obb_i.enlarge(rVerlet);
									obb_j.enlarge(rVerlet);

									if ( ! obb_i.intersect(obb_j) ) return;

									// reset rVerlet
									obb_i . enlarge(-rVerlet);
									obb_j . enlarge(-rVerlet);

									// Add interactions
									item.id_i = id_a[p_a];
									item.p_i = p_a;

									item.cell_i = cell_a;
									item.p_j = p_b;
									item.cell_j = cell_b;

									const Vec3d r = {rx, ry, rz};
									const Vec3d r_nbh = {rx_nbh, ry_nbh, rz_nbh};

									// get particle j data.
									const int nv = shp->get_number_of_vertices();
									const int ne = shp->get_number_of_edges();
									const int nf = shp->get_number_of_faces();
									const int nv_nbh = shp_nbh->get_number_of_vertices();
									const int ne_nbh = shp_nbh->get_number_of_edges();
									const int nf_nbh = shp_nbh->get_number_of_faces();

									item.id_j = id_nbh;
									// exclude possibilities with obb
									for( int i = 0 ; i < nv ; i++)
									{
										auto vi = shp -> get_vertex (i, r, orient);
										OBB obbvi;
										obbvi.center = {vi.x, vi.y, vi.z};
										obbvi.enlarge(shp->m_radius + rVerlet);
										if (obb_j.intersect( obbvi ) )
										{
											item.type = 0; // === Vertex - Vertex
											for(int j = 0; j < nv_nbh ; j++)
											{
												bool contact = exaDEM::filter_vertex_vertex(rVerlet, r, i, shp, orient, r_nbh, j, shp_nbh, orient_nbh);
												if ( contact ) 
												{
													incr_particle_interactions_info(p_a);
													add_contact(local, item, i, j);
												}
											}

											item.type = 1; // === vertex edge
											for(int j = 0; j < ne_nbh ; j++)
											{
												bool contact = exaDEM::filter_vertex_edge <skip_obb> (obbvi, r_nbh, j, shp_nbh, orient_nbh);
												if(contact) 
												{
													incr_particle_interactions_info(p_a);
													add_contact(local, item, i, j);
												}
											}

											item.type = 2; // === vertex face
											for(int j = 0; j < nf_nbh ; j++)
											{
												bool contact = exaDEM::filter_vertex_face <skip_obb> (obbvi, r_nbh, j, shp_nbh, orient_nbh);
												if(contact) 
												{
													incr_particle_interactions_info(p_a);
													add_contact(local, item, i, j);
												}
											}
										}
									}

									item.type = 3; // === edge edge
									for( int i = 0 ; i < ne ; i++)
									{
										OBB obb_edge_i = shp->get_obb_edge(r, i, orient);
										if ( obb_j.intersect (obb_edge_i) )
										{
											obb_edge_i.enlarge(rVerlet);
											for(int j = 0; j < ne_nbh ; j++)
											{
												OBB obb_edge_j = shp_nbh->get_obb_edge(r_nbh, j, orient_nbh);
												if( obb_edge_i.intersect(obb_edge_j)) 
												{
													add_contact(local, item, i, j);
													incr_particle_interactions_info(p_a);
												}
											}
										}
									}

									item.cell_j = cell_a;
									item.id_j = id_a[p_a];
									item.p_j = p_a;

									item.cell_i = cell_b;
									item.p_i = p_b;
									item.id_i = id_nbh;

									for( int j = 0 ; j < nv_nbh ; j++)
									{
										auto vj = shp -> get_vertex (j, r_nbh, orient_nbh);
										OBB obbvj;
										obbvj.center = {vj.x, vj.y, vj.z};
										obbvj.enlarge(shp_nbh->m_radius + rVerlet);

										if (obb_i.intersect( obbvj ) )
										{
											item.type = 1; // === vertex edge
											for(int i = 0; i < ne ; i++)
											{
												bool contact = exaDEM::filter_vertex_edge <skip_obb> (obbvj, r, i, shp, orient);
												if(contact) 
												{
													add_contact(local, item, j, i);
													incr_particle_interactions_info(p_a);
												}
											}

											item.type = 2; // === vertex face
											for(int i = 0; i < nf ; i++)
											{
												bool contact = exaDEM::filter_vertex_face <skip_obb> (obbvj, r, i, shp, orient);
												if(contact) 
												{
													add_contact(local, item, j, i);
													incr_particle_interactions_info(p_a);
												}
											}
										}
									}
								});

						// update offset for interaction
						std::get<0> (info_particles[0]) = 0;
						for ( size_t it = 1 ; it < info_particles.size() ; it++)
						{
							std::get<0> (info_particles[it]) = std::get<0> (info_particles[it-1]) + std::get<1> (info_particles[it-1]);
						} 

						// add history, local history and local have to be sorted. local is sorted by construction.
						update_friction_moment(local, local_history);
						storage.set_data_storage(local);
					}
					GRID_OMP_FOR_END
				}
			}
		};

	template<class GridT> using UpdateGridCellInteractionTmpl = UpdateGridCellInteraction<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "update_grid_interaction", make_grid_variant_operator< UpdateGridCellInteraction > );
	}
}

