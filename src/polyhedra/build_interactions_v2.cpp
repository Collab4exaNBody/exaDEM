//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>

#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/particle_neighbors/chunk_neighbors_apply.h>

#include <memory>
#include <chrono>
#include <ctime>
#include <cmath>


#include <exaDEM/hooke_force_parameters.h>
#include <exaDEM/compute_hooke_force.h>
#include <exaDEM/interaction.hpp>
#include <exaDEM/shapes.hpp>
#include <exaDEM/shape_detection.hpp>

namespace exaDEM
{
  using namespace exanb;

  template<typename GridT
    , class = AssertGridHasFields< GridT, field::_radius >
    >
  class BuildPolyhedronInteraction : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet< field::_vrot, field::_arot >;
    static constexpr ComputeFields compute_field_set {};

    ADD_SLOT( GridT  , grid     , INPUT_OUTPUT , REQUIRED );
    ADD_SLOT( exanb::GridChunkNeighbors , chunk_neighbors   , INPUT , OPTIONAL , DocString{"neighbor list"} );
    ADD_SLOT( std::vector<Interaction> , nbh_interactions , INPUT_OUTPUT , DocString{"TODO"} );
    ADD_SLOT( shapes , shapes_collection, INPUT , DocString{"Collection of shapes"});
		ADD_SLOT(double , rcut_inc          , INPUT_OUTPUT , DocString{"value added to the search distance to update neighbor list less frequently. in physical space"} );

		public:

		inline std::string documentation() const override final
		{
			return R"EOF(
				        )EOF";
		}

		inline void execute () override final
		{
			const auto cells = grid->cells();

			auto& g = *grid;
			const IJK dims = grid->dimension();
			const int gl = grid->ghost_layers();

			auto & interactions = *nbh_interactions;

			auto & shps = *shapes_collection;
			double rVerlet = *rcut_inc;

			if( ! chunk_neighbors.has_value() ) 
			{
				interactions.clear();
				return;
			}

			std::vector<Interaction> history = extract_history_omp(interactions);
			std::sort (history.begin(), history.end());
			interactions.clear();

			auto add_contact = [](std::vector<Interaction>& list, Interaction& item, int sub_i, int sub_j) -> void
			{
				item.sub_i = sub_i;
				item.sub_j = sub_j;
				list.push_back(item);
			};

			// use OBB for vertex/edge and vertex/faces
			constexpr bool skip_obb = false;
			int shift = 0;

#     pragma omp parallel
			{
				Interaction item;
				std::vector<Interaction> local;
				local.reserve(interactions.size() / omp_get_num_threads());
				GRID_OMP_FOR_BEGIN(dims-2*gl,_,block_loc, schedule(guided) )
				{
					IJK loc_a = block_loc + gl;
					size_t cell_a = grid_ijk_to_index( dims , loc_a );

					const auto* __restrict__ id_a = cells[cell_a][ field::id ]; ONIKA_ASSUME_ALIGNED(id_a);
					const auto* __restrict__ rx_a = cells[cell_a][ field::rx ]; ONIKA_ASSUME_ALIGNED(rx_a);
					const auto* __restrict__ ry_a = cells[cell_a][ field::ry ]; ONIKA_ASSUME_ALIGNED(ry_a);
					const auto* __restrict__ rz_a = cells[cell_a][ field::rz ]; ONIKA_ASSUME_ALIGNED(rz_a);
					const auto* __restrict__ t_a = cells[cell_a][ field::type ]; ONIKA_ASSUME_ALIGNED(t_a);
					const auto* __restrict__ orient_a = cells[cell_a][ field::orient ]; ONIKA_ASSUME_ALIGNED(orient_a);

					apply_cell_particle_neighbors(*grid, *chunk_neighbors, cell_a, loc_a, std::false_type() /* not symetric */,
//					apply_cell_particle_neighbors(*grid, *chunk_neighbors, cell_a, loc_a, std::true_type() /* is symetric */,
							[&g , cells, cell_a, &local, &item, &shps, rVerlet, id_a, rx_a, ry_a, rz_a, t_a, orient_a, add_contact]
							( int p_a, size_t cell_b, unsigned int p_b , size_t p_nbh_index )
							{
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
										if ( contact ) add_contact(local, item, i, j);
									}

									item.type = 1; // === vertex edge
									for(int j = 0; j < ne_nbh ; j++)
									{
										bool contact = exaDEM::filter_vertex_edge <skip_obb> (obbvi, r_nbh, j, shp_nbh, orient_nbh);
										if(contact) add_contact(local, item, i, j);
									}

									item.type = 2; // === vertex face
									for(int j = 0; j < nf_nbh ; j++)
									{
										bool contact = exaDEM::filter_vertex_face <skip_obb> (obbvi, r_nbh, j, shp_nbh, orient_nbh);
										if(contact) add_contact(local, item, i, j);
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
										if( obb_edge_i.intersect(obb_edge_j)) add_contact(local, item, i, j);
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
										if(contact) add_contact(local, item, j, i);
									}

									item.type = 2; // === vertex face
									for(int i = 0; i < nf ; i++)
									{
										bool contact = exaDEM::filter_vertex_face <skip_obb> (obbvj, r, i, shp, orient);
										if(contact) add_contact(local, item, j, i);
									}
								}
							}
							});
					if(local.size() > 0)
					{
						//						update_friction_moment(local, history);
						std::sort (local.begin(), local.end());
						update_friction_moment(local, history);
#pragma omp critical
						{
							//							interactions.insert(interactions.end(), local.begin(), local.end());

							size_t size = local.size();
							if( size + shift > interactions.size() ) interactions.resize(size+shift);
							std::copy( local.begin(), local.end(), interactions.data() + shift);
							shift += local.size();
							local.clear();

						}
					}
				}
				GRID_OMP_FOR_END
			}

			interactions.resize(shift);
			//std::cout << "create_history: " << t_history << " all: " << t_all << " add history: "<< t_end << std::endl; 
		}
	};

	template<class GridT> using BuildPolyhedronInteractionTmpl = BuildPolyhedronInteraction<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "build_interactions_v2", make_grid_variant_operator< BuildPolyhedronInteractionTmpl > );
	}

}

