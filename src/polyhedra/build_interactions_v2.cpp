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
		ADD_SLOT(double , rcut_inc          , INPUT , 0.0 , DocString{"value added to the search distance to update neighbor list less frequently. in physical space"} );

		public:

		inline std::string documentation() const override final
		{
			return R"EOF(
				        )EOF";
		}

		inline void execute () override final
		{
			const auto cells = grid->cells();

			const IJK dims = grid->dimension();
			const int gl = grid->ghost_layers();

			auto & interactions = *nbh_interactions;

			auto & shps = *shapes_collection;
			double rVerlet = *rcut_inc;

			if( ! chunk_neighbors.has_value() ) 
			{
				return;
				interactions.clear();
			}
				interactions.clear();
/*
			// remove inactive interactions : 
			exanb::Vec3d null = {0,0,0};
			int last = interactions.size() - 1;
			for(int i = last ; i >= 0 ; i--)
			{
				if(interactions[i].moment == null && interactions[i].friction == null)
				{
					interactions[i] = interactions[last--];
				}
			}
*/

			auto add_contact = [](std::vector<Interaction>& list, Interaction& item, int sub_i, int sub_j) -> void
			{
				item.sub_i = sub_i;
				item.sub_j = sub_j;
			//	auto [exist, I] = exaDEM::get_interaction(list, item);
			//	if(exist) I.update(item);
			//	else list.push_back(item);
				list.push_back(item);
			};

#     pragma omp parallel
			{
				Interaction item;
				std::vector<Interaction> local;
				GRID_OMP_FOR_BEGIN(dims-2*gl,_,block_loc, schedule(dynamic) )
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
							[cells, cell_a, &local, &item, &shps, rVerlet, id_a, rx_a, ry_a, rz_a, t_a, orient_a, add_contact]
							( int p_a, size_t cell_b, unsigned int p_b , size_t p_nbh_index )
							{
							item.id_i = id_a[p_a];
							item.cell_j = cell_b;
							item.cell_i = cell_a;
							item.p_j = p_b;
							item.p_i = p_a;
							const uint64_t id_nbh = cells[cell_b][field::id][p_b];
							const double rx_nbh = cells[cell_b][field::rx][p_b];
							const double ry_nbh = cells[cell_b][field::ry][p_b];
							const double rz_nbh = cells[cell_b][field::rz][p_b];
							const double type_nbh = cells[cell_b][field::type][p_b];
							const Quaternion& orient_nbh = cells[cell_b][field::orient][p_b];

							const Vec3d origin = {0,0,0};
							const Vec3d r_nbh = Vec3d{rx_nbh, ry_nbh, rz_nbh} - Vec3d{rx_a[p_a], ry_a[p_a], rz_a[p_a]};

							const shape* shp = shps[t_a[p_a]];
							const shape* shp_nbh = shps[type_nbh];
							const int nv = shp->get_number_of_vertices();
							const int ne = shp->get_number_of_edges();
							const int nv_nbh = shp_nbh->get_number_of_vertices();
							const int ne_nbh = shp_nbh->get_number_of_edges();
							const int nf_nbh = shp_nbh->get_number_of_faces();

							item.id_j = id_nbh;
							item.type = 0; // === Vertex - Vertex
							for( int i = 0 ; i < nv ; i++)
							{
								for(int j = 0; j < nv_nbh ; j++)
								{
									auto contact = shape_polyhedron::filter_vertex_vertex(rVerlet, origin, i, shp, orient_a[p_a], r_nbh, j, shp_nbh, orient_nbh);
									if(contact) add_contact(local, item, i, j);
								}
							}

							item.type = 1; // === vertex edge
							for( int i = 0 ; i < nv ; i++)
							{
								for(int j = 0; j < ne_nbh ; j++)
								{
									auto contact = shape_polyhedron::filter_vertex_edge(rVerlet, origin, i, shp, orient_a[p_a], r_nbh, j, shp_nbh, orient_nbh);
									if(contact) add_contact(local, item, i, j);
								}
							}

							item.type = 2; // === vertex face
							for( int i = 0 ; i < nv ; i++)
							{
								for(int j = 0; j < nf_nbh ; j++)
								{
									auto contact = shape_polyhedron::filter_vertex_face(rVerlet, origin, i, shp, orient_a[p_a], r_nbh, j, shp_nbh, orient_nbh);
									if(contact) add_contact(local, item, i, j);
								}
							}

							item.type = 3; // === edge edge
							for( int i = 0 ; i < ne ; i++)
							{
								for(int j = 0; j < ne_nbh ; j++)
								{
									auto contact = shape_polyhedron::filter_edge_edge(rVerlet, origin, i, shp, orient_a[p_a], r_nbh, j, shp_nbh, orient_nbh);
									if(contact) add_contact(local, item, i, j);
								}
							}
							});
					if(local.size() > 0)
					{
#pragma omp critical
						{
							interactions.insert(interactions.end(), local.begin(), local.end());
						}
						local.clear();
					}
				}
				GRID_OMP_FOR_END
			}
		}
	};

	template<class GridT> using BuildPolyhedronInteractionTmpl = BuildPolyhedronInteraction<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "build_interactions_v2", make_grid_variant_operator< BuildPolyhedronInteractionTmpl > );
	}

}

