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

#include <exaDEM/interaction.hpp>
#include <exaDEM/shapes.hpp>
#include <exaDEM/shape_detection.hpp>

namespace exaDEM
{
	using namespace exanb;

	template<typename GridT, class = AssertGridHasFields< GridT, field::_id, field::_type, field::_orient >>
		class BuildCylinderInteraction : public OperatorNode
	{
		// attributes processed during computation
		using ComputeFields = FieldSet< field::_vrot, field::_arot >;
		static constexpr ComputeFields compute_field_set {};
		Vec3d null = {0,0,0};
		Vec3d default_axis = Vec3d{1,0,1};
		ADD_SLOT( GridT           , grid                      , INPUT_OUTPUT , REQUIRED );
		ADD_SLOT( shapes          , shapes_collection         , INPUT , shapes()                   , DocString{"Collection of shapes"});
		ADD_SLOT( std::vector<Interaction> , cylinder_interactions , INPUT_OUTPUT , DocString{"TODO"} );
		ADD_SLOT( double          , rcut_inc                  , INPUT        , 0.0      , DocString{"value added to the search distance to update neighbor list less frequently. in physical space"} );
		ADD_SLOT( double          , cylinder_radius           , INPUT        , REQUIRED , DocString{"Radius of the cylinder, positive and should be superior to the biggest sphere radius in the cylinder"});
		ADD_SLOT( Vec3d           , cylinder_center           , INPUT        , REQUIRED , DocString{"Center of the cylinder"});
		ADD_SLOT( Vec3d           , cylinder_axis             , INPUT        , REQUIRED , DocString{"Define the plan of the cylinder"});
		ADD_SLOT( Vec3d           , cylinder_angular_velocity , INPUT        , null     , DocString{"Angular velocity of the cylinder, default is 0 m.s-"});
		ADD_SLOT( Vec3d           , cylinder_velocity         , INPUT        , null     , DocString{"Cylinder velocity, could be used in 'expert mode'"});

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

			auto & interactions = *cylinder_interactions;

			auto & shps = *shapes_collection;
			double rVerlet = *rcut_inc;

			const double radius = *cylinder_radius;
			const Vec3d axis = *cylinder_axis;
			const Vec3d center_proj = (*cylinder_center) * axis; 

			std::vector<Interaction> cylinder_history =	extract_history_omp(interactions);
			interactions.clear();

			const uint64_t id_j = 0;
			const size_t cell_j = 0;
			const size_t p_j = 0;
			exanb::Vec3d null = {0.,0.,0.};

#     pragma omp parallel
			{
				Interaction item;
				item.id_j = id_j;
				item.cell_j = cell_j;
				item.p_j = p_j;
				item.sub_j = 0;
				item.moment = null;
				item.friction = null;
				item.type = 4; // type : cylinder
				std::vector<Interaction> local;
				GRID_OMP_FOR_BEGIN(dims-2*gl,_,block_loc, schedule(dynamic)) 
				{
					IJK loc_a = block_loc + gl;
					size_t cell_a = grid_ijk_to_index( dims , loc_a );
					const size_t n_particles = cells[cell_a].size();
					const auto* __restrict__ id = cells[cell_a][ field::id ]; ONIKA_ASSUME_ALIGNED(id);
					const auto* __restrict__ rx = cells[cell_a][ field::rx ]; ONIKA_ASSUME_ALIGNED(rx);
					const auto* __restrict__ ry = cells[cell_a][ field::ry ]; ONIKA_ASSUME_ALIGNED(ry);
					const auto* __restrict__ rz = cells[cell_a][ field::rz ]; ONIKA_ASSUME_ALIGNED(rz);
					const auto* __restrict__ t = cells[cell_a][ field::type ]; ONIKA_ASSUME_ALIGNED(t);
					const auto* __restrict__ orient = cells[cell_a][ field::orient ]; ONIKA_ASSUME_ALIGNED(orient);

					item.cell_i = cell_a;
					for(size_t j = 0 ; j < n_particles ; j++)
					{
						const shape* shp = shps[t[j]];
						item.id_i = id[j];
						item.p_i = j;

						const Vec3d proj = Vec3d{rx[j], ry[j], rz[j]} * axis;
						int nv = shp->get_number_of_vertices();
						for(int sub = 0 ; sub < nv ; sub++)
						{
							auto contact = exaDEM::filter_vertex_cylinder(rVerlet, proj, sub, shp, orient[j], center_proj, axis, radius);
							if(contact)
							{
								item.sub_i = sub;
								local.push_back(item);	
							}
						}
					}
				}
				GRID_OMP_FOR_END;

				if(local.size()>0)
				{
#pragma omp critical
					{
						interactions.insert(interactions.end(), local.begin(), local.end());
					}
					local.clear();
				}
			}
			update_friction_moment_omp(interactions, cylinder_history);
		}
	};

	template<class GridT> using BuildCylinderInteractionTmpl = BuildCylinderInteraction<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "build_cylinder_interactions_v2", make_grid_variant_operator< BuildCylinderInteractionTmpl > );
	}
}
