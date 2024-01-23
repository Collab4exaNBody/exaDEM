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
		class ComputerCylinderInteraction : public OperatorNode
		{
			// attributes processed during computation
			using ComputeFields = FieldSet< field::_vrot, field::_arot >;
			static constexpr ComputeFields compute_field_set {};
			static constexpr Vec3d null = {0,0,0};

			ADD_SLOT( GridT       , grid     , INPUT_OUTPUT , REQUIRED );
			ADD_SLOT( std::vector<Interaction> , cylinder_interactions , INPUT_OUTPUT , DocString{"TODO"} );
			ADD_SLOT( shapes      , shapes_collection, INPUT_OUTPUT , DocString{"Collection of shapes"});
			ADD_SLOT( HookeParams , config            , INPUT , REQUIRED );
			ADD_SLOT( double      , dt                , INPUT , REQUIRED );
			ADD_SLOT( double      , cylinder_radius           , INPUT        , REQUIRED , DocString{"Radius of the cylinder, positive and should be superior to the biggest sphere radius in the cylinder"});
			ADD_SLOT( Vec3d       , cylinder_center           , INPUT        , REQUIRED , DocString{"Center of the cylinder"});
			ADD_SLOT( Vec3d       , cylinder_axis             , INPUT        , REQUIRED , DocString{"Define the plan of the cylinder"});
			ADD_SLOT( Vec3d       , cylinder_angular_velocity , INPUT        , null     , DocString{"Angular velocity of the cylinder, default is 0 m.s-"});
			ADD_SLOT( Vec3d       , cylinder_velocity         , INPUT        , null     , DocString{"Cylinder velocity, could be used in 'expert mode'"});


			public:

			inline std::string documentation() const override final
			{
				return R"EOF(
				        )EOF";
			}

			inline void execute () override final
			{
				const auto cells         = grid->cells();
				auto & interactions      = *cylinder_interactions;
				auto & shps              = *shapes_collection;
				const HookeParams params = *config;
				const double time        = *dt;

				const Vec3d& axis   = *cylinder_axis;
				const Vec3d rj      = (*cylinder_center) * axis;
				const Vec3d vj      = *cylinder_velocity;
				const Vec3d& vrot_j = *cylinder_angular_velocity;
				const double radius = *cylinder_radius;

				auto get_r = [&cells] (const int cell_id, const int p_id) -> const Vec3d 
				{
					const Vec3d res = {
						cells[cell_id][field::rx][p_id],
						cells[cell_id][field::ry][p_id],
						cells[cell_id][field::rz][p_id]};
					return res;
				};

				auto get_v = [&cells] (const int cell_id, const int p_id) -> const Vec3d 
				{
					const Vec3d res = {
						cells[cell_id][field::vx][p_id],
						cells[cell_id][field::vy][p_id],
						cells[cell_id][field::vz][p_id]};
					return res;
				};



#pragma omp parallel
				{
#pragma omp for schedule(dynamic)
					for(size_t it = 0 ; it < interactions.size() ; it++)
					{
						Interaction& item    = interactions[it];

						// === positions
						const Vec3d ri       = get_r(item.cell_i, item.p_i) * axis;
						// === cell
						auto& cell_i         = cells[item.cell_i];
						// === vrot
						const Vec3d& vrot_i  = cell_i[field::vrot][item.p_i];
						// === type
						const auto& type_i   = cell_i[field::type][item.p_i];
						// === orientation
						const auto& orient_i = cell_i[field::orient][item.p_i];
						// === shapes
						const shape* shp_i   = shps[type_i];

						auto [contact, dn, n, contact_position] = shape_polyhedron::detection_vertex_cylinder(ri, item.sub_i, shp_i, orient_i, rj, axis, radius);

						if(contact)
						{
							auto& mom = cell_i[field::mom][item.p_i];
							const Vec3d vi = get_v(item.cell_i, item.p_i);
							const double meff = cell_i[field::mass][item.p_i];
							Vec3d f = {0,0,0};
							auto& cell_i = cells[item.cell_i];
	
							hooke_force_core(dn, n, time, params.m_kn, params.m_kt, params.m_kr,
									params.m_mu, params.m_damp_rate, meff,
									item.friction, contact_position,
									ri, vi, f, item.moment, vrot_i,  // particle i
									rj, vj, vrot_j // particle j
									);
							// === update informations
							update_moments(mom, contact_position, ri, f, item.moment);
							cell_i[field::fx][item.p_i] += f.x;
							cell_i[field::fy][item.p_i] += f.y;
							cell_i[field::fz][item.p_i] += f.z;
						}
						else
						{
							item.reset();
						}
					}
				}
			}
		};

	template<class GridT> using ComputerCylinderInteractionTmpl = ComputerCylinderInteraction<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "compute_cylinder_interactions_v2", make_grid_variant_operator< ComputerCylinderInteractionTmpl > );
	}
}

