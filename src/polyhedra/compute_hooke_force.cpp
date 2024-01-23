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
#include <exaDEM/neighbor_type.h>
#include <exaDEM/interaction.hpp>
#include <exaDEM/shapes.hpp>
#include <exaDEM/shape_detection.hpp>

namespace exaDEM
{
	using namespace exanb;

	template<typename GridT
		, class = AssertGridHasFields< GridT, field::_radius >
		>
		class ComputeHookeInteraction : public OperatorNode
		{
			// attributes processed during computation
			using ComputeFields = FieldSet< field::_vrot, field::_arot >;
			static constexpr ComputeFields compute_field_set {};

			ADD_SLOT( GridT  , grid     , INPUT_OUTPUT , REQUIRED );
			ADD_SLOT( std::vector<Interaction> , nbh_interactions , INPUT_OUTPUT , DocString{"TODO"} );
			ADD_SLOT( shapes , shapes_collection, INPUT_OUTPUT , DocString{"Collection of shapes"});
			ADD_SLOT( HookeParams           , config            , INPUT , REQUIRED );
			ADD_SLOT( double                , dt                , INPUT , REQUIRED );


			public:

			inline std::string documentation() const override final
			{
				return R"EOF(
				        )EOF";
			}

			inline void execute () override final
			{
				const auto cells = grid->cells();
				auto & interactions = *nbh_interactions;
				auto & shps = *shapes_collection;
				const HookeParams params = *config;
				const double time = *dt;

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
					auto detection = std::vector{
						shape_polyhedron::detection_vertex_vertex,
							shape_polyhedron::detection_vertex_edge,
							shape_polyhedron::detection_vertex_face,
							shape_polyhedron::detection_edge_edge
					};

#pragma omp for schedule(dynamic)
					for(size_t it = 0 ; it < interactions.size() ; it++)
					{
						Interaction& item = interactions[it];

						// === positions
						const Vec3d ri = get_r(item.cell_i, item.p_i);
						const Vec3d rj = get_r(item.cell_j, item.p_j);
						const Vec3d origin = {0,0,0};
						const Vec3d dr = rj - ri;

						// === cell
						auto& cell_i =  cells[item.cell_i];
						auto& cell_j =  cells[item.cell_j];

						// === vrot
						const Vec3d& vrot_i = cell_i[field::vrot][item.p_i];
						const Vec3d& vrot_j = cell_j[field::vrot][item.p_j];

						// === type
						const auto& type_i = cell_i[field::type][item.p_i];
						const auto& type_j = cell_j[field::type][item.p_j];

						// === orientation
						const auto& orient_i = cell_i[field::orient][item.p_i];
						const auto& orient_j = cell_j[field::orient][item.p_j];

						// === shapes
						const shape* shp_i = shps[type_i];
						const shape* shp_j = shps[type_j];

						auto [contact, dn, n, contact_position] = detection[item.type](origin, item.sub_i, shp_i, orient_i, dr, item.sub_j, shp_j, orient_j);

						if(contact)
						{
							const Vec3d vi = get_v(item.cell_i, item.p_i);
							const Vec3d vj = get_v(item.cell_j, item.p_j);
							const auto& m_i = cell_i[field::mass][item.p_i];
							const auto& m_j = cell_j[field::mass][item.p_j];

							Vec3d f = {0,0,0};
							const double meff = compute_effective_mass(m_i, m_j);

							hooke_force_core(dn, n, time, params.m_kn, params.m_kt, params.m_kr,
									params.m_mu, params.m_damp_rate, meff,
									item.friction, contact_position,
									origin, vi, f, item.moment, vrot_i,  // particle 1
									dr, vj, vrot_j // particle nbh
									);

							// === update informations
							//auto& mom = cell_i[field::mom][item.p_i];
							//update_moments(mom, contact_position, ri, f, item.moment);
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

	template<class GridT> using ComputeHookeInteractionTmpl = ComputeHookeInteraction<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "compute_hooke_interactions_v2", make_grid_variant_operator< ComputeHookeInteractionTmpl > );
	}
}

