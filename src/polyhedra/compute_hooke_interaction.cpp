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
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/shape/shapes.hpp>
#include <exaDEM/shape/shape_detection.hpp>
#include <exaDEM/shape/shape_detection_driver.hpp>
#include <exaDEM/mutexes.h>
#include <exaDEM/drivers.h>

namespace exaDEM
{
	using namespace exanb;

	template<typename GridT , class = AssertGridHasFields< GridT, field::_radius >>
		class ComputeHookeInteraction : public OperatorNode
	{
		// attributes processed during computation
		using ComputeFields = FieldSet< field::_vrot, field::_arot >;
		static constexpr ComputeFields compute_field_set {};

		ADD_SLOT( GridT       , grid              , INPUT_OUTPUT , REQUIRED );
		ADD_SLOT( GridCellParticleInteraction , ges  , INPUT_OUTPUT , DocString{"Interaction list"} );
		ADD_SLOT( shapes      , shapes_collection , INPUT_OUTPUT , DocString{"Collection of shapes"});
		ADD_SLOT( HookeParams , config            , INPUT , REQUIRED ); // can be re-used for to dump contact network
		ADD_SLOT( HookeParams , config_driver     , INPUT , OPTIONAL ); // can be re-used for to dump contact network
		ADD_SLOT( mutexes     , locks             , INPUT_OUTPUT );
		ADD_SLOT( double      , dt                , INPUT , REQUIRED );
		ADD_SLOT( Drivers     , drivers           , INPUT , DocString{"List of Drivers"});
		ADD_SLOT( std::vector<size_t>         , idxs              , INPUT_OUTPUT , DocString{"List of non empty cells"});


		public:

		inline std::string documentation() const override final
		{
			return R"EOF(
				        )EOF";
		}

		// C for cell and D for driver
		template<typename C, typename D>
			inline void compute_driver(C& cell, D& driver, const HookeParams& hkp, const shape* shp, Interaction& I, const double time, mutexes& locker)
			{
				const size_t p   = I.p_i;
				const size_t sub = I.sub_i;
				// === positions
				const Vec3d r       = { cell[field::rx][p], cell[field::ry][p], cell[field::rz][p] };
				// === vrot
				const Vec3d& vrot  = cell[field::vrot][p];
				// === vertex array
				const auto& vertices =  cell[field::vertices][p];

				auto [contact, dn, n, contact_position] = exaDEM::detector_vertex_driver(driver, vertices, sub, shp);

				if(contact)
				{
					constexpr Vec3d null = {0,0,0};
					auto& mom = cell[field::mom][p];
					const Vec3d v = { cell[field::vx][p], cell[field::vy][p], cell[field::vz][p] };
					const double meff = cell[field::mass][p];
					Vec3d f = null;
					hooke_force_core(dn, n, time, hkp.m_kn, hkp.m_kt, hkp.m_kr,
							hkp.m_mu, hkp.m_damp_rate, meff,
							I.friction, contact_position,
							r, v, f, I.moment, vrot,  // particle i
							driver.center, driver.vel, driver.vrot // particle j
							);

					// === update informations
					locker.lock(I.cell_i, p);
					mom += compute_moments(contact_position, r, f, I.moment);
					cell[field::fx][p] += f.x;
					cell[field::fy][p] += f.y;
					cell[field::fz][p] += f.z;
					locker.unlock(I.cell_i, p);
				}
				else
				{
					I.reset();
				}
			}

		inline void execute () override final
		{
			//using data_t = std::variant<exaDEM::Cylinder, exaDEM::Surface, exaDEM::UndefinedDriver>;
			if( grid->number_of_cells() == 0 ) { return; }

			Drivers empty;
			Drivers& drvs =  drivers.has_value() ? *drivers : empty;

			const auto cells = grid->cells();
			auto & cell_interactions = ges->m_data;
			auto & shps = *shapes_collection;
			const HookeParams params = *config;
			HookeParams hkp_drvs;

			if ( drivers->get_size() > 0 &&  config_driver.has_value() )
			{
				hkp_drvs = *config_driver;
			}

			const double time = *dt;
			mutexes& locker = *locks;

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


			auto& indexes = *idxs;

#pragma omp parallel
			{
				auto detection = std::vector{

					exaDEM::detection_vertex_vertex_precompute,
						exaDEM::detection_vertex_edge_precompute,
						exaDEM::detection_vertex_face_precompute,
						exaDEM::detection_edge_edge_precompute

				};


#pragma omp for schedule(dynamic)
/*				for(size_t current_cell = 0 ; current_cell < cell_interactions.size() ; current_cell++)
				{
*/
				for( size_t ci = 0 ; ci < indexes.size() ; ci ++ )
{
size_t current_cell = indexes[ci];  

					auto& interactions = cell_interactions[current_cell];
					const unsigned int  n_interactions_in_cell = interactions.m_data.size();
					exaDEM::Interaction* const __restrict__ data_ptr = onika::cuda::vector_data( interactions.m_data ); 


					for( size_t it = 0; it < n_interactions_in_cell ; it++ )
					{
						Interaction& item = data_ptr[it];

						if( item.type < 4 ) // polyhedra 
						{
							// === positions
							const Vec3d ri = get_r(item.cell_i, item.p_i);
							const Vec3d rj = get_r(item.cell_j, item.p_j);

							// === cell
							auto& cell_i =  cells[item.cell_i];
							auto& cell_j =  cells[item.cell_j];

							// === vrot
							const Vec3d& vrot_i = cell_i[field::vrot][item.p_i];
							const Vec3d& vrot_j = cell_j[field::vrot][item.p_j];

							// === type
							const auto& type_i = cell_i[field::type][item.p_i];
							const auto& type_j = cell_j[field::type][item.p_j];

							// === vertex array
							const auto& vertices_i =  cell_i[field::vertices][item.p_i];
							const auto& vertices_j =  cell_j[field::vertices][item.p_j];

							// === shapes
							const shape* shp_i = shps[type_i];
							const shape* shp_j = shps[type_j];

							auto [contact, dn, n, contact_position] = detection[item.type](vertices_i, item.sub_i, shp_i, vertices_j, item.sub_j, shp_j);

							if(contact)
							{
								const Vec3d vi = get_v(item.cell_i, item.p_i);
								const Vec3d vj = get_v(item.cell_j, item.p_j);
								const auto& m_i = cell_i[field::mass][item.p_i];
								const auto& m_j = cell_j[field::mass][item.p_j];

								// temporary vec3d to store forces.
								Vec3d f = {0,0,0};
								const double meff = compute_effective_mass(m_i, m_j);

								hooke_force_core(dn, n, time, params.m_kn, params.m_kt, params.m_kr,
										params.m_mu, params.m_damp_rate, meff,
										item.friction, contact_position,
										ri, vi, f, item.moment, vrot_i,  // particle 1
										rj, vj, vrot_j // particle nbh
										);


								// === update particle informations
								// ==== Particle i
								locker.lock(item.cell_i, item.p_i);

								auto& mom_i = cell_i[field::mom][item.p_i];
								mom_i += compute_moments(contact_position, ri, f, item.moment);
								cell_i[field::fx][item.p_i] += f.x;
								cell_i[field::fy][item.p_i] += f.y;
								cell_i[field::fz][item.p_i] += f.z;

								locker.unlock(item.cell_i, item.p_i);

								// ==== Particle j
								locker.lock(item.cell_j, item.p_j);

								auto& mom_j = cell_j[field::mom][item.p_j];
								mom_j += compute_moments(contact_position, rj, -f, -item.moment);
								cell_j[field::fx][item.p_j] -= f.x;
								cell_j[field::fy][item.p_j] -= f.y;
								cell_j[field::fz][item.p_j] -= f.z;

								locker.unlock(item.cell_j, item.p_j);
							}
							else
							{
								item.reset();
							}
						}
						else // drivers
						{
							const int driver_idx = item.id_j; //
							const auto type = cells[item.cell_i][field::type][item.p_i];
							auto* shp = shps[type];
							if (item.type == 4)
							{
								Cylinder& driver = std::get<Cylinder>(drvs.data(driver_idx)) ; 
								compute_driver(cells[item.cell_i], driver, hkp_drvs, shp, item, time, locker);
							}
							else if (item.type == 5)
							{
								Surface& driver = std::get<Surface>(drvs.data(driver_idx)) ; 
								compute_driver(cells[item.cell_i], driver, hkp_drvs, shp, item, time, locker);
							}
							else if (item.type == 6)
							{
								Ball& driver = std::get<Ball>(drvs.data(driver_idx)) ; 
								compute_driver(cells[item.cell_i], driver, hkp_drvs, shp, item, time, locker);
							}
						}
					}
				}
			}
		}
	};

	template<class GridT> using ComputeHookeInteractionTmpl = ComputeHookeInteraction<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "compute_hooke_interaction", make_grid_variant_operator< ComputeHookeInteractionTmpl > );
	}
}

