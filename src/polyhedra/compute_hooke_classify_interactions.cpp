/*
	 Licensed to the Apache Software Foundation (ASF) under one
	 or more contributor license agreements.  See the NOTICE file
	 distributed with this work for additional information
	 regarding copyright ownership.  The ASF licenses this file
	 to you under the Apache License, Version 2.0 (the
	 "License"); you may not use this file except in compliance
	 with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
 */
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
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/classifier.hpp>
#include <exaDEM/shape/shapes.hpp>
#include <exaDEM/shape/shape_detection.hpp>
#include <exaDEM/shape/shape_detection_driver.hpp>
#include <exaDEM/mutexes.h>
#include <exaDEM/drivers.h>
#include <exaDEM/compute_hooke_interaction.h>



namespace exaDEM
{
	using namespace exanb;

	struct hooke_law_polyhedra
	{
		template<typename Cells>
			ONIKA_HOST_DEVICE_FUNC void operator()(Interaction& item, Cells& cells, const HookeParams& hkp, const shapes& shps, const double time, mutexes& locker)
			{
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

				auto [contact, dn, n, contact_position] = detect[item.type](vertices_i, item.sub_i, shp_i, vertices_j, item.sub_j, shp_j);
				if(contact)
				{
					const Vec3d vi = get_v(item.cell_i, item.p_i);
					const Vec3d vj = get_v(item.cell_j, item.p_j);
					const auto& m_i = cell_i[field::mass][item.p_i];
					const auto& m_j = cell_j[field::mass][item.p_j];

					// temporary vec3d to store forces.
					Vec3d f = {0,0,0};
					const double meff = compute_effective_mass(m_i, m_j);

					hooke_force_core(dn, n, time, hkp.m_kn, hkp.m_kt, hkp.m_kr,
							hkp.m_mu, hkp.m_damp_rate, meff,
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
		typedef decltype (&(exaDEM::detection_vertex_vertex_precompute)) Detector;
		const std::vector<Detector> detect = std::vector{ exaDEM::detection_vertex_vertex_precompute, 
			 exaDEM::detection_vertex_edge_precompute,
			 exaDEM::detection_vertex_face_precompute,
			 exaDEM::detection_edge_edge_precompute};
};

// C for cell and D for driver
template<typename TMPLD>
struct hooke_law_driver
{
	template<typename Cells>
		ONIKA_HOST_DEVICE_FUNC void operator()(Interaction& item, Cells& cells, Drivers& drvs, const HookeParams& hkp, const shapes& shps, const double time, mutexes& locker)
		{
			const int driver_idx = item.id_j; //
			auto& driver = std::get<TMPLD>(drvs.data(driver_idx)) ;
			auto& cell = cells[item.cell_i];
			const auto type = cell[field::type][item.p_i];
			auto* shp = shps[type];

			const size_t p   = item.p_i;
			const size_t sub = item.sub_i;
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
						item.friction, contact_position,
						r, v, f, item.moment, vrot,  // particle i
						driver.center, driver.get_vel(), driver.vrot // particle j
						);

				// === update informations
				locker.lock(item.cell_i, p);
				mom += compute_moments(contact_position, r, f, item.moment);
				cell[field::fx][p] += f.x;
				cell[field::fy][p] += f.y;
				cell[field::fz][p] += f.z;
				locker.unlock(item.cell_i, p);
			}
			else
			{
				item.reset();
			}
		}
};

struct hooke_law_stl
{
	template<typename Cells>
		ONIKA_HOST_DEVICE_FUNC void operator()( Interaction& item, Cells& cells, Drivers& drvs, const HookeParams& hkp, const shapes shps, const double time, mutexes& locker)
		{
			const int driver_idx = item.id_j; //
			auto& driver = std::get<Stl_mesh>(drvs.data(driver_idx)) ;
			auto& cell = cells[item.cell_i];
			const auto type = cell[field::type][item.p_i];
			auto* shp_i = shps[type];

			const size_t p_i   = item.p_i;
			const size_t sub_i = item.sub_i;
			const size_t sub_j = item.sub_j;


			// === positions
			const Vec3d r_i       = { cell[field::rx][p_i], cell[field::ry][p_i], cell[field::rz][p_i] };
			// === vrot
			const Vec3d& vrot_i  = cell[field::vrot][p_i];
			const Quaternion& orient_i  = cell[field::orient][p_i];
			const auto& shp_j = driver.shp;

			const Quaternion orient_j = {1.0,0.0,0.0,0.0};
			auto [contact, dn, n, contact_position] = func(item.type, r_i, sub_i, shp_i, orient_i, driver.center, sub_j, &shp_j, orient_j);

			if(contact)
			{
				constexpr Vec3d null = {0,0,0};
				auto& mom = cell[field::mom][p_i];
				const Vec3d v_i = { cell[field::vx][p_i], cell[field::vy][p_i], cell[field::vz][p_i] };
				const double meff = cell[field::mass][p_i];
				Vec3d f = null;
				hooke_force_core(dn, n, time, hkp.m_kn, hkp.m_kt, hkp.m_kr,
						hkp.m_mu, hkp.m_damp_rate, meff,
						item.friction, contact_position,
						r_i, v_i, f, item.moment, vrot_i,  // particle i
						driver.center, driver.vel, driver.vrot // particle j
						);

				// === update informations
				locker.lock(item.cell_i, p_i);
				mom += compute_moments(contact_position, r_i, f, item.moment);
				cell[field::fx][p_i] += f.x;
				cell[field::fy][p_i] += f.y;
				cell[field::fz][p_i] += f.z;
				locker.unlock(item.cell_i, p_i);
			}
			else
			{
				item.reset();
			}
		}

	const stl_mesh_dectector func;
};
}

namespace exaDEM
{
	using namespace exanb;

	template<typename GridT , class = AssertGridHasFields< GridT, field::_radius >>
		class ComputeHookeClassifyInteraction : public OperatorNode
	{
		// attributes processed during computation
		using ComputeFields = FieldSet< field::_vrot, field::_arot >;
		static constexpr ComputeFields compute_field_set {};

		ADD_SLOT( GridT       , grid              , INPUT_OUTPUT , REQUIRED );
		ADD_SLOT( shapes      , shapes_collection , INPUT_OUTPUT , DocString{"Collection of shapes"});
		ADD_SLOT( HookeParams , config            , INPUT , REQUIRED ); // can be re-used for to dump contact network
		ADD_SLOT( HookeParams , config_driver     , INPUT , OPTIONAL ); // can be re-used for to dump contact network
		ADD_SLOT( mutexes     , locks             , INPUT_OUTPUT );
		ADD_SLOT( double      , dt                , INPUT , REQUIRED );
		ADD_SLOT( Drivers     , drivers           , INPUT , DocString{"List of Drivers"});
		ADD_SLOT( Classifier  , ic                , INPUT_OUTPUT , DocString{"Interaction lists classified according to their types"} );


		public:

		inline std::string documentation() const override final
		{
			return R"EOF(
                )EOF";
		}



		template<typename... Args>
			inline void run_hooke_law(int id, Classifier& ic, Args&&... args)
			{
				auto [ptr, size] = ic.get_info(id);
				apply_hooke_law(ptr, size, std::forward<Args>(args)...);
			}

		template<typename TMPLC, typename TMPLK, typename... Args>
			inline void apply_hooke_law(exaDEM::Interaction* data_ptr, size_t data_size, TMPLK& kernel, TMPLC* cells, Args&&... args)
			{
#pragma omp parallel for schedule(static)
				for(size_t it = 0 ; it < data_size ; it++)
				{
					exaDEM::Interaction& item = data_ptr[it];
					kernel(item, cells, std::forward<Args>(args)...);
				}
			}


		inline void execute () override final
		{
			if( grid->number_of_cells() == 0 ) { return; }

			Drivers empty;
			Drivers& drvs =  drivers.has_value() ? *drivers : empty;

			const auto cells = grid->cells();
			auto & shps = *shapes_collection;
			const HookeParams hkp = *config;
			HookeParams hkp_drvs;

			if ( drivers->get_size() > 0 &&  config_driver.has_value() )
			{
				hkp_drvs = *config_driver;
			}

			const double time = *dt;
			mutexes& locker = *locks;
			auto& classifier = *ic;

			hooke_law_polyhedra poly;
			hooke_law_driver<Cylinder> cyli;
			hooke_law_driver<Surface>  surf;
			hooke_law_driver<Ball>     ball;
			hooke_law_stl stlm = {};

			for(int w = 0 ; w <= 3 ; w++)
			{
				run_hooke_law(w, classifier, poly, cells, hkp, shps, time, locker);	
			}
			run_hooke_law(4, classifier, cyli, cells, drvs, hkp_drvs, shps, time, locker);	
			run_hooke_law(5, classifier, surf, cells, drvs, hkp_drvs, shps, time, locker);	
			run_hooke_law(6, classifier, ball, cells, drvs, hkp_drvs, shps, time, locker);	
			for(int w = 7 ; w <= 12 ; w++)
			{
				run_hooke_law(w, classifier, stlm, cells, drvs, hkp_drvs, shps, time, locker);	
			}
		}
	};

	template<class GridT> using ComputeHookeClassifyInteractionTmpl = ComputeHookeClassifyInteraction<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "compute_hooke_classify_interaction", make_grid_variant_operator< ComputeHookeClassifyInteractionTmpl > );
	}
}

