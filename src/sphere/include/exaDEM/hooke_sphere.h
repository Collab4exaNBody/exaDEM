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
#pragma once

#include <exaDEM/shape/shape.hpp>
#include <exaDEM/shape/shape_detection.hpp>
#include <exaDEM/interaction/interaction.hpp>

namespace exaDEM
{
	ONIKA_HOST_DEVICE_FUNC
	inline void lockAndAdd(double& val, double add)
		{
			ONIKA_CU_ATOMIC_ADD(val, add);
		}

	ONIKA_HOST_DEVICE_FUNC
	inline void lockAndAdd(Vec3d& val, Vec3d&& add)
		{
			ONIKA_CU_ATOMIC_ADD(val.x, add.x);
			ONIKA_CU_ATOMIC_ADD(val.y, add.y);
			ONIKA_CU_ATOMIC_ADD(val.z, add.z);
		}

	namespace sphere
	{
		using namespace exanb;
		template<bool sym>
			struct hooke_law
			{
				template<typename Cell>
					ONIKA_HOST_DEVICE_FUNC inline const Vec3d get_r(Cell& cell, const int p_id) const
					{
						const Vec3d res = {
							cell[field::rx][p_id],
							cell[field::ry][p_id],
							cell[field::rz][p_id]};
						return res;
					};

				template<typename Cell>
					ONIKA_HOST_DEVICE_FUNC inline const Vec3d get_v(Cell& cell, const int p_id) const
					{
						const Vec3d res = {
							cell[field::vx][p_id],
							cell[field::vy][p_id],
							cell[field::vz][p_id]};
						return res;
					};

				template<typename Cells>
					ONIKA_HOST_DEVICE_FUNC inline void operator()(
							Interaction& item, 
							Cells& cells, 
							const HookeParams& hkp, 
							const double time, 
							mutexes& locker) const
					{
						// === cell
						auto& cell_i =  cells[item.cell_i];
						auto& cell_j =  cells[item.cell_j];

						// === positions
						const Vec3d ri = get_r(cell_i, item.p_i);
						const Vec3d rj = get_r(cell_j, item.p_j);

						// === positions
						const double rad_i = cell_i[field::radius][item.p_i];
						const double rad_j = cell_j[field::radius][item.p_j];

						// === vrot
						const Vec3d& vrot_i = cell_i[field::vrot][item.p_i];
						const Vec3d& vrot_j = cell_j[field::vrot][item.p_j];

						auto [contact, dn, n, contact_position] = detection_vertex_vertex_core(ri, rad_i, rj, rad_j); 

						if(contact)
						{
							const Vec3d vi = get_v(cell_i, item.p_i);
							const Vec3d vj = get_v(cell_j, item.p_j);
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

							if constexpr (sym)
							{
								// ==== Particle j
								locker.lock(item.cell_j, item.p_j);

								auto& mom_j = cell_j[field::mom][item.p_j];
								mom_j += compute_moments(contact_position, rj, -f, -item.moment);
								cell_j[field::fx][item.p_j] -= f.x;
								cell_j[field::fy][item.p_j] -= f.y;
								cell_j[field::fz][item.p_j] -= f.z;

								locker.unlock(item.cell_j, item.p_j);
							}
						}
						else
						{
							item.reset();
						}
					}

				template<typename Cells>
					ONIKA_HOST_DEVICE_FUNC inline void operator()(
							Interaction& item, 
							Cells* cells, 
							const HookeParams& hkp, 
							const double time) const
					{
						// === cell
						auto& cell_i =  cells[item.cell_i];
						auto& cell_j =  cells[item.cell_j];

						// === positions
						const Vec3d ri = get_r(cell_i, item.p_i);
						const Vec3d rj = get_r(cell_j, item.p_j);

						// === positions
						const double rad_i = cell_i[field::radius][item.p_i];
						const double rad_j = cell_j[field::radius][item.p_j];

						// === vrot
						const Vec3d& vrot_i = cell_i[field::vrot][item.p_i];
						const Vec3d& vrot_j = cell_j[field::vrot][item.p_j];

						auto [contact, dn, n, contact_position] = detection_vertex_vertex_core(ri, rad_i, rj, rad_j); 

						if(contact)
						{
							const Vec3d vi = get_v(cell_i, item.p_i);
							const Vec3d vj = get_v(cell_j, item.p_j);
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
							auto& mom_i = cell_i[field::mom][item.p_i];
							lockAndAdd(mom_i, compute_moments(contact_position, ri, f, item.moment));
							lockAndAdd(cell_i[field::fx][item.p_i], f.x);
							lockAndAdd(cell_i[field::fy][item.p_i], f.y);
							lockAndAdd(cell_i[field::fz][item.p_i], f.z);

							if constexpr (sym)
							{
								// ==== Particle j
								auto& mom_j = cell_j[field::mom][item.p_j];
								lockAndAdd(mom_j, compute_moments(contact_position, rj, -f, -item.moment));
								lockAndAdd(cell_j[field::fx][item.p_j], -f.x);
								lockAndAdd(cell_j[field::fy][item.p_j], -f.y);
								lockAndAdd(cell_j[field::fz][item.p_j], -f.z);
							}
						}
						else
						{
							item.reset();
						}
					}
			};

		// C for cell and D for driver
		template<typename TMPLD>
			struct hooke_law_driver
			{
				template<typename Cells>
					ONIKA_HOST_DEVICE_FUNC inline void operator()(Interaction& item, Cells& cells, Drivers& drvs, const HookeParams& hkp, const double time, mutexes& locker) const
					{
						const int driver_idx = item.id_j; //
						TMPLD& driver = std::get<TMPLD>(drvs.data(driver_idx)) ;
						auto& cell = cells[item.cell_i];
						const size_t p   = item.p_i;
						// === positions
						const Vec3d r       = { cell[field::rx][p], cell[field::ry][p], cell[field::rz][p] };
						const double rad    = cell[field::radius][p];
						// === vertex array

						auto [contact, dn, n, contact_position] = exaDEM::detector_vertex_driver(driver, r, rad);

						if(contact)
						{
							// === vrot
							const Vec3d& vrot  = cell[field::vrot][p];

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

				template<typename Cells>
					ONIKA_HOST_DEVICE_FUNC inline void operator()(Interaction& item, Cells* cells, TMPLD* drvs, const HookeParams& hkp, const double time) const
					{
						const int driver_idx = item.id_j; //
						TMPLD& driver = drvs[driver_idx];
						auto& cell = cells[item.cell_i];
						const size_t p   = item.p_i;
						// === positions
						const Vec3d r       = { cell[field::rx][p], cell[field::ry][p], cell[field::rz][p] };
						const double rad    = cell[field::radius][p];
						// === vertex array

						auto [contact, dn, n, contact_position] = exaDEM::detector_vertex_driver(driver, r, rad);

						if(contact)
						{
							// === vrot
							const Vec3d& vrot  = cell[field::vrot][p];

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
							lockAndAdd(mom, compute_moments(contact_position, r, f, item.moment));
							//cell[field::fx][p] += f.x;
							//cell[field::fy][p] += f.y;
							//cell[field::fz][p] += f.z;
							//mom += compute_moments(contact_position, r, f, item.moment);
							lockAndAdd(cell[field::fx][p], f.x);
							lockAndAdd(cell[field::fy][p], f.y);
							lockAndAdd(cell[field::fz][p], f.z);
						}
						else
						{
							item.reset();
						}
					}
			};


		struct stl_mesh_detector
		{
			ONIKA_HOST_DEVICE_FUNC inline std::tuple<bool, double, Vec3d, Vec3d> operator() (
					const uint16_t type,
					const Vec3d& pi, const double radius,
					const Vec3d& pj, const int j, const shape* shpj, const exanb::Quaternion& oj) const
			{
#define __params__     pi, radius, pj, j, shpj, oj
				assert( type >= 7 && type <= 12 );
				switch (type)
				{
					case 7: return exaDEM::detection_vertex_vertex ( __params__ );
					case 8: return exaDEM::detection_vertex_edge ( __params__ );
					case 9: return exaDEM::detection_vertex_face ( __params__ );
				}
#undef __params__
				return std::tuple<bool, double, Vec3d, Vec3d>();
			}

		};

		struct hooke_law_stl
		{
			template<typename Cells>
				ONIKA_HOST_DEVICE_FUNC inline void operator()( Interaction& item, Cells& cells, Drivers& drvs, const HookeParams& hkp, const double time, mutexes& locker) const
				{
					const int driver_idx = item.id_j; //
					auto& driver = std::get<Stl_mesh>(drvs.data(driver_idx)) ;
					auto& cell = cells[item.cell_i];

					const size_t p_i   = item.p_i;
					const size_t sub_j = item.sub_j;

					// === positions
					const Vec3d r_i       = { cell[field::rx][p_i], cell[field::ry][p_i], cell[field::rz][p_i] };
					// === vrot
					const Vec3d& vrot_i  = cell[field::vrot][p_i];
					const double radius_i  = cell[field::radius][p_i];
					const auto& shp_j = driver.shp;

					const Quaternion orient_j = {1.0,0.0,0.0,0.0};
					auto [contact, dn, n, contact_position] = func(item.type, r_i, radius_i, driver.center, sub_j, &shp_j, orient_j);

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

			template<typename Cells>
				ONIKA_HOST_DEVICE_FUNC inline void operator()( 
						Interaction& item, 
						Cells* cells, 
						Drivers* drvs, 
						const HookeParams& hkp, 
						const double time) const
				{
					const int driver_idx = item.id_j; //
					auto& driver = std::get<Stl_mesh>(drvs->data(driver_idx)) ;
					auto& cell = cells[item.cell_i];

					const size_t p_i   = item.p_i;
					const size_t sub_j = item.sub_j;

					// === particle i
					const Vec3d r_i       = { cell[field::rx][p_i], cell[field::ry][p_i], cell[field::rz][p_i] };
					const Vec3d& vrot_i   = cell[field::vrot][p_i];
					const double radius_i = cell[field::radius][p_i];
          // === driver j
					const auto& shp_j         = driver.shp;
					const Quaternion orient_j = {1.0,0.0,0.0,0.0};
					auto [contact, dn, n, contact_position] = func(item.type, r_i, radius_i, driver.center, sub_j, &shp_j, orient_j);

					if(contact)
					{
						auto& mom         = cell[field::mom][p_i];
						const Vec3d v_i   = { cell[field::vx][p_i], cell[field::vy][p_i], cell[field::vz][p_i] };
						const double meff = cell[field::mass][p_i];
						Vec3d f           = {0,0,0};
						hooke_force_core(dn, n, time, hkp.m_kn, hkp.m_kt, hkp.m_kr,
								hkp.m_mu, hkp.m_damp_rate, meff,
								item.friction, contact_position,
								r_i, v_i, f, item.moment, vrot_i,  // particle i
								driver.center, driver.vel, driver.vrot // particle j
								);

						// === update informations
						lockAndAdd(mom, compute_moments(contact_position, r_i, f, item.moment));
						lockAndAdd(cell[field::fx][p_i], f.x);
						lockAndAdd(cell[field::fy][p_i], f.y);
						lockAndAdd(cell[field::fz][p_i], f.z);
					}
					else
					{
						item.reset();
					}
				}
			const stl_mesh_detector func;
		};
	}
}
