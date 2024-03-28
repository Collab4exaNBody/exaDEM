#pragma once

#include <exaDEM/shape/shape.hpp>
#include <exaDEM/shape/shape_detection.hpp>
#include <exaDEM/interaction/interaction.hpp>

namespace exaDEM
{

		// C for cell and D for driver
		template<typename C, typename D>
			ONIKA_HOST_DEVICE_FUNC void compute_driver(C& cell, D& driver, const HookeParams& hkp, const shape* shp, Interaction& I, const double time, mutexes& locker)
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

		struct stl_mesh_dectector
		{
			ONIKA_HOST_DEVICE_FUNC std::tuple<bool, double, Vec3d, Vec3d> operator() (
					const uint16_t type,
					const Vec3d& pi, const int i, const shape* shpi, const exanb::Quaternion& oi,
					const Vec3d& pj, const int j, const shape* shpj, const exanb::Quaternion& oj) const
			{
			 	#define __params__     pi, i, shpi, oi, pj, j, shpj, oj
			 	#define __inv_params__ pj, j, shpj, oj, pi, i, shpi, oi
				assert( type >= 7 && type <= 12 );
				switch (type)
				{
					case 7: return exaDEM::detection_vertex_vertex ( __params__ );
					case 8: return exaDEM::detection_vertex_edge ( __params__ );
					case 9: return exaDEM::detection_vertex_face ( __params__ );
					case 10: return exaDEM::detection_edge_edge ( __params__ );
					case 11: return exaDEM::detection_vertex_edge ( __params__ );
					case 12: return exaDEM::detection_vertex_face ( __inv_params__ );
				}
				#undef __params__
				#undef __inv_params__
				return std::tuple<bool, double, Vec3d, Vec3d>();
			}

		};

		template<typename C>
			ONIKA_HOST_DEVICE_FUNC void compute_driver_stl(C& cell, Stl_mesh& driver, const HookeParams& hkp, const shape* shp_i, Interaction& I, const double time, mutexes& locker)
			{
				const stl_mesh_dectector func;
				const size_t p_i   = I.p_i;
				const size_t sub_i = I.sub_i;
				const size_t sub_j = I.sub_j;
				// === positions
				const Vec3d r_i       = { cell[field::rx][p_i], cell[field::ry][p_i], cell[field::rz][p_i] };
				// === vrot
				const Vec3d& vrot_i  = cell[field::vrot][p_i];
				const Quaternion& orient_i  = cell[field::orient][p_i];
				const auto& shp_j = driver.shp;

				// WARNING
				const Quaternion orient_j = {1.0,0.0,0.0,0.0};
				auto [contact, dn, n, contact_position] = func(I.type, r_i, sub_i, shp_i, orient_i, driver.center, sub_j, &shp_j, orient_j);

				if(contact)
				{
					constexpr Vec3d null = {0,0,0};
					auto& mom = cell[field::mom][p_i];
					const Vec3d v_i = { cell[field::vx][p_i], cell[field::vy][p_i], cell[field::vz][p_i] };
					const double meff = cell[field::mass][p_i];
					Vec3d f = null;
					hooke_force_core(dn, n, time, hkp.m_kn, hkp.m_kt, hkp.m_kr,
							hkp.m_mu, hkp.m_damp_rate, meff,
							I.friction, contact_position,
							r_i, v_i, f, I.moment, vrot_i,  // particle i
							driver.center, driver.vel, driver.vrot // particle j
							);

					// === update informations
					locker.lock(I.cell_i, p_i);
					mom += compute_moments(contact_position, r_i, f, I.moment);
					cell[field::fx][p_i] += f.x;
					cell[field::fy][p_i] += f.y;
					cell[field::fz][p_i] += f.z;
					locker.unlock(I.cell_i, p_i);
				}
				else
				{
					I.reset();
				}
			}



		template<class GridT>
			struct HookeForceInteractionFunctor
			{
				GridT& cells;
				mutexes& locker;
				const shapes & shps;
				const HookeParams params;
				const double time;

				inline const Vec3d get_r(const int cell_id, const int p_id)
				{
					const Vec3d res = {
						cells[cell_id][field::rx][p_id],
						cells[cell_id][field::ry][p_id],
						cells[cell_id][field::rz][p_id]};
					return res;
				};

				inline const Vec3d get_v(const int cell_id, const int p_id)
				{
					const Vec3d res = {
						cells[cell_id][field::vx][p_id],
						cells[cell_id][field::vy][p_id],
						cells[cell_id][field::vz][p_id]};
					return res;
				};

				ONIKA_HOST_DEVICE_FUNC inline void compute_force_interaction (  )
				{
				}

				ONIKA_HOST_DEVICE_FUNC inline void operator () ( Interaction * __restrict__ ptr, const size_t size) const
				{
				}

				ONIKA_HOST_DEVICE_FUNC inline void operator () ( Interaction& it) const
				{
				}
			};

}
