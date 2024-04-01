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
#include <exanb/core/basic_types.h>
#include <exaDEM/common_compute_kernels.h>
#include <exaDEM/compute_hooke_force.h>

using exanb::Vec3d; 

namespace exaDEM
{
	struct RigidSurfaceFunctor
	{
		Vec3d m_normal;
		double m_offset;
		double m_vel;
		double m_dt;
		double m_kt;
		double m_kn;
		double m_kr;
		double m_mu;
		double m_dampRate;

		void test() {std::cout << "test" << std::endl;}
		ONIKA_HOST_DEVICE_FUNC inline void operator () (
				const double a_rx, const double a_ry, const double a_rz,
				const double a_vx, const double a_vy, const double a_vz,
				Vec3d& a_vrot, 
				double a_particle_radius,
				double& a_fx, double& a_fy, double& a_fz, 
				const double a_mass,
				Vec3d& a_mom,
				Vec3d& a_ft) const
		{
			[[maybe_unused]] double res = run  ( a_rx, a_ry, a_rz, a_vx, a_vy, a_vz, a_vrot, a_particle_radius, a_fx, a_fy, a_fz, a_mass, a_mom, a_ft);
		}

		ONIKA_HOST_DEVICE_FUNC inline double run (
				const double a_rx, const double a_ry, const double a_rz,
				const double a_vx, const double a_vy, const double a_vz,
				Vec3d& a_vrot, 
				double a_particle_radius,
				double& a_fx, double& a_fy, double& a_fz, 
				const double a_mass,
				Vec3d& a_mom,
				Vec3d& a_ft) const
		{
			Vec3d pos = {a_rx,a_ry,a_rz};
			Vec3d vel = {a_vx,a_vy,a_vz};

			const Vec3d pos_proj = dot(pos, m_normal) * m_normal;
			const Vec3d rigid_surface_center = m_offset * m_normal; 

			Vec3d vec_n = pos_proj - rigid_surface_center;
			double n = norm(vec_n);
			const double dn = n - a_particle_radius;
			if (dn < 0.0)
			{
			  vec_n = vec_n / n;
				const Vec3d contact_position = pos_proj - vec_n * ( a_particle_radius + 0.5 * dn ) ; 
				const Vec3d rigid_surface_velocity = m_normal * m_vel; 
				constexpr Vec3d rigid_surface_angular_velocity = {0.0,0.0,0.0};

				Vec3d f = {0.0,0.0,0.0};
				const double meff = a_mass;

				/*
					 exaDEM::hooke_force_core_v2(
					 dn, vec_n,
					 m_dt, m_kn, m_kt, m_kr, m_mu, m_dampRate, meff,
					 a_ft, contact_position, pos_proj, vel, f, a_mom, a_vrot,
					 rigid_surface_center, rigid_surface_velocity, rigid_surface_angular_velocity
					 );
				 */
				exaDEM::hooke_force_core(
						dn, vec_n,
						m_dt, m_kn, m_kt, m_kr, m_mu, m_dampRate, meff,
						a_ft, contact_position, pos_proj, vel, f, a_mom, a_vrot,
						rigid_surface_center, rigid_surface_velocity, rigid_surface_angular_velocity
						);
				// compute forces (norm)
				const double res = (-1) * exanb::dot(Vec3d{f.x-a_ft.x,f.y-a_ft.y,f.z-a_ft.z}, m_normal);

				// === update forces
				a_fx += f.x ;
				a_fy += f.y ;
				a_fz += f.z ;
				return res;
			}
			else return 0.0;
		}
	};
}


namespace exanb
{
	template<> struct ComputeCellParticlesTraits<exaDEM::RigidSurfaceFunctor>
	{
		static inline constexpr bool RequiresBlockSynchronousCall = false;
		static inline constexpr bool CudaCompatible = true;
	};
}
