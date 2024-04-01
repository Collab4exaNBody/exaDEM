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
struct CylinderWallFunctor
{
	Vec3d m_center = { 0.0 , 0.0 , 0.0 };
	Vec3d m_axis = { 1.0 , 0.0 , 1.0 };
	Vec3d m_cylinder_angular_velocity = {0.0 , 0.0 , 0.0};
	Vec3d m_cylinder_velocity = {0.0,0.0,0.0}; // TODO : do not touch
	double m_radius;
	double m_dt;
	double m_kt;
	double m_kn;
	double m_kr;
	double m_mu;
	double m_dampRate;

	ONIKA_HOST_DEVICE_FUNC inline void operator () (
			const double a_rx, const double a_ry, const double a_rz,
			const double a_vx, const double a_vy, const double a_vz,
			const Vec3d& a_vrot, 
			double a_particle_radius,
			double& a_fx, double& a_fy, double& a_fz, 
			const double a_mass,
			Vec3d& a_mom,
			Vec3d& a_ft) const
	{
		Vec3d pos 	= Vec3d{a_rx, a_ry, a_rz} * m_axis;
		Vec3d pos_proj 	= pos * m_axis;
		Vec3d vel 	= Vec3d{a_vx, a_vy, a_vz};
		// === direction
		const auto dir 	= pos_proj - (m_center * m_axis);

		// === interpenetration
		const double dn = m_radius - ( norm(dir) + a_particle_radius );
		if(dn > 0.0) 
		{
			a_ft = {0.0,0.0,0.0};
			return;
		}

		// === figure out the contact position 
		const auto dir_norm 	= dir / norm(dir) ; 
		const auto contact 	= m_center * m_axis + dir_norm * m_radius ; // compute contact position between the particle and the cylinder

		// === compute damp
		const double meff = a_mass; // mass cylinder >>>>> mass i
		const double damp = exaDEM::compute_damp(m_dampRate, m_kn, meff);

		// === relative velocity	
		const auto total_vel = exaDEM::compute_relative_velocity(contact, 
				m_center, m_cylinder_velocity	, m_cylinder_angular_velocity,
				pos   	, vel              	, a_vrot);

		//const double vn = exanb::dot(total_vel, dir_norm);
		const double vn = exanb::dot(contact-pos, total_vel);

		// === normal force
		const Vec3d fn 	= exaDEM::compute_normal_force(m_kn , damp, dn, vn, dir_norm);

		// === compute tangential force
		auto ft 		= exaDEM::compute_tangential_force(m_kt, m_dt, vn, dir_norm, total_vel);
		//a_ft 			+= exaDEM::compute_tangential_force(m_kt, m_dt, vn, dir_norm, total_vel);
		auto threshold_ft 	= exaDEM::compute_threshold_ft(m_mu, m_kn, dn);

		exaDEM::fit_tangential_force(threshold_ft, ft);
		//exaDEM::fit_tangential_force(threshold_ft, a_ft);

		const auto f = (-1) *(fn + ft); 
		//const auto f = (-1) *(fn + a_ft); 

		// === update forces
		a_fx += f.x ;
		a_fy += f.y ;
		a_fz += f.z ;

		// === update moments
		const Vec3d tmp 	= 	m_kr * (m_cylinder_angular_velocity - vel) * m_dt;
		const auto Ci 		= 	contact - pos;
		const auto moment_i 	= 	exanb::cross(Ci, f) + tmp;
		a_mom 			+= 	moment_i;

	}

	void useless() {}
};

template<> struct exanb::ComputeCellParticlesTraits<CylinderWallFunctor>
{
	static inline constexpr bool RequiresBlockSynchronousCall = false;
	static inline constexpr bool CudaCompatible = true;
};
