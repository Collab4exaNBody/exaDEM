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

namespace exaDEM
{
	ONIKA_HOST_DEVICE_FUNC inline 
	void reset(Vec3d& in)
	{
		in = Vec3d{0.0,0.0,0.0};
	}
 
	ONIKA_HOST_DEVICE_FUNC inline 
	void cohesive_force_core(
			const double dn,
			const Vec3d& n,
			const double dncut,
			const double fc,
			Vec3d& f
			)
	{
		if(dncut == 0) return;

		if(dn <= dncut)
		{
			const double fn_value = (fc/dncut) * dn - fc;
			const Vec3d fn = fn_value * n; 

			// === update forces
			f.x += fn.x;  
			f.y += fn.y; 
			f.z += fn.z;
		}
	}

	ONIKA_HOST_DEVICE_FUNC inline 
	void hooke_force_core_v2(
			const double dn,
			const Vec3d& n, // normal
			const double dt,
			const double kn,
			const double kt,
			const double kr,
			const double mu,
			const double dampRate,
			const double meff,
			Vec3d& ft, // tangential force between particle i and j
			const Vec3d& contact_position,
			const Vec3d& pos_i, // positions i
			const Vec3d& vel_i, // positions i
			Vec3d& f_i, // forces i
			Vec3d & mom_i, // moments i
			const Vec3d& vrot_i, // angular velocities i
			const Vec3d& pos_j, // positions j
			const Vec3d& vel_j, // positions j
			const Vec3d& vrot_j // angular velocities j
			)
			{
				if(dn <= 0.0) 
				{
					const double damp = compute_damp(dampRate, kn,  meff);

					// === Relative velocity (j relative to i)
					auto vel = compute_relative_velocity(
							contact_position,
							pos_i, vel_i, vrot_i,
							pos_j, vel_j, vrot_j
							);

					// compute relative velocity
					const double vn = exanb::dot(n, vel);

					// === Normal force (elatic contact + viscous damping)
					const Vec3d fn = compute_normal_force(kn, damp, dn, vn, n); // fc ==> cohesive force

					// === Tangential force (friction)
					ft	 		+= exaDEM::compute_tangential_force(kt, dt, vn, n, vel);


					// fit tangential force
					auto threshold_ft 	= exaDEM::compute_threshold_ft(mu, kn, dn);
					exaDEM::fit_tangential_force(threshold_ft, ft);

					// === sum forces
					const auto f = fn + ft;

					// === update forces
					//f_i += f;
					f_i += f;

					// === update moments
					const Vec3d mom = kr * (vrot_j - vrot_i) * dt;
					const auto Ci = (contact_position - pos_i);
					const auto Pimoment = exanb::cross(Ci, f) + mom;
					mom_i += Pimoment;
				}
				else
				{
					reset(ft); // no friction if no contact
				}
			}


ONIKA_HOST_DEVICE_FUNC inline 
	void hooke_force_core(
			const double dn,
			const Vec3d& n, // -normal
			const double dt,
			const double kn,
			const double kt,
			const double kr,
			const double mu,
			const double dampRate,
			const double meff,
			Vec3d& ft, // tangential force between particle i and j
			const Vec3d& contact_position,
			const Vec3d& pos_i, // positions i
			const Vec3d& vel_i, // positions i
			Vec3d& f_i, // forces i
			Vec3d& mom_i, // moments i
			const Vec3d& vrot_i, // angular velocities i
			const Vec3d& pos_j, // positions j
			const Vec3d& vel_j, // positions j
			const Vec3d& vrot_j // angular velocities j
			)
			{
				const double damp = compute_damp(dampRate, kn,  meff);


				// === Relative velocity (j relative to i)
				auto vel = compute_relative_velocity(
						contact_position,
						pos_i, vel_i, vrot_i,
						pos_j, vel_j, vrot_j
						);

				// compute relative velocity
				const double vn = exanb::dot(vel, n);

				// === Normal force (elatic contact + viscous damping)
				const Vec3d fn = compute_normal_force(kn, damp, dn, vn, n); // fc ==> cohesive force

				// === Tangential force (friction)
				ft	 	+= exaDEM::compute_tangential_force(kt, dt, vn, n, vel);
				//ft	 	+= exaDEM::compute_tangential_force(kt, dt, vn, n, vel);


				// fit tangential force
				auto threshold_ft 	= exaDEM::compute_threshold_ft(mu, kn, dn);
				exaDEM::fit_tangential_force(threshold_ft, ft);

				// === sum forces
				f_i = fn + ft;

				// === update moments
				mom_i += kr * (vrot_j - vrot_i) * dt;

///*
				// test
				Vec3d branch = contact_position - pos_i;
				double r = ( exanb::dot(branch , vrot_i)) / ( exanb::dot(vrot_i, vrot_i));
				branch -= r * vrot_i;

				constexpr double mur = 0;
				double threshold_mom = std::abs(mur * exanb::norm(branch) * exanb::norm(fn));  // even without fabs, the value should
																																											 // be positive
				double mom_square = exanb::dot(mom_i, mom_i);
				if (mom_square > 0.0 && mom_square > threshold_mom * threshold_mom) mom_i = mom_i * (threshold_mom / sqrt(mom_square));
//*/
			}

ONIKA_HOST_DEVICE_FUNC inline 
	Vec3d compute_moments(
			const Vec3d& contact_position,
			const Vec3d& p, // position
			const Vec3d& f, // forces
			const Vec3d& m) // I.mom
	{
		const auto Ci = (contact_position - p);
		const auto Pimoment = exanb::cross(Ci, f) + m;
		return Pimoment;
	}


ONIKA_HOST_DEVICE_FUNC inline 
	void compute_hooke_force(
			const double dncut,
			const double dt,
			const double kn,
			const double kt,
			const double kr,
			const double fc,
			const double mu,
			const double dampRate,
			Vec3d& ft, // tangential force between particle i and j
			double rx_i, double ry_i, double rz_i, // positions i
			double vx_i, double vy_i, double vz_i, // positions i
			double mass_i, 	// mass i
			double r_i, 	// raduis i
			double& fx_i, double& fy_i, double& fz_i, // forces i
			Vec3d & mom_i, // moments i
			const Vec3d& vrot_i, // angular velocities i
			double rx_j, double ry_j, double rz_j, // positions j
			double vx_j, double vy_j, double vz_j, // positions j
			double mass_j, 	// mass j
			double r_j, 	// raduis j
			double& fx_j, double& fy_j, double& fz_j, // forces j
			Vec3d& mom_j, // moments j
			const Vec3d& vrot_j // angular velocities j
				)
				{
					// === normal n from j to i
					const double dist_x = rx_i - rx_j;
					const double dist_y = ry_i - ry_j;
					const double dist_z = rz_i - rz_j;

					// === compute norm
					const double dist_norm =  sqrt( dist_x*dist_x + dist_y*dist_y + dist_z*dist_z );

					// === inv norm
					const double inv_dist_norm = 1.0 / dist_norm;

					// === normal vector
					const double n_x = dist_x * inv_dist_norm;
					const double n_y = dist_y * inv_dist_norm;
					const double n_z = dist_z * inv_dist_norm;

					// === compute r_i + r_j 
					const double R = r_i + r_j;

					// === compute overlap in dn
					const double dn = dist_norm - R;

					if(dn <= 0.0) 
					{
						const double meff = compute_effective_mass(mass_i, mass_j);
						const double damp = compute_damp(dampRate, kn,  meff);

						// === contact position
						const double posx = rx_i - n_x *(r_i + 0.5*dn); // dn < 0
						const double posy = ry_i - n_y *(r_i + 0.5*dn);
						const double posz = rz_i - n_z *(r_i + 0.5*dn);

						// === using vec3d
						const Vec3d vel_i = {vx_i, vy_i, vz_i};
						const Vec3d pos_i = {rx_i, ry_i, rz_i};

						const Vec3d vel_j = {vx_j, vy_j, vz_j};
						const Vec3d pos_j = {rx_j, ry_j, rz_j};

						const Vec3d contact_position = {posx,posy,posz};

						// === Relative velocity (j relative to i)
						auto vel = compute_relative_velocity(
								contact_position,
								pos_i, vel_i, vrot_i,
								pos_j, vel_j, vrot_j
								);

						// compute relative velocity
						const double vn = exanb::dot(Vec3d{n_x, n_y, n_z}, vel);

						const auto n = Vec3d{n_x,n_y,n_z};

						// === Normal force (elatic contact + viscous damping)
						const double fn_value = compute_normal_force_value_with_cohesive_force(kn, fc, damp, dn, vn); // fc ==> cohesive force
						const Vec3d fn = fn_value * n; 

						// === Tangential force (friction)
						//Vec3d ft 		= exaDEM::compute_tangential_force(kt, dt, vn, n, vel);
						ft	 		+= exaDEM::compute_tangential_force(kt, dt, vn, n, vel);


						// fit tangential force
						//auto threshold_ft 	= exaDEM::compute_threshold_ft(mu, kn, dn);
						auto threshold_ft 	= exaDEM::compute_threshold_ft_with_cohesive_force(mu, fn_value, fc);
						exaDEM::fit_tangential_force(threshold_ft, ft);

						// === sum forces
						const auto f = fn + ft;

						// === update forces
						fx_i += f.x;  
						fy_i += f.y; 
						fz_i += f.z;

						// === update moments
						const Vec3d mom = kr * (vrot_j - vrot_i) * dt;
						const auto Ci = (contact_position - pos_i);
						const auto Pimoment = exanb::cross(Ci, f) + mom;
						mom_i += Pimoment;
					}
					else if(dncut == 0.0)
					{
						// do nothing
						ft = {0,0,0}; // no friction if no contact
					}
					else if(dn <= dncut)
					{
						const auto n = Vec3d{n_x,n_y,n_z};
						const double fn_value = (fc/dncut) * dn - fc;
						const Vec3d fn = fn_value * n; 

						// === update forces
						fx_i += fn.x;  
						fy_i += fn.y; 
						fz_i += fn.z;
						ft = {0,0,0}; // no friction if no contact
					}
					else
					{
						ft = {0,0,0}; // no friction if no contact
					}

				}
}

