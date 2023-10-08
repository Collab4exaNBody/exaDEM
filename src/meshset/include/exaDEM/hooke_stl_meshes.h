#pragma once
#include <exanb/core/basic_types.h>
#include <exaDEM/common_compute_kernels.h>
#include <exaDEM/compute_hooke_force.h>
#include <exaDEM/face.h>
#include <exaDEM/stl_mesh.h>

using exanb::Vec3d; 

struct ApplyHookeSTLMeshesFunctor
{
	std::vector<exaDEM::stl_mesh>& meshes;
	double m_dt;
	double m_kt;
	double m_kn;
	double m_kr;
	double m_mu;
	double m_dampRate;

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
		for(auto& mesh : meshes)
		{
			auto& faces = mesh.m_data;
			for(auto& face : faces)
			{
				this->operator()(face, a_rx, a_ry, a_rz, 
						a_vx, a_vy, a_vz, 
						a_vrot, a_particle_radius, 
						a_fx, a_fy, a_fz, 
						a_mass, a_mom, a_ft);
			}
		}
	}

	ONIKA_HOST_DEVICE_FUNC inline void operator () (
			const exaDEM::Face& face,
			const double a_rx, const double a_ry, const double a_rz,
			const double a_vx, const double a_vy, const double a_vz,
			Vec3d& a_vrot, 
			double a_particle_radius,
			double& a_fx, double& a_fy, double& a_fz, 
			const double a_mass,
			Vec3d& a_mom,
			Vec3d& a_ft) const
	{
		Vec3d pos_proj;
		auto [is_contact, contact_position, type] = face.intersect_sphere(a_rx, a_ry, a_rz, a_particle_radius);

		if(is_contact)
		{
			double m_vel = 0;
			Vec3d pos = {a_rx,a_ry,a_rz};
			Vec3d vel = {a_vx,a_vy,a_vz};


			if(type == 0)
			{
				pos_proj = dot(pos, face.normal) * face.normal;
				// contact_position = face.offset * face.normal; already define
			}
			else if(type == 1)
			{
				pos_proj = pos;
			}

			Vec3d vec_n = pos_proj - contact_position;
			double n = norm(vec_n);
			vec_n = vec_n / n;
			const double dn = n - a_particle_radius;		
			Vec3d rigid_surface_center = contact_position; 
			const Vec3d rigid_surface_velocity = face.normal * m_vel; 
			constexpr Vec3d rigid_surface_angular_velocity = {0.0,0.0,0.0};

			Vec3d f = {0.0,0.0,0.0};
			constexpr double meff = 1;

			exaDEM::hooke_force_core_v2(
					dn, vec_n,
					m_dt, m_kn, m_kt, m_kr, m_mu, m_dampRate, meff,
					a_ft, contact_position, pos_proj, vel, f, a_mom, a_vrot,
					rigid_surface_center, rigid_surface_velocity, rigid_surface_angular_velocity
					);

			// === update forces
			a_fx += f.x ;
			a_fy += f.y ;
			a_fz += f.z ;
		}
	}
};
