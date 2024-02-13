#pragma once
#include <exanb/core/basic_types.h>
#include <exaDEM/common_compute_kernels.h>
#include <exaDEM/compute_hooke_force.h>
#include <exaDEM/face.h>

using exanb::Vec3d; 

/**
 * @brief Functor for calculating forces and torques on a particle interacting with a face.
 *
 * The `HookeFaceFunctor` struct represents a functor used to calculate forces and torques on a particle
 * interacting with a face. It is designed to be used as an operator in simulations. The functor takes various
 * parameters and updates force and torque components acting on the particle.
 */
struct HookeFaceFunctor
{
	exaDEM::Face& face; /**< Reference to the face object. */
	double m_dt; /**< Time step. */
	double m_kt; /**< Tangential spring constant. */
	double m_kn; /**< Normal spring constant. */
	double m_kr; /**< Rotational spring constant. */
	double m_mu; /**< Friction coefficient. */
	double m_dampRate; /**< Damping rate. */


	/**
	 * @brief Operator for calculating forces and torques.
	 *
	 * This operator calculates the forces and torques acting on a particle interacting with a face.
	 *
	 * @param a_rx The x-coordinate of the particle's position.
	 * @param a_ry The y-coordinate of the particle's position.
	 * @param a_rz The z-coordinate of the particle's position.
	 * @param a_vx The x-component of the particle's velocity.
	 * @param a_vy The y-component of the particle's velocity.
	 * @param a_vz The z-component of the particle's velocity.
	 * @param a_vrot The rotational velocity of the particle.
	 * @param a_particle_radius The radius of the particle.
	 * @param a_fx Reference to store the x-component of the calculated force.
	 * @param a_fy Reference to store the y-component of the calculated force.
	 * @param a_fz Reference to store the z-component of the calculated force.
	 * @param a_mass The mass of the particle.
	 * @param a_mom Reference to store the angular momentum.
	 * @param a_ft Reference to store the torque.
	 * @return The result of the operator, representing a calculated value (double).
	 */
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
		//printf("HOOKE FACE\n");
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
