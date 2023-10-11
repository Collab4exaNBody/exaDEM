#pragma once
#include <exanb/core/basic_types.h>
#include <exaDEM/common_compute_kernels.h>
#include <exaDEM/compute_hooke_force.h>
#include <exaDEM/face.h>
#include <exaDEM/stl_mesh.h>

namespace exaDEM
{
	using exanb::Vec3d; 

	/**
	 * @brief Functor for applying Hooke's law to multiple Faces within a grid.
	 *
	 * The `ApplyHookeSTLMeshesFunctor` struct represents a functor used to apply Hooke's law to particles interacting
	 * with multiple STL meshes within a grid. It is designed to be used as an operator in simulations. The functor takes
	 * various parameters and updates forces and torques acting on the particles.
	 *
	 * @tparam GridT The type of grid.
	 */
	struct ApplyHookeSTLMeshesFunctor
	{
		std::vector<exaDEM::stl_mesh>& meshes; /**< A collection of STL meshes. */
		double m_dt; /**< Time step. */
		double m_kt; /**< Tangential spring constant. */
		double m_kn; /**< Normal spring constant. */
		double m_kr; /**< Rotational spring constant. */
		double m_mu; /**< Friction coefficient. */
		double m_dampRate; /**< Damping rate. */

		/**
		 * @brief Operator for applying Hooke's law.
		 *
		 * This operator applies Hooke's law to particles interacting with multiple STL meshes within a grid. It updates
		 * forces and torques acting on the particles based on the specified parameters and properties.
		 *
		 * @param cell_idx The cell index.
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
		 */
		ONIKA_HOST_DEVICE_FUNC inline void operator () (
				const size_t cell_idx,
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
				bool is_face = false; // If there is one contact with a face, we skip contact with edges
				bool do_edge = false;
				auto& idxs = mesh.indexes[cell_idx];
				int fsize = idxs.size();
				auto& faces = mesh.m_data;
				// test face / sphere
				for(int face_idx = 0 ; face_idx < fsize ; face_idx++)
				{
					auto& face = faces[idxs[face_idx]];
					auto [contact, potential, position] = face.contact_face_sphere(a_rx, a_ry, a_rz, a_particle_radius);
					if(contact)
					{
						constexpr int type = 0;
						this->operator()(face, position, type, a_rx, a_ry, a_rz, 
								a_vx, a_vy, a_vz, 
								a_vrot, a_particle_radius, 
								a_fx, a_fy, a_fz, 
								a_mass, a_mom, a_ft);
						is_face = true;
					}
					do_edge = do_edge || potential;
				}

				// contact edge / sphere
				if(is_face == false && do_edge)
				{
					for(int face_idx = 0 ; face_idx < fsize ; face_idx++)
					{
						auto& face = faces[idxs[face_idx]];
						auto [contact, position] = face.contact_edge_sphere(a_rx, a_ry, a_rz, a_particle_radius);
						if(contact)
						{
							constexpr int type = 1;
							this->operator()(face, position, type, a_rx, a_ry, a_rz, 
									a_vx, a_vy, a_vz, 
									a_vrot, a_particle_radius, 
									a_fx, a_fy, a_fz, 
									a_mass, a_mom, a_ft);
						}
					}
				}
			}
		}

		/**
		 * @brief Operator for handling interactions between faces/edges and spheres.
		 *
		 * The operator function handles interactions between faces/edges and spheres. It calculates forces and torques
		 * based on the specified parameters and properties, including the type of interaction (face/sphere or edge/sphere).
		 *
		 * @param face The face to interact with.
		 * @param contact_position The position of contact.
		 * @param type The type of interaction (0 for face/sphere, 1 for edge/sphere).
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
		 */
		ONIKA_HOST_DEVICE_FUNC inline void operator () (
				const exaDEM::Face& face, const Vec3d& contact_position, const int type,
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
	};
}


namespace exanb
{
	template<> struct ComputeCellParticlesTraits<exaDEM::ApplyHookeSTLMeshesFunctor>
	{
		static inline constexpr bool RequiresBlockSynchronousCall = false;
		static inline constexpr bool CudaCompatible = false;
	};

	template<> struct ComputeCellParticlesTraitsUseCellIdx<exaDEM::ApplyHookeSTLMeshesFunctor>
	{
		static inline constexpr bool UseCellIdx = true;
	};
}
