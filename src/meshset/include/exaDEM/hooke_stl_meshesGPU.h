#pragma once
#include <exanb/core/basic_types.h>
#include <exaDEM/common_compute_kernels.h>
#include <exaDEM/compute_hooke_force.h>
#include <exaDEM/face.h>
//#include <exaDEM/stl_mesh.h>
#include <exaDEM/stl_meshGPU.h>
#include <cstdio> 

//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h> 
//#include <thrust/universal_vector.h>

#include <onika/cuda/cuda.h> // mots cles specifiques
#include <onika/memory/allocator.h> // cudaMMVector
#include <onika/cuda/stl_adaptors.h>


namespace exaDEM
{
	using exanb::Vec3d;
	
	using namespace exanb; 
		
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
		
		exaDEM::stl_mesh*  pmeshes;/**< Meshes's data.*/
		long unsigned int smeshes;/** Number of meshes.*/ 
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
		ONIKA_HOST_DEVICE_FUNC 
		inline void operator () (
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
			
			for(size_t i=0; i< smeshes ; i++)
			{
				
				auto& mesh= pmeshes[i];
				
				auto& v_x = mesh.vx_GPU;
				auto* vx= onika::cuda::vector_data(v_x);
				
				auto& v_y= mesh.vy_GPU;
				auto* vy= onika::cuda::vector_data(v_y);
				
				auto& v_z= mesh.vz_GPU;
				auto* vz= onika::cuda::vector_data(v_z);
				
				auto& n_x= mesh.nx_GPU;
				auto* nx= onika::cuda::vector_data(n_x);
				
				auto& n_y= mesh.ny_GPU;
				auto* ny= onika::cuda::vector_data(n_y);
				
				auto& n_z= mesh.nz_GPU;
				auto* nz= onika::cuda::vector_data(n_z);
				
				auto& off= mesh.offsets_GPU;
				auto* offsets= onika::cuda::vector_data(off);
				
				auto& off_face= mesh.offs_faces_GPU;
				auto* of= onika::cuda::vector_data(off_face);
				
				auto& off_mesh= mesh.offs_mesh_GPU;
				auto* om= onika::cuda::vector_data(off_mesh);
				
				auto& cl= mesh.cells_GPU;
				size_t nb_cells= onika::cuda::vector_size(cl);
				auto* cells= onika::cuda::vector_data(cl);
				
				auto& nb= mesh.nb_meshes_GPU;
				auto* nb_meshes= onika::cuda::vector_data(nb);
				
				auto& nf= mesh.nb_faces_GPU;
				auto* nb_faces= onika::cuda::vector_data(nf);
				
				bool pass= false;
				int idx1=0;
				int idx2=0;
				int idx3=0;
				int idx4=0;
				int idx_cell;
				for(int i=0; i < nb_cells; i++){
					if(cells[i]== cell_idx){
						idx_cell=i;
						idx1=i;
						idx2=i;
						idx3=i;
						idx4=i;
						pass=true;
					}
				}
				
				if(pass){
					bool is_face= false;
					bool do_edge= false;
					for(int i = 0; i < nb_faces[idx_cell]; i++){
						if(i > 0){
							idx1= of[idx1];
						}
						Vec3d normal= {nx[idx1], ny[idx1], nz[idx1]};
						double offset= offsets[idx1];
						int size= nb_meshes[idx1];
						int index_start;
						int index_second;
						int index_last;
						for(int j = 0; j < size; j++){
							if( i > 0 || j > 0){
								idx2= om[idx2];
							}
							if(j == 0) index_start= idx2;
							if(j == 1) index_second = idx2;
							if(j == size-1) index_last = idx2;
						}
						bool contact= false;
						bool potential= false;
						Vec3d position= {0,0,0};
						
						
						
						//CONTACT_FACE_SPHERE
						const Vec3d center = {a_rx,a_ry,a_rz};

						double p = exanb::dot(center,normal) - offset;
			
						if(abs(p) <= a_particle_radius)
						{
	
						potential = true;

						const Vec3d& pa = {vx[index_start], vy[index_start], vz[index_start]};
						const Vec3d& pb = {vx[index_second], vy[index_second], vz[index_second]};
						const Vec3d& pc = {vx[index_last], vy[index_last], vz[index_last]};
						Vec3d v1 = pb - pa;
						Vec3d v2 = pc - pa;
						normalize(v1);
						Vec3d n = exanb::cross(v1,v2);
						normalize(n);
						Vec3d iv = center;// - pa;
						double dist = exanb::dot(iv,n);
						if(dist < 0.0)
						{
							dist= -dist;
							n= -n;
						}

						// test if the sphere intersects the surface 
						int intersections = 0;

						// from rockable
						Vec3d P = iv - dist * n;
						v2 = exanb::cross(n, v1);
						double ori1 = exanb::dot(P,v1);
						double ori2 = exanb::dot(P,v2);

						for (int iva = 0; iva < size ; ++iva) {
							int d1;
							int d2;
							if(iva==0){ d1= index_start; d2= index_second;}
							if(iva==1){ d1= index_second; d2= index_last;}
							if(iva==2){ d1= index_last; d2= index_start;}
							const Vec3d& posNodeA_jv = {vx[d1], vy[d1], vz[d1]};
							const Vec3d& posNodeB_jv = {vx[d2], vy[d2], vz[d2]};
							double pa1 = exanb::dot(posNodeA_jv , v1);
							double pb1 = exanb::dot(posNodeB_jv , v1);
							double pa2 = exanb::dot(posNodeA_jv , v2);
							double pb2 = exanb::dot(posNodeB_jv , v2);

							// @see http://local.wasp.uwa.edu.au/~pbourke/geometry/insidepoly/
							// @see http://alienryderflex.com/polygon/
							if ((pa2 < ori2 && pb2 >= ori2) || (pb2 < ori2 && pa2 >= ori2)) {
								if (pa1 + (ori2 - pa2) / (pb2 - pa2) * (pb1 - pa1) < ori1) {
									intersections = 1 - intersections;
								}
							}
						}

						if(intersections == 1) // ODD 
						{
							position = normal*offset;//*contact;
							contact= true;
						}

						}
						//CONTACT_FACE_SPHERE END
						
						
						if(contact)
						{
							constexpr int type = 0;
							this->operator()(normal, position, type, a_rx, a_ry, a_rz, 
									a_vx, a_vy, a_vz, 
									a_vrot, a_particle_radius, 
									a_fx, a_fy, a_fz, 
									a_mass, a_mom, a_ft);
							is_face = true;
						}
						do_edge = do_edge || potential;
					}
					
					if(is_face==false && do_edge)
					{
						for(int i = 0; i < nb_faces[idx_cell]; i++){
							if(i > 0){
								idx3= of[idx3];
							}
							Vec3d normal= {nx[idx3], ny[idx3], nz[idx3]};
							double offset= offsets[idx3];
							int size= nb_meshes[idx3];
							bool contact= false;
							Vec3d position= {0,0,0};
							
							//CONTACT_EDGE_SPHERE
							const Vec3d center = {a_rx,a_ry,a_rz};
							const Vec3d default_contact_point = {0,0,0}; // won't be used
							int index_start = idx4;
							for (size_t iva = 0; iva < size; ++iva) {
								if( i > 0 || iva > 0 ){
									idx4 = om[idx4];
								}
								int ivb = iva + 1;
								int idx4_2;
								if( ivb == size ){
									idx4_2 = index_start;
								} else {
									idx4_2 = om[idx4];
								}
								Vec3d p1 = {vx[idx4], vy[idx4], vz[idx4]};
								Vec3d p2 = {vx[idx4_2], vy[idx4_2], vz[idx4_2]};
								Vec3d edge = p2 - p1;
								Vec3d sphereToEdge = center - p1;

								double distanceToEdge = length(exanb::cross(edge, sphereToEdge)) / length(edge);

								if (distanceToEdge <= a_particle_radius && exanb::dot(sphereToEdge, edge) > 0 && exanb::dot(sphereToEdge - edge, edge) < 0) {
									auto n_edge = edge / exanb::norm(edge);
									Vec3d contact_position = p1 + n_edge * dot(sphereToEdge, n_edge);
									contact = true;
									position = contact_position;
								}
							}
							if(contact)
							{
								constexpr int type = 1;
								this->operator()(normal, position, type, a_rx, a_ry, a_rz, 
									a_vx, a_vy, a_vz, 
									a_vrot, a_particle_radius, 
									a_fx, a_fy, a_fz, 
									a_mass, a_mom, a_ft);
							}
						}
					}
					idx3= idx1;
					idx4 = idx2;
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
				Vec3d normal, const Vec3d& contact_position, const int type,
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


			if(type == 0){
			
				pos_proj = dot(pos, normal) * normal;
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
			const Vec3d rigid_surface_velocity = normal * m_vel; 
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
			a_fx += f.x;// * contact;
			a_fy += f.y;// * contact;
			a_fz += f.z;// * contact;
		}
		
	};
}


namespace exanb
{
	template<> struct ComputeCellParticlesTraits<exaDEM::ApplyHookeSTLMeshesFunctor>
	{
		static inline constexpr bool RequiresBlockSynchronousCall = false;
		static inline constexpr bool CudaCompatible = true	;
	};

	template<> struct ComputeCellParticlesTraitsUseCellIdx<exaDEM::ApplyHookeSTLMeshesFunctor>
	{
		static inline constexpr bool UseCellIdx = true;
	};
}
