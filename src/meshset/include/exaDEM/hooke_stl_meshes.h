#pragma once
#include <exanb/core/basic_types.h>
#include <exaDEM/common_compute_kernels.h>
#include <exaDEM/compute_hooke_force.h>
#include <exaDEM/face.h>
#include <exaDEM/stl_mesh.h>
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
	 * @brief Functor for applying Hooke's law to multiple Faces within a grid adapted to GPU layout thanks to the utilisation of the stl_meshes struc
	 *
	 * The `ApplyHookeSTLMeshesFunctor` struct represents a functor used to apply Hooke's law to particles interacting
	 * with multiple STL meshes within a grid. It is designed to be used as an operator in simulations. The functor takes
	 * various parameters and updates forces and torques acting on the particles.
	 *
	 * @tparam GridT The type of grid.
	 */
	 struct ApplyHookeSTLMeshesFunctor_GPU
	 {
	 	stl_meshes meshes; /**The collection of meshes. */
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
			
			//for(size_t i=0; i< smeshes ; i++)
			for(size_t i=0; i < meshes.nb_meshes; i++)
			{
				//auto& mesh= pmeshes[i];
				//size_t msize = onika::cuda::vector_size(meshes.m_meshes[i]);
				auto* mesh= onika::cuda::vector_data(meshes.m_meshes[i]);
				//auto& indexes= mesh.indexes;
				//auto* indexes2= onika::cuda::vector_data(indexes);
				auto* indexes= onika::cuda::vector_data(meshes.indexes[i]);
				//auto& idx= indexes2[cell_idx];
				auto& idx= indexes[cell_idx];
				auto* idx2= onika::cuda::vector_data(idx);
				size_t fsize= onika::cuda::vector_size(idx);
				//auto& faces= mesh.m_data; Je l'ai déjà avec mon vector mesh
				//auto* faces2= onika::cuda::vector_data(faces); Je l'ai déjà avec mon vecteur mesh
				bool is_face = false; // If there is one contact with a face, we skip contact with edges
				bool do_edge = false;
				for(size_t face_idx = 0 ; face_idx < fsize ; face_idx++)
				{
					//auto& face = faces2[idx2[face_idx]];
					int idx_face = idx2[face_idx];
					int y = 1;
					Vec3d normal;
					double offset;
					onika::memory::CudaMMVector< Vec3d > vertices;
					int x = 0;
					bool boolean = true;
					//for(int x=0; x < mesh[0]; x++){
					while(boolean && x < mesh[0]){
						int nb_Vertices = mesh[y];
						normal = {mesh[y+1], mesh[y+2], mesh[y+3]};
						offset = mesh[y+4];
						y+=5;
						if(x == idx_face){
							for(; y < y + nb_Vertices*3; y+=3){
								Vec3d v = {mesh[y], mesh[y+1], mesh[y+2]};
								vertices.push_back(v);
							}
							boolean= false;
						} else {
							y+= nb_Vertices*3;
						}
					}
					bool contact= false;
					bool potential= false;
					Vec3d position= {0,0,0};
					contact_face_sphere(a_rx, a_ry, a_rz, a_particle_radius, contact, potential, position, vertices, normal, offset);
					constexpr int type = 0;
					this->operator()(normal, position, type, a_rx, a_ry, a_rz, 
							a_vx, a_vy, a_vz, 
							a_vrot, a_particle_radius, 
							a_fx, a_fy, a_fz, 
							a_mass, a_mom, a_ft,
							contact);
					is_face = contact || is_face;
					do_edge = do_edge || potential;
				}

				// contact edge / sphere
				if(is_face == false && do_edge)
				{
					for( size_t face_idx = 0 ; face_idx < fsize ; face_idx++)
					{
						//auto& face = faces2[idx2[face_idx]];
						int idx_face = idx2[face_idx];
						int y = 1;
						Vec3d normal;
						//double offset;
						onika::memory::CudaMMVector< Vec3d > vertices;
						bool boolean = true;
						int x = 0;
						while(boolean && x < mesh[0]){
						//for(int x=0; x < mesh[0]; x++){
							int nb_Vertices = mesh[y];
							 normal = {mesh[y+1], mesh[y+2], mesh[y+3]};
							//offset = mesh[y+4];
							y+=5;
							if(x == idx_face){
								for(; y < y + nb_Vertices*3; y+=3){
									Vec3d v = {mesh[y], mesh[y+1], mesh[y+2]};
									vertices.push_back(v);
								}
								boolean = false;
							} else {
								y+= nb_Vertices*3;
							}
							x++;
						}
						bool contact= false;
						Vec3d position = {0,0,0};
						contact_edge_sphere(a_rx, a_ry, a_rz, a_particle_radius, contact, position, vertices);
						constexpr int type = 1;
						this->operator()(normal, position, type, a_rx, a_ry, a_rz, 
								a_vx, a_vy, a_vz, 
								a_vrot, a_particle_radius, 
								a_fx, a_fy, a_fz, 
								a_mass, a_mom, a_ft,
								contact);
					}
				}
			}
		}
		
		ONIKA_HOST_DEVICE_FUNC void contact_face_sphere(const double rx, const double ry, const double rz, const double rad, bool& contact, bool& potential, Vec3d& position, onika::memory::CudaMMVector< Vec3d > vertices, Vec3d normal, double offset) const
		{
			const Vec3d center = {rx,ry,rz};

			double p = exanb::dot(center,normal) - offset;

			potential = abs(p) <= rad;

			const int nb_vertices = onika::cuda::vector_size(vertices);
			const Vec3d* vertices_array = onika::cuda::vector_data(vertices);
			const Vec3d& pa = vertices_array[0];
			const Vec3d& pb = vertices_array[1];
			const Vec3d& pc = vertices_array[nb_vertices-1];
			Vec3d v1 = pb - pa;
			Vec3d v2 = pc - pa;
			normalize(v1);
			Vec3d n = exanb::cross(v1,v2);
			normalize(n);
			Vec3d iv = center;// - pa;
			double dist = exanb::dot(iv,n);
			
			dist = (dist < 0.0)*-dist + (1 - (dist < 0.0))*dist;
			n = (dist < 0.0)*-n + (1 - (dist < 0.0))*n;
			

			// test if the sphere intersects the surface 
			int intersections = 0;

			// from rockable
			Vec3d P = iv - dist * n;
			v2 = exanb::cross(n, v1);
			double ori1 = exanb::dot(P,v1);
			double ori2 = exanb::dot(P,v2);

			for (int iva = 0; iva < nb_vertices ; ++iva) {
				int ivb = iva + 1;
				if (ivb == nb_vertices) ivb = 0;
				const Vec3d& posNodeA_jv = vertices_array[iva];
				const Vec3d& posNodeB_jv = vertices_array[ivb];
				double pa1 = exanb::dot(posNodeA_jv , v1);
				double pb1 = exanb::dot(posNodeB_jv , v1);
				double pa2 = exanb::dot(posNodeA_jv , v2);
				double pb2 = exanb::dot(posNodeB_jv , v2);

				// @see http://local.wasp.uwa.edu.au/~pbourke/geometry/insidepoly/
				// @see http://alienryderflex.com/polygon/
				//if ((pa2 < ori2 && pb2 >= ori2) || (pb2 < ori2 && pa2 >= ori2)) {
				//	if (pa1 + (ori2 - pa2) / (pb2 - pa2) * (pb1 - pa1) < ori1) {
				bool boolean = ((pa2 < ori2 && pb2 >= ori2) || (pb2 < ori2 && pa2 >= ori2)) && (pa1 + (ori2 - pa2) / (pb2 - pa2) * (pb1 - pa1) < ori1);
						intersections = boolean*(1 - intersections) + (1 - boolean)*intersections;
			}

			contact = (intersections==1) && potential;
			position = normal*offset*contact;
		}
		
		ONIKA_HOST_DEVICE_FUNC	
		void contact_edge_sphere(const double rx, const double ry, const double rz, const double rad, bool& contact, Vec3d& position, const onika::memory::CudaMMVector< Vec3d > vertices)
		const 
		{
			const Vec3d center = {rx,ry,rz};
			const Vec3d default_contact_point = {0,0,0}; // won't be used
			const Vec3d* vertices_array = onika::cuda::vector_data(vertices);
			for (size_t i = 0; i < onika::cuda::vector_size(vertices); ++i) {
				Vec3d p1 = vertices_array[i];
				Vec3d p2 = vertices_array[(i + 1) % onika::cuda::vector_size(vertices)];
				Vec3d edge = p2 - p1;
				Vec3d sphereToEdge = center - p1;

				double distanceToEdge = length(exanb::cross(edge, sphereToEdge)) / length(edge);

				if (distanceToEdge <= rad && exanb::dot(sphereToEdge, edge) > 0 && exanb::dot(sphereToEdge - edge, edge) < 0) {
					auto n_edge = edge / exanb::norm(edge);
					Vec3d contact_position = p1 + n_edge * dot(sphereToEdge, n_edge);
					contact = true;
					position = contact_position;
				}
			}
			
			contact = false;
			position = default_contact_point;
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
				Vec3d& a_ft,
				bool contact) const
		{
			Vec3d pos_proj;
			double m_vel = 0;
			Vec3d pos = {a_rx,a_ry,a_rz};
			Vec3d vel = {a_vx,a_vy,a_vz};

			
			pos_proj = (type)*pos + (1 - type)*dot(pos, normal) * normal;

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
			a_fx += f.x * contact;
			a_fy += f.y * contact;
			a_fz += f.z * contact;
		}
		
		

	 
	 };
	
	
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
		//std::vector<exaDEM::stl_mesh>& meshes; /**< A collection of STL meshes. */
		//onika::memory::CudaMMVector< exaDEM::stl_mesh >  meshes;
		exaDEM::stl_mesh*  pmeshes;
		long unsigned int smeshes;
		double m_dt; /**< Time step. */
		double m_kt; /**< Tangential spring constant. */
		double m_kn; /**< Normal spring constant. */
		double m_kr; /**< Rotational spring constant. */
		double m_mu; /**< Friction coefficient. */
		//cudaMallocManaged(&m_mu, sizeof(double));
		//int *gpuVariable;
		//int cpu= 0;
		//cudaMalloc((void**)&gpuVariable, sizeof(int));
		//cudaMemcpy(gpuVariable, &cpu, sizeof(int), cudaMemcpyHostToDevice);*/
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
			//auto* meshes_array= pmeshes; //onika::cuda::vector_data(meshes);
			//auto* meshes_array= onika::cuda::vector_data(meshes);
			//for(auto& mesh : meshes)
			for(size_t i=0; i< smeshes ; i++)//onika::cuda::vector_size(meshes); i++)
			{
				//exaDEM::stl_mesh & mesh= meshes[i];
				//auto& mesh= meshes_array[i];
				auto& mesh= pmeshes[i];
				auto& indexes= mesh.indexes;
				auto* indexes2= onika::cuda::vector_data(indexes);
				auto& idx= indexes2[cell_idx];
				auto* idx2= onika::cuda::vector_data(idx);
				size_t fsize= onika::cuda::vector_size(idx);
				auto& faces= mesh.m_data;
				auto* faces2= onika::cuda::vector_data(faces);
				bool is_face = false; // If there is one contact with a face, we skip contact with edges
				bool do_edge = false;
				/**bool contact;
				bool potential;
				Vec3d position;*/
				//printf("SIZE: %d\n", fsize);
				for(size_t face_idx = 0 ; face_idx < fsize ; face_idx++)
				{
					auto& face = faces2[idx2[face_idx]];
					bool contact= false;
					bool potential= false;
					Vec3d position= {0,0,0};
					face.contact_face_sphere(a_rx, a_ry, a_rz, a_particle_radius, contact, potential, position);
					//printf("CONTACT : %d\n", contact);
					//printf("POTENTIAL : %d\n", potential);
					//printf("Position.x : %f Position.y : %f Position.z : %f\n", position.x, position.y, position.z);
					//if(contact)
					//{
						//printf("TRUE\n");
						constexpr int type = 0;
						this->operator()(face, position, type, a_rx, a_ry, a_rz, 
								a_vx, a_vy, a_vz, 
								a_vrot, a_particle_radius, 
								a_fx, a_fy, a_fz, 
								a_mass, a_mom, a_ft,
								contact);
						//is_face = true;
						is_face = contact || is_face;
					//}
					do_edge = do_edge || potential;
				}

				// contact edge / sphere
				if(is_face == false && do_edge)
				{
					for( size_t face_idx = 0 ; face_idx < fsize ; face_idx++)
					{
						//auto& face = faces[idxs[face_idx]];
						auto& face = faces2[idx2[face_idx]];
						//auto [contact, position] = face.contact_edge_sphere(a_rx, a_ry, a_rz, a_particle_radius);
						bool contact = false;
						Vec3d position = {0,0,0};
						face.contact_edge_sphere(a_rx, a_ry, a_rz, a_particle_radius, contact, position);
						//if(contact)
						//{
							constexpr int type = 1;
							this->operator()(face, position, type, a_rx, a_ry, a_rz, 
									a_vx, a_vy, a_vz, 
									a_vrot, a_particle_radius, 
									a_fx, a_fy, a_fz, 
									a_mass, a_mom, a_ft,
									contact);
						//}
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
				Vec3d& a_ft,
				bool contact) const
		{
			//printf("JE SUIS ICI\n");
			Vec3d pos_proj;
			double m_vel = 0;
			Vec3d pos = {a_rx,a_ry,a_rz};
			Vec3d vel = {a_vx,a_vy,a_vz};


			/**if(type == 0)
			{
				pos_proj = dot(pos, face.normal) * face.normal;
			}
			else if(type == 1)
			{
				pos_proj = pos;
			}*/
			
			pos_proj = (type)*pos + (1 - type)*dot(pos, face.normal) * face.normal;

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
			a_fx += f.x * contact;
			a_fy += f.y * contact;
			a_fz += f.z * contact;
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
