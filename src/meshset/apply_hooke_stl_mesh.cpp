//#pragma xstamp_cuda_enable


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
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_operators.h>
#include <exanb/core/basic_types_stream.h>
#include <onika/memory/allocator.h> // for ONIKA_ASSUME_ALIGNED macro
#include <exanb/compute/compute_pair_optional_args.h>
#include <vector>
#include <iomanip>

#include <exanb/compute/compute_cell_particles.h>
//#include <exaDEM/face.h>
//#include <exaDEM/stl_mesh.h>
//#include <exaDEM/hooke_stl_meshes.h>
#include <exaDEM/interaction.h>

#include <onika/cuda/cuda.h> // mots cles specifiques
#include <onika/memory/allocator.h> // cudaMMVector

#include <exaDEM/common_compute_kernels.h>
#include <exaDEM/compute_hooke_force.h>

#include <mpi.h>

namespace exaDEM
{
  using namespace exanb;
  
  ONIKA_HOST_DEVICE_FUNC
  void normalize (Vec3d& in)
  {
  	const double norm = exanb::norm (in);
  	in = in / norm;
  }
  
  ONIKA_HOST_DEVICE_FUNC
  double length2(Vec3d& v) {return std::sqrt(v.x*v.x+v.y*v.y+v.z*v.z);}
  
  ONIKA_HOST_DEVICE_FUNC
  double length2(const Vec3d& v) {return std::sqrt(v.x*v.x+v.y*v.y+v.z*v.z);}
  
  
  template<class GridT>__global__ void ApplyHookeSTLMesh_GPU(GridT* cells, 
									int* pa,
									int* cella,
									int* faces,
									double* nx,
									double* ny,
									double* nz,
									double* offsets,
									int* num_vertices,
									int* contacts,
									int* which_particle,
									int* add,
									int* pot,
									double* vx,
									double* vy,
									double* vz,
									double* px,
									double* py,
									double* pz,
									int size)
		{
		//printf("INCLUDERE\n");
		//Index of the particle in the interaction's list
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		
		if(idx < size){
			
			int p_a = pa[idx];
			int cell_a = cella[idx];
			
			double n_x = nx[idx];
			double n_y = ny[idx];
			double n_z = nz[idx];
			
			Vec3d normal = {n_x, n_y, n_z};
			
			double off = offsets[idx];
			
			int face_idx = faces[idx];
			
			int num = num_vertices[idx];
			
			int s = face_idx;
			int e = s + num;
			
			double rx = cells[cell_a][field::rx][p_a];
			double ry = cells[cell_a][field::ry][p_a];
			double rz = cells[cell_a][field::rz][p_a];
			double radius = cells[cell_a][field::radius][p_a];
			
			bool contact = false;
			bool potential = false;
			Vec3d position = {0, 0, 0};
			bool do_edge;
			
			const Vec3d center = {rx, ry, rz};
			double p = exanb::dot(center, normal) - off;
			
			if(abs(p) <= radius)
			{
				potential = true;
				const Vec3d& pa = { vx[s], vy[s], vz[s] };
				const Vec3d& pb = { vx[s + 1], vy[s + 1], vz[s + 1] };
				const Vec3d& pc = { vx[e - 1], vy[e - 1], vz[e - 1] };
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

					for (int iva = 0; iva < num ; ++iva) {
						int ivb = iva + 1;
						if (ivb == num) ivb = 0;
						const Vec3d& posNodeA_jv = { vx[s + iva], vy[s + iva], vz[s + iva] };
						const Vec3d& posNodeB_jv = { vx[s + ivb], vy[s + ivb], vz[s + ivb] };
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
						position = normal*off;
						contact= true;
					}

			}
			
			if(contact)
			{
				//printf("UNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN\n");
				px[idx] = position.x;
				py[idx] = position.y;
				pz[idx] = position.z;
				contacts[idx] = 1;
				//printf("VAIIIIIII\n");
			}
			
			int particle = which_particle[idx];
			
			if(contact){ atomicAdd(&add[particle], 1); }
			
			if(potential){ atomicAdd(&pot[particle], 1); }		
				
			}
		}
		
		
		template<class GridT>__global__ void ApplyHookeSTLMesh_GPU2(GridT* cells, 
									int* pa,
									int* cella,
									int* faces,
									double* nx,
									double* ny,
									double* nz,
									double* offsets,
									int* num_vertices,
									int* contacts,
									int* which_particle,
									int* add,
									int* pot,
									double* vx,
									double* vy,
									double* vz,
									double* px,
									double* py,
									double* pz,
									int size)
		{
		
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		
		if(idx < size)
		{
			int particle = which_particle[idx];
			if(add[particle] == 0 && pot[particle] > 0)
			{
				int p_a = pa[idx];
				int cell_a = cella[idx];
			
				double n_x = nx[idx];
				double n_y = ny[idx];
				double n_z = nz[idx];
			
				Vec3d normal = {n_x, n_y, n_z};
			
				double off = offsets[idx];
			
				int face_idx = faces[idx];
			
				int num = num_vertices[idx];
				
				int s = face_idx;
				int e = s + num;
			
				double rx = cells[cell_a][field::rx][p_a];
				double ry = cells[cell_a][field::ry][p_a];
				double rz = cells[cell_a][field::rz][p_a];
				double radius = cells[cell_a][field::radius][p_a];
				
				bool contact = false;
				Vec3d position = {0, 0, 0};
				
				const Vec3d center = {rx, ry, rz};
				for (size_t j = 0; j < num; ++j) {
						int z = j + 1;
						if(z == num) z = 0;
						Vec3d p1 = { vx[s + j], vy[s + j], vz[s + j] };
						Vec3d p2 = { vx[s + z], vy[s + z], vz[s + z] };
						Vec3d edge = p2 - p1;
						Vec3d sphereToEdge = center - p1;

						double distanceToEdge = length2(exanb::cross(edge, sphereToEdge)) / length2(edge);

						if (distanceToEdge <= radius && exanb::dot(sphereToEdge, edge) > 0 && exanb::dot(sphereToEdge - edge, edge) < 0) {
							auto n_edge = edge / exanb::norm(edge);
							Vec3d contact_position = p1 + n_edge * dot(sphereToEdge, n_edge);
							contact = true;
							position = contact_position;
						}
				}
				
				if(contact)
				{
					//printf("DEUXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n");
					contacts[idx] = 1;
					px[idx] = position.x;
					py[idx] = position.y;
					pz[idx] = position.z;
					//printf("DOOOOOOOOOOOOOVE\n");
				}
				
			}
		}
		
		}
		
		
		template<class GridT>__global__ void ApplyHookeSTLMesh_GPU3(GridT* cells, 
									int* pa,
									int* cella,
									double* nx,
									double* ny,
									double* nz,
									double* offsets,
									int* contacts,
									double* px,
									double* py,
									double* pz,
									double* ftx,
									double* fty,
									double* ftz,
									int* add,
									int* which_particle,
									double dt, 
									double kt, 
									double kn, 
									double kr, 
									double mu, 
									double dampRate,
									int size)
		{
			int idx = threadIdx.x + blockIdx.x * blockDim.x;
			if(idx < size)
			{
				if(contacts[idx] == 1)
				{
					
					int p_a = pa[idx];
					int cell_a = cella[idx];
					
					int particle = which_particle[idx];
					
					//Vec3d mom = cells[cell_a][field::mom][p_a];
					Vec3d ft = {ftx[idx], fty[idx], ftz[idx]};
					Vec3d vrot = cells[cell_a][field::vrot][p_a];
					
					double rx = cells[cell_a][field::rx][p_a];
					double ry = cells[cell_a][field::ry][p_a];
					double rz = cells[cell_a][field::rz][p_a];
					
					double vx = cells[cell_a][field::vx][p_a];
					double vy = cells[cell_a][field::vy][p_a];
					double vz = cells[cell_a][field::vz][p_a];
					
					double radius = cells[cell_a][field::radius][p_a];
					
					
					
					Vec3d pos_proj;
					double m_vel = 0;
					Vec3d pos = {rx, ry, rz};
					Vec3d vel = {vx, vy, vz};
					
					Vec3d normal = {nx[idx], ny[idx], nz[idx]};
					
					
					
					if(add[particle] > 0)
					{
						pos_proj = dot(pos, normal) * normal;
					}
					else
					{
						pos_proj = pos;
					}
					
					Vec3d contact_position = {px[idx], py[idx], pz[idx]};
					
					//printf("POSITIONS: RX:%f RY:%f RZ:%f CPX:%f CPY:%f CPZ:%f PROJX:%f PROJY:%f PROJZ:%f\n", rx, ry, rz, px[idx], py[idx], pz[idx], pos_proj.x, pos_proj.y, pos_proj.z);
					
					Vec3d vec_n = pos_proj - contact_position;
					Vec3d vec_n_before = vec_n;
					double n = exanb::norm(vec_n);
					vec_n = vec_n / n;
					const double dn = n - radius;
					Vec3d rigid_surface_center = contact_position;
					const Vec3d rigid_surface_velocity = normal * m_vel;
					constexpr Vec3d rigid_surface_angular_velocity = {0.0, 0.0, 0.0};
					
					//Vec3d f_i = {0.0, 0.0, 0.0};
					double fx=0;
					double fy=0;
					double fz=0;
					double momx = 0;
					double momy=0;
					double momz=0;
					constexpr double meff = 1;
					
					auto pos_i = pos_proj;
					auto vel_i = vel;
					auto vrot_i = vrot;
					auto pos_j = rigid_surface_center;
					auto vel_j = rigid_surface_velocity;
					auto vrot_j = rigid_surface_angular_velocity;
					
					if(dn <= 0.0) 
					{
						
						const double damp = compute_damp(dampRate, kn,  meff);

						// === Relative velocity (j relative to i)
						auto vel = compute_relative_velocity(
							contact_position,
							pos_i, vel_i, vrot_i,
							pos_j, vel_j, vrot_j
							);
							
						//printf("VELOCITY : (%f, %f, %f)\n", vel.x, vel.y, vel.z);

						// compute relative velocity
						const double vn = exanb::dot(vec_n, vel);

						// === Normal force (elatic contact + viscous damping)
						const Vec3d fn = compute_normal_force(kn, damp, dn, vn, vec_n); // fc ==> cohesive force
						
						

						// === Tangential force (friction)
						ft	 		+= exaDEM::compute_tangential_force(kt, dt, vn, vec_n, vel);


						// fit tangential force
						auto threshold_ft 	= exaDEM::compute_threshold_ft(mu, kn, dn);
						exaDEM::fit_tangential_force(threshold_ft, ft);

						// === sum forces
						const auto f = fn + ft;

						// === update forces
						//f_i += f;
						//f_i += f;
						fx = f.x;
						fy = f.y;
						fz = f.z;

						// === update moments
						const Vec3d mom = kr * (vrot_j - vrot_i) * dt;
						const auto Ci = (contact_position - pos_i);
						const auto Pimoment = exanb::cross(Ci, f) + mom;
						//mom_i += Pimoment;
						momx = Pimoment.x;
						momy = Pimoment.y;
						momz = Pimoment.z;
						
						
					}
					else
					{
						reset(ft); // no friction if no contact
					}
					
					
					
					
					ftx[idx] = ft.x;
					fty[idx] = ft.y;
					ftz[idx] = ft.z;
					
					//printf("FX:%f FY:%f FZ:%f FTX:%f FTY:%f FTZ:%f\n", cells[cell_a][field::fx][p_a], cells[cell_a][field::fy][p_a], cells[cell_a][field::fz][p_a], ft.x, ft.y, ft.z);
					
					atomicAdd(&cells[cell_a][field::mom][p_a].x, momx);
					atomicAdd(&cells[cell_a][field::mom][p_a].y, momy);
					atomicAdd(&cells[cell_a][field::mom][p_a].z, momz);
					atomicAdd(&cells[cell_a][field::fx][p_a], fx);
					atomicAdd(&cells[cell_a][field::fy][p_a], fy);
					atomicAdd(&cells[cell_a][field::fz][p_a], fz);
					
					//printf("MOMX:%f MOMY:%f MOMZ:%f FX:%f FY:%f FZ:%f\n", momx, momy, momz, fx, fy, fz);
				}
				
				contacts[idx] = 0;
			}
		}
		
		__global__ void set(int* add, int* pot, int size)
		{
			int idx = threadIdx.x + blockIdx.x * blockDim.x;
			if(idx < size)
			{
				add[idx] = 0;
				pot[idx] = 0;
			}
		}
  
  
  
  
  template<
    class GridT,
	  class = AssertGridHasFields< GridT, field::_fx, field::_fy, field::_fz >
	    >
	    class ApplyHookeSTLMeshesOperator : public OperatorNode
	    {

	      using ComputeFields = FieldSet< field::_rx ,field::_ry ,field::_rz, field::_vx ,field::_vy ,field::_vz, field::_vrot, field::_radius , field::_fx ,field::_fy ,field::_fz, field::_mass, field::_mom >;
	      static constexpr ComputeFields compute_field_set {};
	      ADD_SLOT( MPI_Comm , mpi                         , INPUT        , MPI_COMM_WORLD);
	      ADD_SLOT( GridT    , grid                        , INPUT_OUTPUT );
	      ADD_SLOT( Domain   , domain                      , INPUT        , REQUIRED );
	      //ADD_SLOT( std::vector<stl_mesh> , stl_collection , INPUT_OUTPUT , DocString{"list of verticies"});
	      ADD_SLOT( double                , dt             , INPUT        , REQUIRED , DocString{"Timestep of the simulation"});
	      ADD_SLOT( double                , kt             , INPUT        , REQUIRED , DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
	      ADD_SLOT( double                , kn             , INPUT        , REQUIRED , DocString{"Parameter of the force law used to model contact cyclinder/sphere"} );
	      ADD_SLOT( double                , kr             , INPUT        , REQUIRED , DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
	      ADD_SLOT( double                , mu             , INPUT        , REQUIRED , DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
	      ADD_SLOT( double                , damprate       , INPUT        , REQUIRED , DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
	      ADD_SLOT(Interactions, Int, INPUT_OUTPUT, DocString{"List of interactions between the particles and the faces of the STL meshes"});

	      public:
	      inline std::string documentation() const override final
	      {
		return R"EOF(
                )EOF";
	      }

	      inline void execute () override final
	      {
	      	//printf("APPLY START\n");
		//ApplyHookeSTLMeshesFunctor func { *stl_collection, *dt, *kt, *kn, *kr, *mu, *damprate};
		//compute_cell_particles( *grid , false , func , compute_field_set , parallel_execution_context() );
		auto& g = *grid;
								const auto cells = g.cells();
								
								auto& I = *Int;
								//printf("INTERACTIONS : %d\n", I.nb_interactions);
								//if(I.nb_interactions > 0) getchar();
								int size = I.nb_interactions;
								int blockSize = 128;
								int numBlocks;
								if(size % blockSize == 0){ numBlocks = size/blockSize;}
								else if(size / blockSize < 1) { numBlocks=1; blockSize = size;}
								else  { numBlocks= int(size/blockSize)+1; }
								
								onika::memory::CudaMMVector<int> tata;
								tata.resize(1);
								
								
								ApplyHookeSTLMesh_GPU<<<numBlocks, blockSize>>>(cells, I.pa_GPU2.data(), I.cella_GPU2.data(), I.faces_idx_GPU2.data(), I.nx_GPU2.data(), I.ny_GPU2.data(), I.nz_GPU2.data(), I.offsets_GPU2.data(), I.num_vertices_GPU2.data(), I.contact.data(), I.which_particle2.data(), I.add_particle.data(), I.potentiels.data(), I.vx_GPU2.data(), I.vy_GPU2.data(), I.vz_GPU2.data(), I.posx.data(), I.posy.data(), I.posz.data(), size);
								
								ApplyHookeSTLMesh_GPU2<<<numBlocks, blockSize>>>(cells, I.pa_GPU2.data(), I.cella_GPU2.data(), I.faces_idx_GPU2.data(), I.nx_GPU2.data(), I.ny_GPU2.data(), I.nz_GPU2.data(), I.offsets_GPU2.data(), I.num_vertices_GPU2.data(), I.contact.data(), I.which_particle2.data(), I.add_particle.data(), I.potentiels.data(), I.vx_GPU2.data(), I.vy_GPU2.data(), I.vz_GPU2.data(), I.posx.data(), I.posy.data(), I.posz.data(), size);
								
								ApplyHookeSTLMesh_GPU3<<<numBlocks, blockSize>>>(cells, I.pa_GPU2.data(), I.cella_GPU2.data(), I.nx_GPU2.data(), I.ny_GPU2.data(), I.nz_GPU2.data(), I.offsets_GPU2.data(), I.contact.data(), I.posx.data(), I.posy.data(), I.posz.data(), I.ftx_GPU2.data(), I.fty_GPU2.data(), I.ftz_GPU2.data(), I.add_particle.data(), I.which_particle2.data(), *dt, *kt, *kn, *kr, *mu, *damprate, size);
								
								int size2 = I.nb_particles;
								if(size2 % blockSize == 0){ numBlocks = size2/blockSize;}
								else if(size2 / blockSize < 1) { numBlocks=1; blockSize = size2;}
								else  { numBlocks= int(size2/blockSize)+1; }
								
								set<<<numBlocks, blockSize>>>(I.add_particle.data(), I.potentiels.data(), size2);
								
								/*auto& pa = I.pa_GPU;
								auto& cella = I.cella_GPU;
								auto& faces = I.faces_build_GPU;
								auto& nx = I.nx_GPU;
								auto& ny = I.ny_GPU;
								auto& nz = I.nz_GPU;
								auto& offsets = I.offsets_GPU;
								auto& num_vertices = I.num_vertices_GPU;
								auto& contacts = I.contact;
								auto& which_particle = I.which_particle;
								auto& add = I.add_particle;
								auto& pot = I.potentiels;
								auto& vx = I.vx_GPU;
								auto& vy = I.vy_GPU;
								auto& vz = I.vz_GPU;
								auto& px = I.posx;
								auto& py = I.posy;
								auto& pz = I.posz;
								
								auto& ftx = I.ftx_GPU;
								auto& fty = I.fty_GPU;
								auto& ftz = I.ftz_GPU;
								
								auto& start = I.start_GPU;
									
								double dt2 = *dt; 
								double kt2 = *kt; 
								double kn2 = *kn; 
								double kr2 = *kr; 
								double mu2 = *mu; 
								double dampRate2 = *damprate;
								
								bool zeta = false;
								
								/*for(int k = 0; k < I.nb_interactions; k++)
								{
									printf("INTERACTION: %d\n", k);
									int s = start[k];
									printf("OK\n");
									int nb = num_vertices[k];
									printf("OK2\n");
									for(int z = s; z < s+nb; z++)
									{
										printf("VX[%d] = %f  ", z, vx[z]);
									}
									printf("\n");
									for(int z = s; z < s+nb; z++)
									{
										printf("VY[%d] = %f  ", z, vy[z]);
									}
									printf("\n");
									for(int z = s; z < s+nb; z++)
									{
										printf("VZ[%d] = %f  ", z, vz[z]);
									}
									printf("\n");
								}*/
								
								/*for(int i = 0; i < I.nb_particles; i++)
								{
									printf("PA= %d CELLA= %d\n", I.pa[i], I.cella[i]);
								}*/
								
								/*double forcex = 0;
								double forcey = 0;
								double forcez = 0;
								
								
								for(int i = 0; i < I.nb_particles; i++)
								{
									int p_a = I.pa[i];
									int cell_a = I.cella[i];
									forcex = cells[cell_a][field::fx][p_a]; 
									forcey = cells[cell_a][field::fy][p_a]; 
									forcez = cells[cell_a][field::fz][p_a];
								}
								
								for(int idx = 0; idx < I.nb_interactions; idx++)
								{
									//UNNNNNNNNNNNNNNNNNNNNNNN
									int p_a = pa[idx];
									int cell_a = cella[idx];
			
									double n_x = nx[idx];
									double n_y = ny[idx];
									double n_z = nz[idx];
			
									Vec3d normal = {n_x, n_y, n_z};
			
									double off = offsets[idx];
			
									int face_idx = faces[idx];
			
									int num = num_vertices[idx];
			
									int s = start[idx];
									int e = s + num;
			
									double rx = cells[cell_a][field::rx][p_a];
									double ry = cells[cell_a][field::ry][p_a];
									double rz = cells[cell_a][field::rz][p_a];
									double radius = cells[cell_a][field::radius][p_a];
			
									bool contact = false;
									bool potential = false;
									Vec3d position = {0, 0, 0};
									bool do_edge;
			
									const Vec3d center = {rx, ry, rz};
									double p = exanb::dot(center, normal) - off;
			
									if(abs(p) <= radius)
									{
										potential = true;
										const Vec3d& pa = { vx[s], vy[s], vz[s] };
										const Vec3d& pb = { vx[s + 1], vy[s + 1], vz[s + 1] };
										const Vec3d& pc = { vx[e - 1], vy[e - 1], vz[e - 1] };
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

										for (int iva = 0; iva < num ; ++iva) {
											int ivb = iva + 1;
											if (ivb == num) ivb = 0;
											const Vec3d& posNodeA_jv = { vx[s + iva], vy[s + iva], vz[s + iva] };
											const Vec3d& posNodeB_jv = { vx[s + ivb], vy[s + ivb], vz[s + ivb] };
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
											position = normal*off;
											contact= true;
										}

									}
			
									if(contact)
									{
				
										px[idx] = position.x;
										py[idx] = position.y;
										pz[idx] = position.z;
										contacts[idx] = 1;
										
									}
			
									int particle = which_particle[idx];
			
									if(contact){ add[particle]+= 1; }
			
									if(potential){ pot[particle]+= 1; }
									
											
				
								}
								
								for(int idx = 0; idx < I.nb_interactions; idx++)
								{
									int particle = which_particle[idx];
									if(add[particle] == 0 && pot[particle] > 0)
									{
										int p_a = pa[idx];
										int cell_a = cella[idx];
			
										double n_x = nx[idx];
										double n_y = ny[idx];
										double n_z = nz[idx];
			
										Vec3d normal = {n_x, n_y, n_z};
			
										double off = offsets[idx];
			
										int face_idx = faces[idx];
			
										int num = num_vertices[idx];
				
										int s = start[idx];
										int e = s + num;
			
										double rx = cells[cell_a][field::rx][p_a];
										double ry = cells[cell_a][field::ry][p_a];
										double rz = cells[cell_a][field::rz][p_a];
										double radius = cells[cell_a][field::radius][p_a];
				
										bool contact = false;
										Vec3d position = {0, 0, 0};
				
										const Vec3d center = {rx, ry, rz};
										for (size_t j = 0; j < num; ++j) {
											int z = j + 1;
											if(z == num) z = 0;
											Vec3d p1 = { vx[s + j], vy[s + j], vz[s + j] };
											Vec3d p2 = { vx[s + z], vy[s + z], vz[s + z] };
											Vec3d edge = p2 - p1;
											Vec3d sphereToEdge = center - p1;

											double distanceToEdge = length2(exanb::cross(edge, sphereToEdge)) / length2(edge);

											if (distanceToEdge <= radius && exanb::dot(sphereToEdge, edge) > 0 && exanb::dot(sphereToEdge - edge, edge) < 0) {
												auto n_edge = edge / exanb::norm(edge);
												Vec3d contact_position = p1 + n_edge * dot(sphereToEdge, n_edge);
												contact = true;
												position = contact_position;
											}
										}
				
										if(contact)
										{
					
											contacts[idx] = 1;
											px[idx] = position.x;
											py[idx] = position.y;
											pz[idx] = position.z;
					
										}
				
									}
								}
								
								for(int idx = 0; idx < I.nb_interactions; idx++)
								{
									if(contacts[idx] == 1)
									{
										printf("DUCE\n");
					
										int p_a = pa[idx];
										int cell_a = cella[idx];
					
										int particle = which_particle[idx];
					
										//Vec3d mom = cells[cell_a][field::mom][p_a];
										Vec3d ft = {ftx[idx], fty[idx], ftz[idx]};
										Vec3d vrot = cells[cell_a][field::vrot][p_a];
					
										double rx = cells[cell_a][field::rx][p_a];
										double ry = cells[cell_a][field::ry][p_a];
										double rz = cells[cell_a][field::rz][p_a];
					
										double vx2 = cells[cell_a][field::vx][p_a];
										double vy2 = cells[cell_a][field::vy][p_a];
										double vz2 = cells[cell_a][field::vz][p_a];
						
										double radius = cells[cell_a][field::radius][p_a];
					
					
					
										Vec3d pos_proj;
										double m_vel = 0;
										Vec3d pos = {rx, ry, rz};
										Vec3d vel = {vx2, vy2, vz2};
					
										Vec3d normal = {nx[idx], ny[idx], nz[idx]};
					
					
					
										if(add[particle] > 0)
										{
											pos_proj = dot(pos, normal) * normal;
										}
										else
										{
											pos_proj = pos;
										}
					
										Vec3d contact_position = {px[idx], py[idx], pz[idx]};
					
					
										Vec3d vec_n2 = pos_proj - contact_position;
										Vec3d vec_n_before = vec_n2;
										double n = exanb::norm(vec_n_before);
										double nsdrumox = n;
										//vec_n = vec_n / n;
										printf("CALCUL DE MERDE ::: VNX = %f N = %f RESULTAT = %f\n", vec_n_before.x, n, (vec_n_before.x/n));
										double coordx = vec_n_before.x/n;
										double coordy = vec_n_before.y/n;
										double coordz = vec_n_before.z/n;
										printf("COORDONNEES(%f, %f, %f)\n", coordx, coordy, coordz);
										Vec3d vec_n = {coordx, coordy, coordz};
										printf("COORDOONEES AFTER(%f, %f, %f)\n", vec_n.x, vec_n.y, vec_n.z);
										const double dn = n - radius;
										Vec3d rigid_surface_center = contact_position;
										const Vec3d rigid_surface_velocity = normal * m_vel;
										constexpr Vec3d rigid_surface_angular_velocity = {0.0, 0.0, 0.0};
					
					
										double fx=0;
										double fy=0;
										double fz=0;
										double momx = 0;
										double momy=0;
										double momz=0;
										constexpr double meff = 1;
					
										auto pos_i = pos_proj;
										auto vel_i = vel;
										auto vrot_i = vrot;
										auto pos_j = rigid_surface_center;
										auto vel_j = rigid_surface_velocity;
										auto vrot_j = rigid_surface_angular_velocity;
					
										if(dn <= 0.0) 
										{
						
											const double damp = compute_damp(dampRate2, kn2,  meff);

											// === Relative velocity (j relative to i)
											auto vel = compute_relative_velocity(
												contact_position,
												pos_i, vel_i, vrot_i,
												pos_j, vel_j, vrot_j
											);
							
											//printf("VELOCITY : (%f, %f, %f)\n", vel.x, vel.y, vel.z);

											// compute relative velocity
											printf("VECN UUUNNN (%f, %f, %f)\n", vec_n.x, vec_n.y, vec_n.z);
											const double vn = exanb::dot(vec_n, vel);
											printf("VECN DEUUUUUUUX (%f, %f, %f)\n", vec_n.x, vec_n.y, vec_n.z);

											// === Normal force (elatic contact + viscous damping)
											const Vec3d fn = compute_normal_force(kn2, damp, dn, vn, vec_n); // fc ==> cohesive force
											printf("VECN TROIIIS(%f, %f, %f)\n", vec_n.x, vec_n.y, vec_n.z);
						

											// === Tangential force (friction)
											ft	 		+= exaDEM::compute_tangential_force(kt2, dt2, vn, vec_n, vel);


											// fit tangential force
											auto threshold_ft 	= exaDEM::compute_threshold_ft(mu2, kn2, dn);
											exaDEM::fit_tangential_force(threshold_ft, ft);

											// === sum forces
											const auto f = fn + ft;

											// === update forces
											//f_i += f;
											//f_i += f;
											fx = f.x;
											fy = f.y;
											fz = f.z;

											// === update moments
											const Vec3d mom = kr2 * (vrot_j - vrot_i) * dt2;
											const auto Ci = (contact_position - pos_i);
											const auto Pimoment = exanb::cross(Ci, f) + mom;
											//mom_i += Pimoment;
											momx = Pimoment.x;
											momy = Pimoment.y;
											momz = Pimoment.z;
						
											//if(add[particle] > 0)
											//{
												printf("PA:%d CELLA:%d F(%f, %f, %f) FT(%f, %f, %f) MOM(%f, %f, %f) FN(%f, %f, %f) KN:%f DAMP:%f DN:%f VN:%f VEL(%f, %f, %f) VECN(%f, %f, %f) POS_PROJ(%f, %f, %f) CONTACT_POSITION(%f, %f, %f) BEFORE(%f, %f, %f) NSDRUMOX:%f\n", p_a, cell_a, fx, fy, fz, ft.x, ft.y, ft.z, momx, momy, momz, fn.x, fn.y, fn.z, kn2, damp, dn, vn, vel.x, vel.y, vel.z, vec_n.x, vec_n.y, vec_n.z, pos_proj.x, pos_proj.y, pos_proj.z, contact_position.x, contact_position.y, contact_position.z, vec_n_before.x, vec_n_before.y, vec_n_before.z, nsdrumox);
											zeta = true;
											//}
											//else
											//{
												//printf("DEUXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX F(%f, %f, %f) FT(%f, %f, %f) MOM(%f, %f, %f) FN(%f, %f, %f) KN:%f DAMP:%f DN:%f VN:%f VEL(%f, %f, %f) VECN(%f, %f, %f)\n", fx, fy, fz, ft.x, ft.y, ft.z, momx, momy, momz, fn.x, fn.y, fn.z, kn, damp, dn, vn, vec_n.x, vel.x, vel.y, vel.z, vec_n.y, vec_n.z);
											//}
										}
										else
										{
											reset(ft); // no friction if no contact
										}
					
					
					
					
										ftx[idx] = ft.x;
										fty[idx] = ft.y;
										ftz[idx] = ft.z;
										
										Vec3d mom_cell = {momx, momy, momz};
					
										//printf("FX:%f FY:%f FZ:%f FTX:%f FTY:%f FTZ:%f\n", cells[cell_a][field::fx][p_a], cells[cell_a][field::fy][p_a], cells[cell_a][field::fz][p_a], ft.x, ft.y, ft.z);
					
										cells[cell_a][field::mom][p_a]+= mom_cell;
										
										cells[cell_a][field::fx][p_a]+= fx;
										cells[cell_a][field::fy][p_a]+= fy;
										cells[cell_a][field::fz][p_a]+= fz;
					
										//printf("MOMX:%f MOMY:%f MOMZ:%f FX:%f FY:%f FZ:%f\n", momx, momy, momz, fx, fy, fz);
									}
				
									contacts[idx] = 0;
								}
								
								for(int idx = 0; idx < I.nb_particles; idx++)
								{
									add[idx] = 0;
									pot[idx] = 0;
								}
								
								if(zeta)
								{
								for(int i = 0; i < I.nb_particles; i++)
								{
									int p_a = I.pa[i];
									int cell_a = I.cella[i];
									//printf("AVANT (PA:%d, CELLA:%d) F(FX:%f, FY:%f, FZ:%f)\n", pa, cella, forcex, forcey, forcez);
									//printf("AFTER (PA:%d, CELLA:%d) F(FX:%f, FY:%f, FZ:%f)\n", pa, cella, cells[cell_a][field::fx][p_a], cells[cell_a][field::fy][p_a], cells[cell_a][field::fz][p_a]);
									printf("VITESSE%d(%f, %f, %f)\n", i, cells[cell_a][field::vx][p_a], cells[cell_a][field::vy][p_a], cells[cell_a][field::vz][p_a]);
									//getchar();
								}
								}*/
							};
								
								 
								
								 
		
		//printf("APPLY END\n")
	    };


  // this helps older versions of gcc handle the unnamed default second template parameter
  template <class GridT> using ApplyHookeSTLMeshesOperatorTemplate = ApplyHookeSTLMeshesOperator<GridT>;

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "apply_hooke_stl_meshes", make_grid_variant_operator< ApplyHookeSTLMeshesOperatorTemplate > );
  }
}

