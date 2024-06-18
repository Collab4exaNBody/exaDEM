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
#include <exaDEM/stl_meshes.h>
#include <exaDEM/interaction.h>

#include <mpi.h>

namespace exaDEM
{

	inline vec3r conv_to_vec3r (const Vec3d& v)
	{
		return vec3r {v.x, v.y, v.z};
	}
		
	inline std::vector<vec3r> conv_to_vec3r (std::vector<Vec3d> vector)
	{
		std::vector<vec3r> res;
		for(auto v: vector){
			res.push_back(conv_to_vec3r(v));
		}
		return res;
	}
	
	inline OBB build_OBB ( std::vector<Vec3d>& vertices, double radius)
	{
		
		std::vector<vec3r> vec = conv_to_vec3r(vertices);
		
		//double radius = 0.1;	
		
		OBB obb;
		vec3r mu;
		mat9r C;
		for (size_t i = 0; i < vec.size(); i++) {
			mu += vec[i];
		}
		mu /= (double)vec.size();

		// loop over the points again to build the
		// covariance matrix.  Note that we only have
		// to build terms for the upper trianglular
		// portion since the matrix is symmetric
		double cxx = 0.0, cxy = 0.0, cxz = 0.0, cyy = 0.0, cyz = 0.0, czz = 0.0;
		for (size_t i = 0; i < vec.size(); i++) {
			vec3r p = vec[i];
			cxx += p.x * p.x - mu.x * mu.x;
			cxy += p.x * p.y - mu.x * mu.y;
			cxz += p.x * p.z - mu.x * mu.z;
			cyy += p.y * p.y - mu.y * mu.y;
			cyz += p.y * p.z - mu.y * mu.z;
			czz += p.z * p.z - mu.z * mu.z;
		}


		// now build the covariance matrix
		C.xx = cxx;
		C.xy = cxy;
		C.xz = cxz;
		C.yx = cxy;
		C.yy = cyy;
		C.yz = cyz;
		C.zx = cxz;
		C.zy = cyz;
		C.zz = czz;

		// ==== set the OBB parameters from the covariance matrix
		// extract the eigenvalues and eigenvectors from C
		mat9r eigvec;
		vec3r eigval;
		C.sym_eigen(eigvec, eigval);

		// find the right, up and forward vectors from the eigenvectors
		vec3r r(eigvec.xx, eigvec.yx, eigvec.zx);
		vec3r u(eigvec.xy, eigvec.yy, eigvec.zy);
		vec3r f(eigvec.xz, eigvec.yz, eigvec.zz);
		r.normalize();
		u.normalize(), f.normalize();

		// now build the bounding box extents in the rotated frame
		vec3r minim(1e20, 1e20, 1e20), maxim(-1e20, -1e20, -1e20);
		for (size_t i = 0; i < vec.size(); i++) {
			vec3r p_prime(r * vec[i], u * vec[i], f * vec[i]);
			if (minim.x > p_prime.x) minim.x = p_prime.x;
			if (minim.y > p_prime.y) minim.y = p_prime.y;
			if (minim.z > p_prime.z) minim.z = p_prime.z;
			if (maxim.x < p_prime.x) maxim.x = p_prime.x;
			if (maxim.y < p_prime.y) maxim.y = p_prime.y;
			if (maxim.z < p_prime.z) maxim.z = p_prime.z;
		}

		// set the center of the OBB to be the average of the
		// minimum and maximum, and the extents be half of the
		// difference between the minimum and maximum
		obb.center = eigvec * (0.5 * (maxim + minim));
		obb.e[0] = r;
		obb.e[1] = u;
		obb.e[2] = f;
		obb.extent = 0.5 * (maxim - minim);
		//printf("EXTENT: (%f, %f, %f)\n", obb.extent.x, obb.extent.y, obb.extent.y);

		obb.enlarge(radius);  // Add the Minskowski radius
		return obb;
	}
	
	OBB sphere_to_obb(double rx, double ry, double rz, double radius)
	{
			OBB obb;
			Vec3d pos = {rx, ry, rz};
			Vec3d r = {radius, radius, radius};
			Box b1 = Box{pos - r, pos +r};
		
			Vec3d sup = b1.sup;
			Vec3d inf = b1.inf;
			double supx = sup.x;
			double supy = sup.y;
			double supz = sup.z;
			double infx = inf.x;
			double infy = inf.y;
			double infz = inf.z;
			//onika::memory::CudaMMVector<Vec3d> vertices;
			std::vector<Vec3d> vertices;
			vertices.push_back({supx, supy, supz});
			vertices.push_back({supx, infy, supz});
			vertices.push_back({supx, supy, infz});
			vertices.push_back({supx, infy, infz});
			vertices.push_back({infx, supy, supz});
			vertices.push_back({infx, supy, infz});
			vertices.push_back({infx, infy, supz});
			vertices.push_back({infx, infy, infz});
		
		
			OBB res = build_OBB(vertices, 0);
			return res;
	}
	
		/*__global__ void setGPU(int* pa,
					int* cella,
					int* faces,
					double* nx,
					double* ny,
					double* nz,
					double* offsets,
					int* num_vertices,
					double* ftx,
					double* fty,
					double* ftz,
					int* which_particle,
					int* contact,
					double* px,
					double* py,
					double* pz,
					onika::memory::CudaMMVector<int> pa2,
					onika::memory::CudaMMVector<int> cella2,
					onika::memory::CudaMMVector<int> faces2,
					onika::memory::CudaMMVector<double> nx2,
					onika::memory::CudaMMVector<double> ny2,
					onika::memory::CudaMMVector<double> nz2,
					onika::memory::CudaMMVector<double> offsets2,
					onika::memory::CudaMMVector<int> num_vertices2,
					onika::memory::CudaMMVector<double> ftx2,
					onika::memory::CudaMMVector<double> fty2,
					onika::memory::CudaMMVector<double> ftz2,
					onika::memory::CudaMMVector<int> which_particle2,
					int size)
		{
			int idx = threadIdx.x + blockIdx.x * blockDim.x;
			if(idx < size)
			{
				auto pa3 = onika::cuda::vector_data(pa2);
				auto cella3 = onika::cuda::vector_data(cella2);
				auto faces3 = onika::cuda::vector_data(faces2);
				auto nx3 = onika::cuda::vector_data(nx2);
				auto ny3 = onika::cuda::vector_data(ny2);
				auto nz3 = onika::cuda::vector_data(nz2);
				auto offsets3 = onika::cuda::vector_data(offsets2);
				auto num_vertices3 = onika::cuda::vector_data(num_vertices2);
				auto ftx3 = onika::cuda::vector_data(ftx2);
				auto fty3 = onika::cuda::vector_data(fty2);
				auto ftz3 = onika::cuda::vector_data(ftz2);
				auto which_particle3 = onika::cuda::vector_data(which_particle2);
				
				pa[idx] = pa3[idx];
				cella[idx] = cella3[idx];
				faces[idx] = faces3[idx];
				nx[idx] = nx3[idx];
				ny[idx] = ny3[idx];
				nz[idx] = nz3[idx];
				offsets[idx] = offsets3[idx];
				num_vertices[idx] = num_vertices3[idx];
				ftx[idx] = ftx3[idx];
				fty[idx] = fty3[idx];
				ftz[idx] = ftz3[idx];
				
				which_particle[idx] = which_particle3[idx];
				contact[idx] = 0;
				px[idx] = 0;
				py[idx] = 0;
				pz[idx] = 0;
			}
		}
		
		__global__ void setGPU2(int* add,
					int* pot,
					int size)
		{
			int idx = threadIdx.x + blockIdx.x * blockDim.x;
			if(idx < size)
			{
				add[idx] = 0;
				pot[idx] = 0;
			}
		}
		
		__global__ void setGPU3(double* vx,
					double* vy,
					double* vz,
					onika::memory::CudaMMVector<double> vx2,
					onika::memory::CudaMMVector<double> vy2,
					onika::memory::CudaMMVector<double> vz2,
					int size)
		{
			int idx = threadIdx.x + blockIdx.x * blockDim.x;
			if(idx < size)
			{
				auto vx3 = onika::cuda::vector_data(vx2);
				auto vy3 = onika::cuda::vector_data(vy2);
				auto vz3 = onika::cuda::vector_data(vz2);
				
				vx[idx] = vx3[idx];
				vy[idx] = vy3[idx];
				vz[idx] = vz3[idx];
			}
		}	*/
	
	
		
	
	
	using namespace exanb;
	template<	class GridT, class = AssertGridHasFields< GridT >> class BuildGridSTLMeshOperator : public OperatorNode
	{
		using ComputeFields = FieldSet< field::_rx ,field::_ry ,field::_rz>;
		static constexpr ComputeFields compute_field_set {};
		ADD_SLOT( MPI_Comm , mpi      , INPUT , MPI_COMM_WORLD , DocString{"MPI communicator for parallel processing."});
		ADD_SLOT( GridT    , grid     , INPUT_OUTPUT , DocString{"Grid used for computations."} );
		ADD_SLOT( double   , rcut_max , INPUT , 0.0, DocString{"Maximum cutoff radius for computations. Default is 0.0."} );
		//ADD_SLOT( std::vector<exaDEM::stl_mesh> , stl_collection, INPUT_OUTPUT , DocString{"Collection of meshes from stl files"});
		ADD_SLOT( double              , rcut_inc        , INPUT );
		ADD_SLOT(Interactions, Int, INPUT_OUTPUT);
		ADD_SLOT( exaDEM::stl_meshes, meshes, INPUT_OUTPUT, DocString{"Collection of meshes from stl files"});

		public:
		inline std::string documentation() const override final
		{
			return R"EOF( 
    	    			)EOF";
		}

		inline void execute () override final
		{
		
			/*printf("BUILD START\n");
			//auto& collection = *stl_collection;
			auto& mesh= *meshes;
			const double rad = *rcut_max;

			const auto cells = grid->cells();
			const size_t n_cells = grid->number_of_cells(); // nbh.size();
			const IJK dims = grid->dimension();
			const int gl = grid->ghost_layers();
<<<<<<< HEAD
			
			Interactions& interactions_new = *Int;
			Interactions interactions_old = interactions_new;
			
			//printf("ICI-UN\n");
			
			interactions_old.maj_friction();
=======
			for(auto &mesh : collection)
			{
				auto& ind = mesh.indexes;
				ind.resize(n_cells);
				mesh.build_boxes();
				std::cout << " JE SUIS LA " << std::endl;
#     pragma omp parallel
				{
					GRID_OMP_FOR_BEGIN(dims-2*gl,_,block_loc, schedule(dynamic))
					{
						IJK loc_a = block_loc + gl;
						size_t cell_a = grid_ijk_to_index( dims , loc_a );
						ind[cell_a].clear();
						auto cb = grid->cell_bounds(loc_a);
						Box bx = { cb.bmin - rad , cb.bmax + rad };
>>>>>>> origin/main

			//printf("ICI-DEUX\n");
			
			interactions_new.reset();
			
			//printf("ICI-TROIS\n");
			
			auto& ind2 = mesh.indexes2;
			auto& obb_faces = mesh.m_obbs;
			ind2.resize(n_cells);
			//mesh.build_boxes();
			mesh.build_obbs();
			
			//printf("ICI-QUATRE\n");
			
			std::vector< std::vector< std::vector< int>>> cell_particles_faces;
			std::vector< std::vector<int>> id_cell_particles_faces;
			
			//printf("ICI-CINQ\n");
				
			cell_particles_faces.resize(n_cells);
			id_cell_particles_faces.resize(n_cells);
			
			//printf("ICI\n");
				

#     pragma omp parallel
			{
				GRID_OMP_FOR_BEGIN(dims-2*gl,_,block_loc, schedule(dynamic))
				{
					IJK loc_a = block_loc + gl;
					size_t cell_a = grid_ijk_to_index( dims , loc_a );
					//ind[cell_a].clear();
					ind2[cell_a].clear();
					auto cb = grid->cell_bounds(loc_a);
					Box bx = { cb.bmin - rad , cb.bmax + rad };

					const int n_particles = cells[cell_a].size();
					if (n_particles == 0) continue;
					mesh.update_indexes2(cell_a, bx);
					if(ind2[cell_a].size() > 0){
						//const int n_particles = cells[cell_a].size();
						const uint64_t* __restrict__ id_a = cells[cell_a][ field::id ]; ONIKA_ASSUME_ALIGNED(id_a);
						const auto* __restrict__ rx_a = cells[cell_a][ field::rx ]; ONIKA_ASSUME_ALIGNED(rx_a);
						const auto* __restrict__ ry_a = cells[cell_a][ field::ry ]; ONIKA_ASSUME_ALIGNED(ry_a);
						const auto* __restrict__ rz_a = cells[cell_a][ field::rz ]; ONIKA_ASSUME_ALIGNED(rz_a);
						const auto* __restrict__ radius_a = cells[cell_a][ field::radius ]; ONIKA_ASSUME_ALIGNED(radius_a);
						cell_particles_faces[cell_a].resize(n_particles);
						id_cell_particles_faces[cell_a].resize(n_particles);
							
						for(int particle = 0; particle < n_particles; particle++){
							
							id_cell_particles_faces[cell_a][particle] = id_a[particle];
							for(int j = 0; j < ind2[cell_a].size(); j++){
								int idx = ind2[cell_a][j];
								OBB& obbface = obb_faces[idx];
								OBB sphere = sphere_to_obb(rx_a[particle], ry_a[particle], rz_a[particle], radius_a[particle]);
								sphere.enlarge(*rcut_inc);
								if(obbface.intersect(sphere))
								{
									cell_particles_faces[cell_a][particle].push_back(idx);	
								}
							}
						}
					}
				}
			}
			GRID_OMP_FOR_END
			
			//printf("ICI2\n");
			
			/*int rs = 0;
			
			for(int i = 0; i < cell_particles_faces.size(); i++)
			{
				int cell = i;
				for(int j = 0; j < cell_particles_faces[i].size(); j++)
				{
					int particle = j;
					auto ida = id_cell_particles_faces[i][j];
					auto faces = cell_particles_faces[i][j];
					if(faces.size() > 0)
					{
						rs++;
					}
				}
			}
			
			interactions_new.resize(rs);*/
			
			
			
			/*for(int i = 0; i < cell_particles_faces.size(); i++)
			{
				int cell = i;
				for(int j = 0; j < cell_particles_faces[i].size(); j++)
				{
					int particle = j;
					auto ida = id_cell_particles_faces[i][j];
					auto faces = cell_particles_faces[i][j];
					if(faces.size() > 0)
					{
						interactions_new.add_particle_func(particle, cell, faces, ida, mesh);
					}
				}
			}
			
			//printf("ICI3\n");
			
			interactions_new.quickSort();
			
			//printf("ICI4\n");
				
			interactions_new.init_friction(interactions_old);
			
			//printf("ICI5\n");
				
			interactions_new.init_GPU(mesh);
			
			//interactions_new.putParticles();
			
			//interactions_new.printParticles();
			
			
			//interactions_new.test(mesh);
			
			//printf("ICI6\n");
				
			int size = interactions_new.nb_interactions;
			int blockSize = 128;
			int numBlocks;
			if(size % blockSize == 0){ numBlocks = size/blockSize;}
			else if(size / blockSize < 1){ numBlocks=1; blockSize = size;}
			else{ numBlocks = int(size/blockSize)+1; }
			
			setGPU<<<numBlocks, blockSize>>>(interactions_new.pa_GPU2.data(), interactions_new.cella_GPU2.data(), interactions_new.faces_idx_GPU2.data(), interactions_new.nx_GPU2.data(), interactions_new.ny_GPU2.data(), interactions_new.nz_GPU2.data(), interactions_new.offsets_GPU2.data(), interactions_new.num_vertices_GPU2.data(), interactions_new.ftx_GPU2.data(), interactions_new.fty_GPU2.data(), interactions_new.ftz_GPU2.data(),  interactions_new.which_particle2.data(), interactions_new.contact.data(), interactions_new.posx.data(), interactions_new.posy.data(), interactions_new.posz.data(),
				interactions_new.pa_GPU, interactions_new.cella_GPU, interactions_new.faces_build_GPU, interactions_new.nx_GPU, interactions_new.ny_GPU, interactions_new.nz_GPU, interactions_new.offsets_GPU, interactions_new.num_vertices_GPU, interactions_new.ftx_GPU, interactions_new.fty_GPU, interactions_new.ftz_GPU, interactions_new.which_particle,
				size);
				
			int size2 = interactions_new.nb_particles;
			if(size2 % blockSize == 0){ numBlocks = size2/blockSize;}
			else if(size2 / blockSize < 1){ numBlocks=1; blockSize = size2;}
			else{ numBlocks = int(size2/blockSize)+1; }
				
			setGPU2<<<numBlocks, blockSize>>>(interactions_new.add_particle.data(), interactions_new.potentiels.data(), size2);
				
			int size3 = interactions_new.vx_GPU.size();
			if(size3 % blockSize == 0){ numBlocks = size3/blockSize;}
			else if(size3 / blockSize < 1){ numBlocks=1; blockSize = size3;}
			else{ numBlocks = int(size3/blockSize)+1; }
				
			setGPU3<<<numBlocks, blockSize>>>(interactions_new.vx_GPU2.data(), interactions_new.vy_GPU2.data(), interactions_new.vz_GPU2.data(),
							interactions_new.vx_GPU, interactions_new.vy_GPU, interactions_new.vz_GPU, size3);
							
			printf("BUILD END\n");*/
				
		};
	};

	// this helps older versions of gcc handle the unnamed default second template parameter
	template <class GridT> using BuildGridSTLMeshOperatorTemplate = BuildGridSTLMeshOperator<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "build_grid_stl_mesh", make_grid_variant_operator< BuildGridSTLMeshOperatorTemplate > );
	}
}
