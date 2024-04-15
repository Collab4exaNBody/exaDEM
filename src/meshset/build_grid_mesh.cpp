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
#include <exaDEM/face.h>

//#include <exaDEM/stl_mesh.h>
#include <exaDEM/stl_meshes.h>

#include <exaDEM/interaction.h>

#include <mpi.h>

namespace exaDEM
{
	using namespace exanb;
	
	
	
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
	
	
	template<	class GridT, class = AssertGridHasFields< GridT >> class BuildGridSTLMeshOperator : public OperatorNode
	{
		using ComputeFields = FieldSet< field::_rx ,field::_ry ,field::_rz, field::_vx ,field::_vy ,field::_vz, field::_vrot, field::_radius , field::_fx ,field::_fy ,field::_fz, field::_mass, field::_mom, field::_friction >;
		static constexpr ComputeFields compute_field_set {};
		ADD_SLOT( MPI_Comm , mpi      , INPUT , MPI_COMM_WORLD , DocString{"MPI communicator for parallel processing."});
		ADD_SLOT( GridT    , grid     , INPUT_OUTPUT , DocString{"Grid used for computations."} );
		ADD_SLOT( double   , rcut_max , INPUT , 0.0, DocString{"Maximum cutoff radius for computations. Default is 0.0."} );
		// ADD_SLOT( onika::memory::CudaMMVector< exaDEM::stl_mesh > , stl_collection, INPUT_OUTPUT , DocString{"Collection of meshes from stl files"});
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
			//auto& collection = *stl_collection;
			auto& mesh= *meshes;
			const double rad = *rcut_max;

			const auto cells = grid->cells();
			const size_t n_cells = grid->number_of_cells(); // nbh.size();
			const IJK dims = grid->dimension();
			const int gl = grid->ghost_layers();
			
			Interactions& I = *Int;
			I.reset();
			
			
			
			//for(auto &mesh : collection)
			//{
				//auto& ind = mesh.indexes;
				auto& ind2 = mesh.indexes2;
				auto& obb_faces = mesh.m_obbs;
				ind2.resize(n_cells);
				//mesh.build_boxes();
				mesh.build_obbs();
				
				std::vector< std::vector<Interaction_Particle>> interactions_cells;
				std::vector< std::vector<bool>> interactions_indexes;
								
				//interactions_cells.resize(n_cells);
				//interactions_particles.resize(n_cells);
				interactions_cells.resize(n_cells);
				interactions_indexes.resize(n_cells);
				
				std::vector< int > faces;

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
							const int n_particles = cells[cell_a].size();
							const uint64_t* __restrict__ id_a = cells[cell_a][ field::id ]; ONIKA_ASSUME_ALIGNED(id_a);
							const auto* __restrict__ rx_a = cells[cell_a][ field::rx ]; ONIKA_ASSUME_ALIGNED(rx_a);
							const auto* __restrict__ ry_a = cells[cell_a][ field::ry ]; ONIKA_ASSUME_ALIGNED(ry_a);
							const auto* __restrict__ rz_a = cells[cell_a][ field::rz ]; ONIKA_ASSUME_ALIGNED(rz_a);
							const auto* __restrict__ radius_a = cells[cell_a][ field::radius ]; ONIKA_ASSUME_ALIGNED(radius_a);
							interactions_cells[cell_a].resize(n_particles);
							interactions_indexes[cell_a].resize(n_particles);
							std::vector<int> faces_idx;
							for(int particle = 0; particle < n_particles; particle++){
								bool add_particle = false;
								faces_idx.resize(0);
								//std::vector< int > faces_idx;
								interactions_indexes[cell_a][particle] = false;
								for(int j = 0; j < ind2[cell_a].size(); j++){
									int idx = ind2[cell_a][j];
									OBB& obbface = obb_faces[idx];
									OBB sphere = sphere_to_obb(rx_a[particle], ry_a[particle], rz_a[particle], radius_a[particle]);
									//obbface.enlarge(*rcut_inc);
									sphere.enlarge(*rcut_inc);
									if(obbface.intersect(sphere))
									{
										add_particle = true;
										auto it = std::find(faces.begin(), faces.end(), idx);
										if(it==faces.end()){ 
											faces.push_back(idx);
											faces_idx.push_back(faces.size() - 1);
										} else {
											int index = std::distance(faces.begin(), it);
											faces_idx.push_back(index);
										}
									}
								}
								if(add_particle){
									interactions_cells[cell_a][particle]= {particle, cell_a, faces_idx};
									interactions_indexes[cell_a][particle] = true;
								}
							}
						}
						
						
						
					}
				}
				GRID_OMP_FOR_END
				
				int nf = 0;
				std::vector<double> nx;
				std::vector<double> ny;
				std::vector<double> nz;
				std::vector<double> offsets;
				std::vector<int> num_vertices;
				std::vector<int> start;
				std::vector<int> end;
				std::vector<double> vx;
				std::vector<double> vy;
				std::vector<double> vz;
				
				for(auto idx : faces){
					nx.push_back(mesh.nx[idx]);
					ny.push_back(mesh.ny[idx]);
					nz.push_back(mesh.nz[idx]);
					offsets.push_back(mesh.offsets[idx]);
					num_vertices.push_back(mesh.nb_vertices[idx]);
					if(start.size() == 0){ 
						start.push_back(0);
					} else {
						start.push_back(end.back());
					}
					end.push_back(start.back() + num_vertices.back());
					
					for(int i = mesh.start[idx]; i < mesh.end[idx]; i++){
						vx.push_back(mesh.vx[i]);
						vy.push_back(mesh.vy[i]);
						vz.push_back(mesh.vz[i]);	
					}
					nf++;
				}
				
				
				for(int i = 0; i < interactions_cells.size(); i++){
					for(int j = 0; j < interactions_cells[i].size(); j++){
						if(interactions_indexes[i][j]){
							I.add_particle(interactions_cells[i][j]);
						}
					}
				}
				
				I.add_mesh(nf, nx, ny, nz, offsets, start, end, num_vertices, vx, vy, vz);
				
			}
		};
	//};

	// this helps older versions of gcc handle the unnamed default second template parameter
	template <class GridT> using BuildGridSTLMeshOperatorTemplate = BuildGridSTLMeshOperator<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "build_grid_stl_mesh", make_grid_variant_operator< BuildGridSTLMeshOperatorTemplate > );
	}
};
