//#pragma xstamp_cuda_enable

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

#include <exaDEM/stl_mesh.h>
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
		
	/** Kernel used to apply Hooke's law t particles interacting with multiple STL meshes within a Grid */
	template<class GridT>__global__ void ApplyHookeSTLMesh_GPU(GridT* cells, Interactions I, int size, double dt, double kt, double kn, double kr, double mu, double damprate){
		
		//Index of the particle in the interaction's list
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		
		if(idx < size){
			
			//Get the Interactions's structure attributes
			auto particules = onika::cuda::vector_data(I.p_i);
			auto cellules = onika::cuda::vector_data(I.cell_i);
			auto faces = onika::cuda::vector_data(I.faces);
			auto start_particle = onika::cuda::vector_data(I.start_particle);
			auto num_faces = onika::cuda::vector_data(I.num_faces);
			auto nx = onika::cuda::vector_data(I.nx);
			auto ny = onika::cuda::vector_data(I.ny);
			auto nz = onika::cuda::vector_data(I.nz);
			auto offsets = onika::cuda::vector_data(I.offsets);
			auto sf = onika::cuda::vector_data(I.start_face);
			auto ef = onika::cuda::vector_data(I.end_face);
			auto num_vertices = onika::cuda::vector_data(I.num_vertices);
			auto vertex_x = onika::cuda::vector_data(I.vertex_x);
			auto vertex_y = onika::cuda::vector_data(I.vertex_y);
			auto vertex_z = onika::cuda::vector_data(I.vertex_z);
			
			//Attributes associated to the particle
			int p_i = particules[idx];
			int cell_i = cellules[idx];
			int start = start_particle[idx];
			int nf = num_faces[idx];
			
			double rx = cells[cell_i][field::rx][p_i];
			double ry = cells[cell_i][field::ry][p_i];
			double rz = cells[cell_i][field::rz][p_i];
			double radius = cells[cell_i][field::radius][p_i];
			double vx = cells[cell_i][field::vx][p_i];
			double vy = cells[cell_i][field::vy][p_i];
			double vz = cells[cell_i][field::vz][p_i];
			Vec3d vrot = cells[cell_i][field::vrot][p_i];
			
			bool is_face = false;// If there is one contact with a face, we skip contact with edges
			bool do_edge = false;
			
			//contact face / sphere
			for(int i = 0 ; i < nf ; i++ ){
				
				//The face
				int index = faces[start + i];
				Vec3d normal = {nx[index], ny[index], nz[index]};
				double offset = offsets[index];
				//First vertice corresponding to the face int the vertices's list
				int s = sf[index];
				//Last vertice corresponding to the face in the vertices's list
				int e = ef[index];
				//Number of vertices for the correspodig face
				int vertices_size = num_vertices[index];
				
				bool contact = false;
				bool potential = false;
				Vec3d position = {0,0,0};
				
				

				//CONTACT FACE SPHERE (from the Face structure in face.h)
				const Vec3d center = {rx,ry,rz};
				double p = exanb::dot(center,normal) - offset;
				if(abs(p) <= radius)
				{

					potential = true;
					const Vec3d& pa = { vertex_x[s], vertex_y[s], vertex_z[s] };
					const Vec3d& pb = { vertex_x[s + 1], vertex_y[s + 1], vertex_z[s + 1] };
					const Vec3d& pc = { vertex_x[e - 1], vertex_y[e - 1], vertex_z[e - 1] };
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

					for (int iva = 0; iva < vertices_size ; ++iva) {
						int ivb = iva + 1;
						if (ivb == vertices_size) ivb = 0;
						const Vec3d& posNodeA_jv = { vertex_x[s + iva], vertex_y[s + iva], vertex_z[s + iva] };
						const Vec3d& posNodeB_jv = { vertex_x[s + ivb], vertex_y[s + ivb], vertex_z[s + ivb] };
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
						position = normal*offset;
						contact= true;
					}

				}
				//CONTACT FACE SPHERE END
				
				//If there is a contact we handle the interaction between the sphere and the face (from the ApplyHookeSTLMeshesFunctor in hooke_stl_meshes.h)
				if(contact)
				{	
					Vec3d mom = cells[cell_i][field::mom][p_i];
					Vec3d ft = cells[cell_i][field::friction][p_i];
					
					Vec3d pos_proj;
					double m_vel = 0;
					Vec3d pos = {rx ,ry ,rz};
					Vec3d vel = {vx ,vy ,vz};


					pos_proj = dot(pos, normal) * normal;
			
			

					Vec3d vec_n = pos_proj - position;
					double n = norm(vec_n);
					vec_n = vec_n / n;
					const double dn = n - radius;		
					Vec3d rigid_surface_center = position; 
					const Vec3d rigid_surface_velocity = normal * m_vel; 
					constexpr Vec3d rigid_surface_angular_velocity = {0.0,0.0,0.0};

					Vec3d f = {0.0,0.0,0.0};
					constexpr double meff = 1;
				
					exaDEM::hooke_force_core_v2(
						dn, vec_n,
						dt, kn, kt, kr, mu, damprate, meff,
						ft, position, pos_proj, vel, f, mom, vrot,
						rigid_surface_center, rigid_surface_velocity, rigid_surface_angular_velocity
						);
				
					cells[cell_i][field::mom][p_i]= mom;
					cells[cell_i][field::friction][p_i]= ft;
					cells[cell_i][field::fx][p_i]+= f.x;
					cells[cell_i][field::fy][p_i]+= f.y;
					cells[cell_i][field::fz][p_i]+= f.z;
					
					is_face = true;
					
				}
				
				do_edge = do_edge || potential;
				
			}
			
			
			if(is_face == false && do_edge){
				//contact edge / sphere
				for(int i = 0; i < nf; i++){
					
					int index = faces[start + i];
					bool contact = false;
					Vec3d position = {0, 0, 0};
					Vec3d normal = {nx[index], ny[index], nz[index]};
					int s = sf[index];
					int vertices_size = num_vertices[index];
					
					//CONTACT EDGE SPHERE (from the Face structure in face.h)
					const Vec3d center = {rx,ry,rz};
					for (size_t j = 0; j < vertices_size; ++j) {
						Vec3d p1 = { vertex_x[s + j], vertex_y[s + j], vertex_z[s + j] };
						Vec3d p2 = { vertex_x[s + ((j + 1) % vertices_size)], vertex_y[s + ((j + 1) % vertices_size)], vertex_z[s + ((j + 1) % vertices_size)] };
						Vec3d edge = p2 - p1;
						Vec3d sphereToEdge = center - p1;

						double distanceToEdge = length(exanb::cross(edge, sphereToEdge)) / length(edge);

						if (distanceToEdge <= radius && exanb::dot(sphereToEdge, edge) > 0 && exanb::dot(sphereToEdge - edge, edge) < 0) {
							auto n_edge = edge / exanb::norm(edge);
							Vec3d contact_position = p1 + n_edge * dot(sphereToEdge, n_edge);
							contact = true;
							position = contact_position;
						}
					}
					//CONTACT EDGE SPHERE END
					
					//If there is a contact we handle the interaction between the edge and the sphere (from the ApplyHookeSTLMeshesFunctor in hooke_stl_meshes.h)
					if(contact)
					{
						Vec3d mom = cells[cell_i][field::mom][p_i];
						Vec3d ft = cells[cell_i][field::friction][p_i];
						
						Vec3d pos_proj;
						double m_vel = 0;
						Vec3d pos = {rx ,ry ,rz};
						Vec3d vel = {vx ,vy ,vz};

						pos_proj = pos;
						Vec3d vec_n = pos_proj - position;
						double n = norm(vec_n);
						vec_n = vec_n / n;
						const double dn = n - radius;		
						Vec3d rigid_surface_center = position; 
						const Vec3d rigid_surface_velocity = normal * m_vel; 
						constexpr Vec3d rigid_surface_angular_velocity = {0.0,0.0,0.0};

						Vec3d f = {0.0,0.0,0.0};
						constexpr double meff = 1;
					
						exaDEM::hooke_force_core_v2(
						dn, vec_n,
						dt, kn, kt, kr, mu, damprate, meff,
						ft, position, pos_proj, vel, f, mom, vrot,
						rigid_surface_center, rigid_surface_velocity, rigid_surface_angular_velocity
						);
						
						cells[cell_i][field::mom][p_i]= mom;
						cells[cell_i][field::friction][p_i]= ft;
						cells[cell_i][field::fx][p_i]+= f.x;
						cells[cell_i][field::fy][p_i]+= f.y;
						cells[cell_i][field::fz][p_i]+= f.z;
					}
				}
			}
		}
	}
		
		
	template<
		class GridT,
					class = AssertGridHasFields< GridT, field::_fx, field::_fy, field::_fz, field::_friction >
						>
						class ApplyHookeSTLMeshesOperator : public OperatorNode
						{

							using ComputeFields = FieldSet< field::_rx ,field::_ry ,field::_rz, field::_vx ,field::_vy ,field::_vz, field::_vrot, field::_radius , field::_fx ,field::_fy ,field::_fz, field::_mass, field::_mom, field::_friction >;
							static constexpr ComputeFields compute_field_set {};
							ADD_SLOT( MPI_Comm , mpi      , INPUT , MPI_COMM_WORLD);
							ADD_SLOT( GridT  , grid    , INPUT_OUTPUT );
							ADD_SLOT( Domain , domain  , INPUT , REQUIRED );
							//ADD_SLOT( onika::memory::CudaMMVector< exaDEM::stl_mesh > , stl_collection, INPUT_OUTPUT , DocString{"list of verticies"});
							ADD_SLOT( double  , dt                		, INPUT 	, REQUIRED 	, DocString{"Timestep of the simulation"});
							ADD_SLOT( double  , kt  			, INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
							ADD_SLOT( double  , kn  			, INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"} );
							ADD_SLOT( double  , kr  			, INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
							ADD_SLOT( double  , mu  			, INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
							ADD_SLOT( double  , damprate  			, INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
							ADD_SLOT(Interactions, Int, INPUT_OUTPUT, DocString{"List of interactions between the particles and the faces of the STL meshes"});
							
 
							public:
							inline std::string documentation() const override final
							{
								return R"EOF(
    	    			)EOF";
							}
							

							inline void execute () override final
							{
								
								auto& g = *grid;
								const auto cells = g.cells();
								
								auto& I = *Int;
								
								int size = I.nb_particles;
								int blockSize = 32;
								int numBlocks;
								if(size % blockSize == 0){ numBlocks = size/blockSize;}
								else if(size / blockSize < 1) { numBlocks=1; blockSize = size;}
								else  { numBlocks= int(size/blockSize)+1; }
								
								
								ApplyHookeSTLMesh_GPU<<<numBlocks, blockSize>>>(cells, I, size, *dt, *kt, *kn, *kr, *mu, *damprate);
								
						}
					};


	// this helps older versions of gcc handle the unnamed default second template parameter
	template <class GridT> using ApplyHookeSTLMeshesOperatorTemplate = ApplyHookeSTLMeshesOperator<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "apply_hooke_stl_meshes", make_grid_variant_operator< ApplyHookeSTLMeshesOperatorTemplate > );
	}
};

