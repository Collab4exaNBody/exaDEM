

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
#include <exaDEM/stl_mesh.h>

#include <mpi.h>

#include <onika/cuda/cuda.h> // mots cles specifiques
#include <onika/memory/allocator.h> // cudaMMVector

namespace exaDEM
{
	using namespace exanb;
	template<	class GridT, class = AssertGridHasFields< GridT >> class ReadSTLOperator : public OperatorNode
	{
		using ComputeFields = FieldSet< field::_rx ,field::_ry ,field::_rz>;
		static constexpr ComputeFields compute_field_set {};
		ADD_SLOT( MPI_Comm , mpi      , INPUT , MPI_COMM_WORLD);
		ADD_SLOT( GridT    , grid     , INPUT_OUTPUT );
		ADD_SLOT( Domain   , domain   , INPUT , REQUIRED );
		ADD_SLOT( std::string , filename, INPUT , REQUIRED , DocString{"Inpute filename"});
		//ADD_SLOT( std::vector<exaDEM::stl_mesh> , stl_collection, INPUT_OUTPUT , DocString{"Collection of meshes from stl files"});
		ADD_SLOT( onika::memory::CudaMMVector< exaDEM::stl_mesh > , stl_collection, INPUT_OUTPUT , DocString{"Collection of meshes from stl files"});
		ADD_SLOT( stl_meshes, meshes, INPUT_OUTPUT, DocString{"Meshes"});

		public:
		inline std::string documentation() const override final
		{
			return R"EOF( This operator initialize a mesh composed of faces from an stl input file.
    	    			)EOF";
		}

		inline void execute () override final
		{
			auto& collection = *stl_collection;
			stl_mesh mesh; 
			mesh.read_stl(*filename);
			
			auto& m = *meshes;
			m.read2_stl(*filename);
			printf("VERIFICATION READ_STL\n");
			int j = 1;
			int i=0;
			bool breaking = true;
			int size = mesh.m_data.size();
			onika::memory::CudaMMVector< Vec3d > vert = mesh.m_data[size - 1].vertices;
			//printf("VERIFICATION SIZE: %d\n", vert.size());
			
			//for(int i = 0; i < mesh.m_data.size(); i++){
			while(breaking && i < mesh.m_data.size()){
				//printf("IIIII: %d\n", i);
				//i++; 
				
				onika::memory::CudaMMVector< Vec3d > vertices = mesh.m_data[i].vertices;
				std::vector<Vec3d> vertices2;
				int nb_Vertices = m.m_meshes[m.nb_meshes - 1][j];
				j+=5;
				int j_2 = j;
				for(int z = j_2; z < j_2 + nb_Vertices*3; z+=3){
					//printf("ZZZZZ: %d\n", z);
					Vec3d v = {m.m_meshes[m.nb_meshes - 1][z], m.m_meshes[m.nb_meshes - 1][z + 1], m.m_meshes[m.nb_meshes - 1][z + 2]};
					vertices2.push_back(v);
					j+=3;
				}
				if(vertices.size() != vertices2.size()){ printf("VERTICES1: %d, VERTICES2: %d\n", vertices.size(), vertices2.size()); breaking = false; printf("ICI1\n");}
				else {
					int x = 0;
					while(breaking && x < vertices.size()){
						//for(int x=0; x< vertices.size(); x++){
							x++;
							Vec3d v1 = vertices[j];
							Vec3d v2 = vertices[j];
							if(v1.x != v2.x || v1.y != v2.y || v1.z != v2.z){ breaking=false; printf("ICI2\n");}
						//}
					}
				}
				vertices.clear();
				vertices2.clear();
				i++;
			}
			if(breaking){ printf("LE MESH EST BON\n");}
			else { printf("LE MESH EST PAS BON\n");}
			//m.
			//printf("OUI\n");
			int taille = m.nb_meshes - 1;
			m.build2_boxes(taille);
			//auto exec_ctx = parallel_execution_context();
			//bool gpu_present = exec_ctx != nullptr
			//	&& exec_ctx->has_gpu_context()
			//	&& exec_ctx->gpu_context()->has_devices();
			//if( gpu_present ) {
			//	int NbBlocks = 128;
			//	int BlockSize = 32;
			//	ONIKA_CU_LAUNCH_KERNEL( NbBlocks, BlockSize
			//		, 0, exec_ctx->gpu_stream()
			//		, mesh.build_boxes_kernel);
			//}
			//else {
			mesh.build_boxes();
			printf("VERIFICATION BOXES\n");
			i =0;
			if(mesh.m_boxes.size() != m.m_boxes[m.nb_meshes - 1].size()) breaking = false;
			printf("BOOOOOX1: %d EEEEEET BOOOOOOX2: %d\n", mesh.m_boxes.size(), m.m_boxes[m.nb_meshes - 1].size());
			while(breaking && i < mesh.m_boxes.size()){
				printf("IBOXIBOXI: %d\n", i);
				Box b1 = mesh.m_boxes[i];
				Box b2 = m.m_boxes[m.nb_meshes - 1][i];
				Vec3d inf1 = b1.inf;
				Vec3d sup1 = b1.sup;
				Vec3d inf2 = b2.inf;
				Vec3d sup2 = b2.sup;
				if(inf1.x != inf2.x || inf1.y != inf2.y || inf1.z != inf2.z){ breaking = false; printf("INFFFFFFF\n");} 
				if(sup1.x != sup2.x || sup1.y != sup2.y || sup1.z != sup2.z){ breaking = false; printf("SUUUUUUUUUP\n");}
				i++;
			}
			if(breaking){ printf("LES BOX C'EST BON\n");}
			else{ printf("LES BOX C'EST PAS BON\n");}
			//for(int i = 0; i < mesh.m_boxes.size(); i++){
			
			//mesh.build_obbs();
			//}
			
			collection.push_back(mesh);
		};
	};

	// this helps older versions of gcc handle the unnamed default second template parameter
	template <class GridT> using ReadSTLOperatorTemplate = ReadSTLOperator<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "read_stl", make_grid_variant_operator< ReadSTLOperatorTemplate > );
	}
}
