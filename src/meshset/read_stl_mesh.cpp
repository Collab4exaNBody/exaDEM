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
		ADD_SLOT( std::vector<exaDEM::stl_mesh> , stl_collection, INPUT_OUTPUT , DocString{"Collection of meshes from stl files"});

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
			mesh.build_boxes();
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
