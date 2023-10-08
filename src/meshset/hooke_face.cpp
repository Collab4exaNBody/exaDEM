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
#include <exaDEM/hooke_face.h>

#include <mpi.h>

namespace exaDEM
{
	using namespace exanb;
	template<
		class GridT,
					class = AssertGridHasFields< GridT, field::_fx, field::_fy, field::_fz, field::_friction >
						>
						class HookeFaceOperator : public OperatorNode
						{

							using ComputeFields = FieldSet< field::_rx ,field::_ry ,field::_rz, field::_vx ,field::_vy ,field::_vz, field::_vrot, field::_radius , field::_fx ,field::_fy ,field::_fz, field::_mass, field::_mom, field::_friction >;
							static constexpr ComputeFields compute_field_set {};
							ADD_SLOT( MPI_Comm , mpi      , INPUT , MPI_COMM_WORLD);
							ADD_SLOT( GridT  , grid    , INPUT_OUTPUT );
							ADD_SLOT( Domain , domain  , INPUT , REQUIRED );
							ADD_SLOT( std::vector<Vec3d> , verticies, INPUT , REQUIRED , DocString{"list of verticies"});
							ADD_SLOT( double  , dt                		, INPUT 	, REQUIRED 	, DocString{"Timestep of the simulation"});
							ADD_SLOT( double  , kt  			, INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
							ADD_SLOT( double  , kn  			, INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"} );
							ADD_SLOT( double  , kr  			, INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
							ADD_SLOT( double  , mu  			, INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
							ADD_SLOT( double  , damprate  			, INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"});

							public:
							inline std::string documentation() const override final
							{
								return R"EOF(
    	    			)EOF";
							}

							inline void execute () override final
							{
								Face face(*verticies);
								HookeFaceFunctor func {face, *dt, *kt, *kn, *kr, *mu, *damprate};
								compute_cell_particles( *grid , false , func , compute_field_set , parallel_execution_context() );
							}
						};


	// this helps older versions of gcc handle the unnamed default second template parameter
	template <class GridT> using HookeFaceOperatorTemplate = HookeFaceOperator<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "hooke_face", make_grid_variant_operator< HookeFaceOperatorTemplate > );
	}
}

