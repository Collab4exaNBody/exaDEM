//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/compute/compute_cell_particles.h>
#include <memory>
#include <exaDEM/set_fields.h>

namespace exaDEM
{
	using namespace exanb;

	template<typename GridT
		, class = AssertGridHasFields< GridT, field::_orient>
		>
		class SetQuaternion : public OperatorNode
		{
			static constexpr Quaternion default_quaternion = {0.0,0.0,0.0,1.0}; // impossible : Quaternion{0.0,0.0,0.0,1.0}
			using ComputeFields = FieldSet< field::_orient>;
			static constexpr ComputeFields compute_field_set {};

		ADD_SLOT( GridT  		, grid  , INPUT_OUTPUT );
		ADD_SLOT( Quaternion  , quat  	, INPUT , default_quaternion	, DocString{"Quaternion value for all particles"} );

			public:

		inline std::string documentation() const override final
		{
			return R"EOF(
        This operator sets the radius value for every particles.
        )EOF";
		}

		inline void execute () override final
		{
			std::tuple<Quaternion> default_values = std::make_tuple(*quat);
			setFunctor<Quaternion> func = {default_values};
			compute_cell_particles( *grid , false , func , compute_field_set , gpu_execution_context() , gpu_time_account_func() );
		}
		};

	template<class GridT> using SetQuaternionTmpl = SetQuaternion<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "set_quaternion", make_grid_variant_operator< SetQuaternionTmpl > );
	}

}

