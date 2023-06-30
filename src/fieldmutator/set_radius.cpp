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
		, class = AssertGridHasFields< GridT, field::_radius>
		>
		class SetRadius : public OperatorNode
		{
			static constexpr double default_radius = 0.5;
		using ComputeFields = FieldSet< field::_radius>;
		static constexpr ComputeFields compute_field_set {};

		ADD_SLOT( GridT  		, grid  , INPUT_OUTPUT );
		ADD_SLOT( double  		, rad  	, INPUT , default_radius	, DocString{"default radius value for all particles"} );

			public:

		inline std::string documentation() const override final
		{
			return R"EOF(
        This operator sets the radius value for every particles.
        )EOF";
		}

		inline void execute () override final
		{
			std::tuple<double> default_values = std::make_tuple(*rad);
			setFunctor<double> func = {default_values};
			compute_cell_particles( *grid , false , func , compute_field_set , gpu_execution_context() , gpu_time_account_func() );
		}
		};

	template<class GridT> using SetRadiusTmpl = SetRadius<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "set_radius", make_grid_variant_operator< SetRadiusTmpl > );
	}

}

