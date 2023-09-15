#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/compute/compute_cell_particles.h>
#include <exanb/core/grid.h>
#include <memory>
#include <exaDEM/set_fields.h>

namespace exaDEM
{
	using namespace exanb;

	template<typename GridT
		, class = AssertGridHasFields< GridT, field::_homothety >
		>
		class SetHomothety : public OperatorNode
		{
			static constexpr double default_h = 0.0;
			// attributes processed during computation
			using ComputeFields = FieldSet< field::_homothety>;
			static constexpr ComputeFields compute_field_set {};
			ADD_SLOT( GridT , grid  , INPUT_OUTPUT );
			ADD_SLOT( double, homothety , INPUT, default_h, DocString{"homoethety value"});

			public:

			// -----------------------------------------------
			// ----------- Operator documentation ------------
			inline std::string documentation() const override final
			{
				return R"EOF(
        This operator sets the same homothetu value to all particles.
        )EOF";
			}

			inline void execute () override final
			{
//				std::tuple<double> default_values = std::make_tuple(*homothety);
				SetFunctor<double> func = { {*homothety} };
				compute_cell_particles( *grid , false , func , compute_field_set , parallel_execution_context() );
			}
		};

	template<class GridT> using SetHomothetyTmpl = SetHomothety<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "set_homothety", make_grid_variant_operator< SetHomothetyTmpl > );
	}

}

