//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/compute/compute_cell_particles.h>
#include <exaDEM/set_fields.h>


namespace exaDEM
{
	using namespace exanb;
	template<typename GridT
		, class = AssertGridHasFields< GridT, field::_fx, field::_fy, field::_fz, field::_mom >
		>
		class ResetForceMomentNode : public OperatorNode
		{
			ADD_SLOT( GridT , grid  , INPUT_OUTPUT );

			static inline constexpr FieldSet<field::_fx, field::_fy, field::_fz, field::_mom> compute_field_set = {};

			public:

			inline std::string documentation() const override final
			{
				return R"EOF(
        This operator resets two grid fields : moments and forces.
        )EOF";
			}

			inline void execute () override final
			{
				//ResetForceMomentFunctor func = {};
				auto default_values = std::make_tuple(double(0.0), double(0.0), double(0.0), Vec3d{0.0,0.0,0.0});
				setFunctor<double,double, double, Vec3d> func = {default_values};
				compute_cell_particles( *grid , false , func , compute_field_set , gpu_execution_context() , gpu_time_account_func() );
			}
		};

	template<class GridT> using ResetForceMomentNodeTmpl = ResetForceMomentNode<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "reset_force_moment", make_grid_variant_operator< ResetForceMomentNodeTmpl > );
	}

}

