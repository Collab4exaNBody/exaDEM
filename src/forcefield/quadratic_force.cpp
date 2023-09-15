//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <memory>
#include <exanb/compute/compute_cell_particles.h>
#include <exaDEM/quadratic_force.h>

namespace exaDEM
{
	using namespace exanb;

	template<typename GridT
		, class = AssertGridHasFields< GridT, field::_fx,field::_fy,field::_fz,  field::_vx, field::_vy, field::_vz >
		>
		class QuadraticForce : public OperatorNode
		{
			// attributes processed during computation
			using ComputeFields = FieldSet< field::_fx,field::_fy, field::_fz, field::_vx , field::_vy , field::_vz >;
			static constexpr ComputeFields compute_field_set {};

			ADD_SLOT( GridT  , grid     , INPUT_OUTPUT );
			ADD_SLOT( double  , cx  , INPUT , REQUIRED , DocString{"aerodynamic coefficient."});
			ADD_SLOT( double  , mu  , INPUT , REQUIRED , DocString{"drag coefficient. air = 0.000015"});

			public:

			inline std::string documentation() const override final
			{
				return R"EOF(
        This operator computes friction forces related to air or fluide.
        )EOF";
			}

			inline void execute () override final
			{
				QuadraticForceFunctor func { (*cx) * (*mu)};
				compute_cell_particles( *grid , false , func , compute_field_set , parallel_execution_context() );
			}
		};

	template<class GridT> using QuadraticForceTmpl = QuadraticForce<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "quadratic_force", make_grid_variant_operator< QuadraticForceTmpl > );
	}
}

