//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/quaternion_operators.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/compute/compute_cell_particles.h>
#include <memory>

#include <exaDEM/angular_acceleration.h>

namespace exaDEM
{
	using namespace exanb;

	template<typename GridT
		, class = AssertGridHasFields< GridT, field::_orient, field::_mom, field::_vrot, field::_arot, field::_inertia >
		>
		class PushToAngularAcceleration : public OperatorNode
		{
			// attributes processed during computation
			using ComputeFields = FieldSet< field::_orient, field::_mom, field::_vrot, field::_arot, field::_inertia >;
			static constexpr ComputeFields compute_field_set {};

			ADD_SLOT( GridT  , grid     , INPUT_OUTPUT );

			public:
			inline std::string documentation() const override final
			{
				return R"EOF(
        This operator computes the new values of angular acceleration from moments, orientations, angular velocities, angular accelerations and inertia.
        )EOF";
			}

			inline void execute () override final
			{
				PushToAngularAccelerationFunctor func {};
				compute_cell_particles( *grid , false , func , compute_field_set , gpu_execution_context() , gpu_time_account_func() );
			}
		};

	template<class GridT> using PushToAngularAccelerationTmpl = PushToAngularAcceleration<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "push_to_angular_acceleration", make_grid_variant_operator< PushToAngularAccelerationTmpl > );
	}
}

