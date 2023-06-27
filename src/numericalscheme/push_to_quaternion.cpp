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

#include <exaDEM/push_to_quaternion.h>

namespace exaDEM
{
using namespace exanb;

  template<typename GridT
    , class = AssertGridHasFields< GridT, field::_orient, field::_vrot, field::_arot >
    >
  class PushToQuaternion : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet< field::_orient, field::_vrot, field::_arot >;
    static constexpr ComputeFields compute_field_set {};

    ADD_SLOT( GridT  , grid , INPUT_OUTPUT );
    ADD_SLOT( double , dt   , INPUT, DocString{"dt is the time increment of the timeloop"});

  public:

		inline std::string documentation() const override final
		{
			return R"EOF(
        This operator computes particle quaternion values from orientation, angular velocities and angular accelerations. 
        )EOF";
		}

    inline void execute () override final
    {
      const double dt = *(this->dt);
      const double dt_2 = 0.5 * dt;
      const double dt2_2 = dt_2 * dt;
      PushToQuaternionFunctor func {dt, dt_2, dt2_2};
      compute_cell_particles( *grid , false , func , compute_field_set , gpu_execution_context() , gpu_time_account_func() );
    }
  };
  
  template<class GridT> using PushToQuaternionTmpl = PushToQuaternion<GridT>;
  
 // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "push_to_quaternion", make_grid_variant_operator< PushToQuaternionTmpl > );
  }
}
