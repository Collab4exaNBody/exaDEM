//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/compute/compute_cell_particles.h>
#include <memory>

#include <exaDEM/angular_velocity.h>

namespace exaDEM
{
using namespace exanb;

  template<typename GridT
    , class = AssertGridHasFields< GridT, field::_vrot, field::_arot >
    >
  class PushToAngularVelocity : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet< field::_vrot, field::_arot >;
    static constexpr ComputeFields compute_field_set {};

    ADD_SLOT( GridT  , grid     , INPUT_OUTPUT );
    ADD_SLOT( double , dt       , INPUT, DocString{"dt is the time increment of the timeloop"});

  public:

		inline std::string documentation() const override final
		{
			return R"EOF(
        This operator computes particle angular velocitiy values from angular velocities and angular accelerations. 
        )EOF";
		}

    inline void execute () override final
    {
      const double dt = *(this->dt);
      const double dt_2 = 0.5 * dt;
      PushToAngularVelocityFunctor func {dt_2};
      compute_cell_particles( *grid , false , func , compute_field_set , gpu_execution_context() , gpu_time_account_func() );
    }
  };
  
  template<class GridT> using PushToAngularVelocityTmpl = PushToAngularVelocity<GridT>;
  
 // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "push_to_angular_velocity", make_grid_variant_operator< PushToAngularVelocityTmpl > );
  }

}

