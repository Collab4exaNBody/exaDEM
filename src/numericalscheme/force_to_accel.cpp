//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/compute/compute_cell_particles.h>
#include <memory>
#include <exaDEM/force_to_accel.h>

namespace exaDEM
{
using namespace exanb;

  template<typename GridT
    , class = AssertGridHasFields< GridT, field::_mass, field::_fx,field::_fy,field::_fz >
    >
  class ForceToAccel : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet< field::_mass, field::_fx ,field::_fy ,field::_fz >;
    static constexpr ComputeFields compute_field_set {};

    ADD_SLOT( GridT  , grid     , INPUT_OUTPUT );

  public:

		inline std::string documentation() const override final
		{
			return R"EOF(
        This operator computes particle accelerations from forces and mass.
        )EOF";
		}

		inline void execute () override final
		{
			ForceToAccelFunctor func {};
			compute_cell_particles( *grid , false , func , compute_field_set , gpu_execution_context() , gpu_time_account_func() );
		}
	};

	template<class GridT> using ForceToAccelTmpl = ForceToAccel<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "force_to_accel", make_grid_variant_operator< ForceToAccelTmpl > );
	}

}

