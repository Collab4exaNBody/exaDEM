//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <memory>
#include <exaDEM/gravity_force.h>

namespace exaDEM
{
  using namespace exanb;

  template<typename GridT
    , class = AssertGridHasFields< GridT, field::_mass, field::_fx,field::_fy,field::_fz >
    >
  class GravityForce : public OperatorNode
  {
    static constexpr Vec3d default_gravity = { 0.0, 0.0, -9.807 };
    // attributes processed during computation
    using ComputeFields = FieldSet< field::_mass, field::_fx ,field::_fy ,field::_fz >;
    static constexpr ComputeFields compute_field_set {};

    ADD_SLOT( GridT  , grid     , INPUT_OUTPUT );
    ADD_SLOT( Vec3d  , gravity  , INPUT , default_gravity , DocString{"define the gravity constant in function of the gravity axis, default value are x axis = 0, y axis = 0 and z axis = -9.807"});

  public:

		inline std::string documentation() const override final
		{
			return R"EOF(
        This operator computes forces related to the gravity.
        )EOF";
		}

		inline void execute () override final
		{
			const Vec3d g = *gravity;
			GravityForceFunctor func { *gravity};
			compute_cell_particles( *grid , false , func , compute_field_set , gpu_execution_context() , gpu_time_account_func() );
		}
	};

	template<class GridT> using GravityForceTmpl = GravityForce<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "gravity_force", make_grid_variant_operator< GravityForceTmpl > );
	}

}

