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
		, class = AssertGridHasFields< GridT, field::_shape, field::_radius, field::_mass, field::_orient>
		>
		class SetMaterialProperties : public OperatorNode
		{
			static constexpr double default_radius = 0.5;
			static constexpr double default_density = 1;
			static constexpr Quaternion default_quaternion = {0.0,0.0,0.0,1.0}; // impossible : Quaternion{0.0,0.0,0.0,1.0}
		using ComputeFields = FieldSet<field::_shape, field::_radius, field::_mass, field::_orient>;
		static constexpr ComputeFields compute_field_set {};

		ADD_SLOT( GridT  		, grid  , INPUT_OUTPUT );
		ADD_SLOT( uint8_t  		, type  	, INPUT , REQUIRED 	, DocString{"type of particle to setialize"} );
		ADD_SLOT( double  		, rad  	, INPUT , default_radius	, DocString{"default radius value is 0.5 for all particles"} );
		ADD_SLOT( double  		, density  	, INPUT , default_density	, DocString{"default density value is 0 for all particles"} );
		ADD_SLOT( Quaternion  	, quat 	, INPUT , default_quaternion	, DocString{"default quaternion value for all particles "} );

			public:

		inline std::string documentation() const override final
		{
			return R"EOF(
        This operator sets material properties, ie radius, denstiy and quaternion values.
        )EOF";
		}

		inline void execute () override final
		{
			// compute mass
			const double d 	= (*density);
			const double r 	= (*rad);
			const double pi 	= 4*std::atan(1);
			const double coeff	= ((4.0)/(3.0)) * pi * d; 	 
			const double mass = coeff * r * r * r; // 4/3 * pi * r^3 * d 
			std::tuple<double, double, Quaternion> default_values = std::make_tuple(r,mass,*quat);
			FilteredSetFunctor<double, double, Quaternion> func { *type, default_values};
			compute_cell_particles( *grid , false , func , compute_field_set , gpu_execution_context() , gpu_time_account_func() );
		}
		};

	template<class GridT> using SetMaterialPropertiesTmpl = SetMaterialProperties<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "set_material_properties", make_grid_variant_operator< SetMaterialPropertiesTmpl > );
	}
}
