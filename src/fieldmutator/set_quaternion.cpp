//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

#include <memory>
#include <random>
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/compute/compute_cell_particles.h>
#include <exanb/grid_cell_particles/particle_region.h>
#include <exaDEM/set_fields.h>


namespace exaDEM
{
	using namespace exanb;

	template<typename GridT, class = AssertGridHasFields< GridT, field::_orient> >
		class SetQuaternion : public OperatorNode
	{
		static constexpr Quaternion default_quaternion = {0.0,0.0,0.0,1.0}; 
		using ComputeFields = FieldSet<field::_orient>;
		using ComputeRegionFields = FieldSet<field::_rx, field::_ry, field::_rz, field::_id, field::_orient>;
		static constexpr ComputeFields compute_field_set {};
		static constexpr ComputeRegionFields compute_region_field_set {};

		ADD_SLOT( GridT             , grid             , INPUT_OUTPUT );
		ADD_SLOT( bool              , random           , INPUT , false	, DocString{"This option generates random orientations for each particle"} );
		ADD_SLOT( Quaternion        , quat             , INPUT , default_quaternion	, DocString{"Quaternion value for all particles"} );
		ADD_SLOT( ParticleRegions   , particle_regions , INPUT , OPTIONAL );
		ADD_SLOT( ParticleRegionCSG , region           , INPUT , OPTIONAL );

		public:

		inline std::string documentation() const override final
		{
			return R"EOF(
        This operator sets the orientation value for every particles.
        )EOF";
		}

		inline void execute () override final
		{
			if( region.has_value() )
			{
				if( !particle_regions.has_value() )
				{
					fatal_error() << "Region is defined, but particle_regions has no value" << std::endl;
				}

        if( region->m_nb_operands==0 )
        {
          ldbg << "rebuild CSG from expr "<< region->m_user_expr << std::endl;
          region->build_from_expression_string( particle_regions->data() , particle_regions->size() );
        }

				ParticleRegionCSGShallowCopy prcsg = *region;
				SetRegionFunctor<Quaternion> func = { prcsg, *quat };
				compute_cell_particles( *grid , false , func , compute_region_field_set , parallel_execution_context() );
			}
			else
			{
				SetFunctor<Quaternion> func = {*quat };
				compute_cell_particles( *grid , false , func , compute_field_set , parallel_execution_context() );
			}
		}
	};

	template<class GridT> using SetQuaternionTmpl = SetQuaternion<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "set_quaternion", make_grid_variant_operator< SetQuaternionTmpl > );
	}

}

