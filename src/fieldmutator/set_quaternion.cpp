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
#include <memory>
#include <random>


namespace exaDEM
{
	using namespace exanb;

	template<typename GridT
		, class = AssertGridHasFields< GridT, field::_orient>
		>
		class SetQuaternion : public OperatorNode
		{
			static constexpr Quaternion default_quaternion = {0.0,0.0,0.0,1.0}; // impossible : Quaternion{0.0,0.0,0.0,1.0}
			using ComputeFields = FieldSet< field::_orient>;
			static constexpr ComputeFields compute_field_set {};

		ADD_SLOT( GridT  		, grid  , INPUT_OUTPUT );
		ADD_SLOT( bool  , random  	, INPUT , false	, DocString{"This option generates random orientations for each particle"} );
		ADD_SLOT( Quaternion  , quat  	, INPUT , default_quaternion	, DocString{"Quaternion value for all particles"} );

			public:

		inline std::string documentation() const override final
		{
			return R"EOF(
        This operator sets the radius value for every particles.
        )EOF";
		}

		inline void execute () override final
		{
			if( ! (*random) )
			{
				SetFunctor<Quaternion> func = { {*quat} };
				compute_cell_particles( *grid , false , func , compute_field_set , parallel_execution_context() );
			}
			else
			{
				struct jammer
				{
					inline ONIKA_HOST_DEVICE_FUNC Quaternion& operator()(Quaternion& quat)
					{
						quat.x = m_dist(m_seed);
						quat.y = m_dist(m_seed); 
						quat.z = m_dist(m_seed);
						return quat;
					}
					std::uniform_real_distribution<double> m_dist;
					std::default_random_engine m_seed;
				};

				std::uniform_real_distribution<double> dist(-100, 100);
				std::default_random_engine seed;
				seed.seed(0); // TODO : Warning

				jammer processing {dist, seed};
				SetFunctorWithProcessing<jammer, Quaternion> func = { {processing}, {*quat}};
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

