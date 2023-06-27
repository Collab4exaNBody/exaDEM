
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <memory>

namespace exaDEM
{
using namespace exanb;

  template<typename GridT
    , class = AssertGridHasFields< GridT, field::_inertia, field::_radius, field::_mass>
    >
  class UpdateInertia : public OperatorNode
  {
    ADD_SLOT( GridT , grid  , INPUT_OUTPUT );

		public:
		// -----------------------------------------------
		// ----------- Operator documentation ------------
		inline std::string documentation() const override final
		{
			return R"EOF(
        This operator updates the inertia field (0.4*mass*radius*radius).
        )EOF";
		}

		inline void execute () override final
		{
			auto cells = grid->cells();
			const IJK dims = grid->dimension();

#     pragma omp parallel
			{
				GRID_OMP_FOR_BEGIN(dims,i,loc, schedule(dynamic) )
				{
					auto* __restrict__ r = cells[i][field::radius];
					auto* __restrict__ m = cells[i][field::mass];
					auto* __restrict__ inertia = cells[i][field::inertia];
					const size_t n = cells[i].size();
#         pragma omp simd
					for(size_t j=0;j<n;j++)
					{
						const double inertia_value = 0.4 * m[j] * r[j] * r[j];
						inertia[j] = {inertia_value, inertia_value, inertia_value}; 
					}
				}
				GRID_OMP_FOR_END
			}
		}

	};

	template<class GridT> using UpdateInertiaTmpl = UpdateInertia<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "update_inertia", make_grid_variant_operator< UpdateInertiaTmpl > );
	}
}

