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
		, class = AssertGridHasFields< GridT, field::_radius, field::_shape>
		>
		class SetMultipleRadius : public OperatorNode
		{
			ADD_SLOT( GridT , grid  , INPUT_OUTPUT );
			ADD_SLOT( std::vector<double>, radius , INPUT, REQUIRED, DocString{"Array of radius values"});

			public:

			// -----------------------------------------------
			// ----------- Operator documentation ------------
			inline std::string documentation() const override final
			{
				return R"EOF(
        This operator applies various radius according to their material properties.
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
						double* __restrict__ m = cells[i][field::mass];
						double* __restrict__ r = cells[i][field::radius];
						uint32_t* __restrict__ myType = cells[i][field::shape];
						const double* rad 	= (*radius).data();
						const size_t n = cells[i].size();
#         pragma omp simd
						for(size_t j=0;j<n;j++)
						{
							r[j] = rad[myType[j]]; // 4/3 * pi * r^3 * d 
						}
					}
					GRID_OMP_FOR_END
				}
			}

		};

	template<class GridT> using SetMultipleRadiusTmpl = SetMultipleRadius<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "set_radius_multiple_materials", make_grid_variant_operator< SetMultipleRadiusTmpl > );
	}

}

