
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <memory>
#include <exaDEM/shape/shapes.hpp>

namespace exaDEM
{
using namespace exanb;

  template<typename GridT
    , class = AssertGridHasFields< GridT, field::_inertia, field::_radius, field::_mass>
    >
  class PolyhedraUpdateInertia : public OperatorNode
  {
    ADD_SLOT( GridT , grid  , INPUT_OUTPUT );
		ADD_SLOT( shapes  , shapes_collection, INPUT_OUTPUT , DocString{"Collection of shapes"});

		public:
		// -----------------------------------------------
		// ----------- Operator documentation ------------
		inline std::string documentation() const override final
		{
			return R"EOF(
        This operator updates the inertia field.
        )EOF";
		}

		inline void execute () override final
		{
			auto cells = grid->cells();
			const IJK dims = grid->dimension();
			auto& sphs = *shapes_collection; 

#     pragma omp parallel
			{
				GRID_OMP_FOR_BEGIN(dims,i,loc, schedule(dynamic) )
				{
					auto* __restrict__ m = cells[i][field::mass];
					auto* __restrict__ inertia = cells[i][field::inertia];
					uint8_t* __restrict__ t = cells[i][field::type];
					const size_t n = cells[i].size();
#         pragma omp simd
					for(size_t j=0;j<n;j++)
					{

						inertia[j] = m[j] * sphs[t[j]]->get_Im();
					}
				}
				GRID_OMP_FOR_END
			}
		}

	};

	template<class GridT> using PolyhedraUpdateInertiaTmpl = PolyhedraUpdateInertia<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "polyhedra_update_inertia", make_grid_variant_operator< PolyhedraUpdateInertiaTmpl > );
	}
}

