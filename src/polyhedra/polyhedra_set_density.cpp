#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <memory>
#include <random>
#include <exaDEM/shapes.hpp>


namespace exaDEM
{
	using namespace exanb;

	template<typename GridT
		, class = AssertGridHasFields< GridT, field::_radius, field::_mass>
		>
		class PolyhedraSetDensity : public OperatorNode
		{
			ADD_SLOT( GridT , grid  , INPUT_OUTPUT );
			ADD_SLOT( double, density , INPUT, 1, DocString{"density value applied to all particles"});
			ADD_SLOT( shapes  , shapes_collection, INPUT_OUTPUT , DocString{"Collection of shapes"});

			// -----------------------------------------------
			// ----------- Operator documentation ------------
			inline std::string documentation() const override final
			{
				return R"EOF(
        This operator applies the same density to all particles. If you want to apply various densities according to their material properties, use set_densities_multiple_materials.
        )EOF";
			}

			public:
			inline void execute () override final
			{
				auto cells = grid->cells();
				const IJK dims = grid->dimension();
				const shapes shps = *shapes_collection;
				const double d = *density;
#     pragma omp parallel
				{
					GRID_OMP_FOR_BEGIN(dims,i,loc, schedule(dynamic) )
					{
						double* __restrict__ m = cells[i][field::mass];
						uint8_t* __restrict__ t = cells[i][field::type];
						const size_t n = cells[i].size();
#         pragma omp simd
						for(size_t j=0;j<n;j++)
						{
							m[j] = d * shps[t[j]]->get_volume();
						}
					}
					GRID_OMP_FOR_END
				}
			}

		};

	template<class GridT> using PolyhedraSetDensityTmpl = PolyhedraSetDensity<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "polyhedra_set_density", make_grid_variant_operator< PolyhedraSetDensityTmpl > );
	}

}

