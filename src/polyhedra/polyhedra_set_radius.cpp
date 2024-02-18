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
		, class = AssertGridHasFields< GridT, field::_radius>
		>
		class PolyhedraDefineRadius : public OperatorNode
		{
			ADD_SLOT( GridT , grid  , INPUT_OUTPUT );
			ADD_SLOT( shapes  , shapes_collection, INPUT_OUTPUT , DocString{"Collection of shapes"});
			ADD_SLOT( double                , rcut_max          , INPUT_OUTPUT , 0.0 );
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
				const size_t size = shps.get_size();
				std::vector<double> r;
				r.resize(size);
				double rmax=0;
				for(size_t i = 0 ; i < size ; i++)
				{
					double rad_max = shps[i]->compute_max_rcut();
					r[i] = rad_max; 
					rmax = std::max(rmax, 2 *  rad_max); // r * maxrcut
				}
			*rcut_max = rmax; 

#     pragma omp parallel
				{
					GRID_OMP_FOR_BEGIN(dims,i,loc, schedule(dynamic) )
					{
						double* __restrict__ rad = cells[i][field::radius];
						uint8_t* __restrict__ t = cells[i][field::type];
						const size_t n = cells[i].size();
#         pragma omp simd
						for(size_t j=0;j<n;j++)
						{
							rad[j] = r[t[j]]; 
						}
					}
					GRID_OMP_FOR_END
				}
			}

		};

	template<class GridT> using PolyhedraDefineRadiusTmpl = PolyhedraDefineRadius<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "polyhedra_define_radius", make_grid_variant_operator< PolyhedraDefineRadiusTmpl > );
	}

}

