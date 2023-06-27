#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <memory>
#include <random>

namespace exaDEM
{
using namespace exanb;

  template<typename GridT
    , class = AssertGridHasFields< GridT, field::_vx, field::_vy, field::_vz >
    >
  class SetRandVelocity : public OperatorNode
  {
    ADD_SLOT( GridT , grid  , INPUT_OUTPUT );
    ADD_SLOT( double, var , INPUT, 0, DocString{"Variance (same for all dimensions)"});
    ADD_SLOT( Vec3d, mean , INPUT, Vec3d{0,0,0}, DocString{"Average vector value."});

		public:

		inline std::string documentation() const override final
		{
			return R"EOF(
        This operator generates random velocities using a normal distribution law (var[double], mean[vec3d]).
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
					std::normal_distribution<> dist(0, *var);
					const auto* __restrict__ id = cells[i][field::id];
					double* __restrict__ vx = cells[i][field::vx];
					double* __restrict__ vy = cells[i][field::vy];
					double* __restrict__ vz = cells[i][field::vz];
					const size_t n = cells[i].size();
#         pragma omp simd
					for(size_t j=0;j<n;j++)
					{
						std::default_random_engine seed;
						seed.seed(id[j]); // TODO : Warning
						vx[j] = (*mean).x + dist(seed);
						seed.seed(id[j]+1); // TODO : Warning
						vy[j] = (*mean).y + dist(seed);
						seed.seed(id[j]+2); // TODO : Warning
						vz[j] = (*mean).z + dist(seed);
					}
				}
				GRID_OMP_FOR_END
			}
		}
	};

	template<class GridT> using SetRandVelocityTmpl = SetRandVelocity<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "set_rand_velocity", make_grid_variant_operator< SetRandVelocityTmpl > );
	}
}

