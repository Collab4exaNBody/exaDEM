//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE
#include "exanb/core/operator.h"
#include "exanb/core/operator_slot.h"
#include "exanb/core/operator_factory.h"
#include "exanb/core/make_grid_variant_operator.h"
#include "exanb/core/parallel_grid_algorithm.h"
#include "exanb/core/grid.h"
#include "exanb/core/domain.h"
#include "exanb/compute/compute_cell_particles.h"
#include <mpi.h>
#include <memory>
#include <exaDEM/driver_base.h>
#include <exaDEM/drivers.h>
#include <exaDEM/surface.h>

namespace exaDEM
{

  using namespace exanb;

  template<typename GridT>
    class AddSurface : public OperatorNode
    {
      ADD_SLOT( Drivers , drivers   , INPUT_OUTPUT               , DocString{"List of Drivers"});
      ADD_SLOT( int     , id        , INPUT , REQUIRED           , DocString{"Driver index"});
			ADD_SLOT( double  , offset    , INPUT , 0.0                , DocString{"Offset from the origin (0,0,0) of the rigid surface"} );
			ADD_SLOT( Vec3d   , velocity  , INPUT , Vec3d{0.0,0.0,0.0} , DocString{"Surface velocity"});
			ADD_SLOT( Vec3d   , normal    , INPUT , Vec3d{0.0,0.0,1.0} , DocString{"Normal vector of the rigid surface"});

			public:

			inline std::string documentation() const override final
			{
				return R"EOF(
        This operator add a surface to the drivers list.
        )EOF";
			}

			inline void execute () override final
			{
      	constexpr Vec3d null= { 0.0, 0.0, 0.0 };
				exaDEM::Surface driver = {*offset, *normal, null, *velocity, null}; // 
				driver.initialize();
				drivers->add_driver(*id, driver);
			}
		};

	template<class GridT> using AddSurfaceTmpl = AddSurface<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "add_surface", make_grid_variant_operator< AddSurfaceTmpl > );
	}
}

