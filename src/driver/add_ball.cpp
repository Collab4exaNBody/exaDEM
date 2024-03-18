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
#include <exaDEM/ball.h>

namespace exaDEM
{

  using namespace exanb;

  template<typename GridT>
    class AddBall : public OperatorNode
    {
      static constexpr Vec3d null= { 0.0, 0.0, 0.0 };

      ADD_SLOT( Drivers , drivers    , INPUT_OUTPUT, DocString{"List of Drivers"});
      ADD_SLOT( int     , id         , INPUT       , REQUIRED , DocString{"Driver index"});
      ADD_SLOT( double  , radius     , INPUT       , REQUIRED , DocString{"Radius of the ball, positive and should be superior to the biggest sphere radius in the ball"});
      ADD_SLOT( Vec3d   , center     , INPUT       , REQUIRED , DocString{"Center of the ball"});
      ADD_SLOT( Vec3d   , angular_velocity, INPUT  , null     , DocString{"Angular velocity of the ball, default is 0 m.s-"});
      ADD_SLOT( Vec3d   , velocity   , INPUT       , null     , DocString{"Ball velocity"});

      public:

      inline std::string documentation() const override final
      {
        return R"EOF(
        This operator add a ball (boundary condition) to the drivers list.
        )EOF";
      }

      inline void execute () override final
      {
        exaDEM::Ball driver = {*radius, *center, *velocity, *angular_velocity};
				driver.initialize();
        drivers->add_driver(*id, driver);
      }
    };

  template<class GridT> using AddBallTmpl = AddBall<GridT>;

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "add_ball", make_grid_variant_operator< AddBallTmpl > );
  }
}

