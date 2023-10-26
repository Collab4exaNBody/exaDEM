//#pragma xstamp_cuda_enable // DO NOT REMOVE THIS LINE

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <onika/memory/allocator.h>

#include <onika/soatl/field_pointer_tuple.h>
#include <exanb/compute/compute_cell_particles.h>
#include <exanb/grid_cell_particles/particle_region.h>
#include <exanb/grid_cell_particles/particle_cell_projection.h>
#include <exanb/compute/fluid_friction.h>

namespace exaDEM
{
  using namespace exanb;

  struct SphereFrictionFunctor
  {
    ONIKA_HOST_DEVICE_FUNC inline Vec3d operator () ( const Vec3d& r, const Vec3d& pv, const Vec3d& fv , const double cx, const double radius ) const
    {
      const Vec3d relative_velocity = fv - pv;
      const double relative_velocity_norm = norm(relative_velocity);
      return cx * (relative_velocity * relative_velocity_norm * M_PI * radius*radius);
    }
  };

  template<class GridT> using SphereFluidFriction = FluidFriction< GridT , SphereFrictionFunctor , FieldSet<field::_radius> >;
  
 // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "sphere_fluid_friction", make_grid_variant_operator< SphereFluidFriction > );
  }

}

