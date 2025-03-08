#pragma once

#include <onika/math/basic_types.h>
#include <exanb/grid_cell_particles/particle_region.h>
#include <onika/flat_tuple.h>

namespace exaDEM
{
  using namespace exanb;
  struct ReduceParticleCounterTypeFunctor
  {
    const ParticleRegionCSGShallowCopy region;
    const uint16_t filter_type;
    ONIKA_HOST_DEVICE_FUNC inline void operator()(int &local, const double rx, const double ry, const double rz, const uint16_t type, reduce_thread_local_t = {}) const
    {
      Vec3d r = {rx, ry, rz};
      if( region.contains(r) )
      {
        if( type  == filter_type )
        {
          local++;
        }
      }
    }

    ONIKA_HOST_DEVICE_FUNC inline void operator()(int &global, int local, reduce_thread_block_t) const
    {
      ONIKA_CU_ATOMIC_ADD(global, local);
    }

    ONIKA_HOST_DEVICE_FUNC inline void operator()(int &global, int local, reduce_global_t) const
    {
      ONIKA_CU_ATOMIC_ADD(global, local);
    }
  };
}
