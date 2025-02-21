#pragma once

#include <onika/math/basic_types.h>
#include <exanb/grid_cell_particles/particle_region.h>
#include <onika/flat_tuple.h>


namespace exaDEM
{
  using namespace exanb;

  struct ParticleBarycenterTypeValue
  {
    int count; // the number of particles of a type
    Vec3d barycenter; // sum of particle centers
    inline ONIKA_HOST_DEVICE_FUNC void operator+=(Vec3d& r) 
    {
      count++; 
      barycenter += r; 
    }
  };

  struct ReduceParticleBarycenterTypeFunctor
  {
    const ParticleRegionCSGShallowCopy region; //
    const uint16_t filter_type; // 
    ONIKA_HOST_DEVICE_FUNC inline void operator()(ParticleBarycenterTypeValue& local, const double rx, const double ry, const double rz, const uint16_t type, reduce_thread_local_t = {}) const
    {
      Vec3d r = {rx, ry, rz};
      if( region.contains(r) )
      {
        if( type  == filter_type )
        {
          local += r;
        }
      }
    }

    ONIKA_HOST_DEVICE_FUNC inline void operator()(ParticleBarycenterTypeValue& global, const ParticleBarycenterTypeValue& local, reduce_thread_block_t) const
    {
      ONIKA_CU_ATOMIC_ADD(global.count, local.count);
      ONIKA_CU_ATOMIC_ADD(global.barycenter.x, local.barycenter.x);
      ONIKA_CU_ATOMIC_ADD(global.barycenter.y, local.barycenter.y);
      ONIKA_CU_ATOMIC_ADD(global.barycenter.z, local.barycenter.z);
    }

    ONIKA_HOST_DEVICE_FUNC inline void operator()(ParticleBarycenterTypeValue& global, const ParticleBarycenterTypeValue& local, reduce_global_t) const
    {
      ONIKA_CU_ATOMIC_ADD(global.count, local.count);
      ONIKA_CU_ATOMIC_ADD(global.barycenter.x, local.barycenter.x);
      ONIKA_CU_ATOMIC_ADD(global.barycenter.y, local.barycenter.y);
      ONIKA_CU_ATOMIC_ADD(global.barycenter.z, local.barycenter.z);
    }
  };
}
