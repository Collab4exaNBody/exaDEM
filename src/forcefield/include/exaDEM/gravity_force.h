#pragma once

#include <exanb/core/basic_types.h>
#include <exanb/compute/compute_cell_particles.h>

namespace exaDEM
{
  struct GravityForceFunctor
  {
    exanb::Vec3d g = { 0.0 , 0.0 , -9.807};
    ONIKA_HOST_DEVICE_FUNC inline void operator () (double mass, double& fx, double& fy, double& fz ) const
    {
      fx += g.x * mass;
      fy += g.y * mass;
      fz += g.z * mass;
    }
  };
}

namespace exanb
{
  template<> struct ComputeCellParticlesTraits<exaDEM::GravityForceFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };
}

