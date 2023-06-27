#pragma once

#include <exanb/compute/compute_cell_particles.h>
#include <onika/cuda/cuda.h>

namespace exaDEM
{

  struct ForceToAccelFunctor
  {
    ONIKA_HOST_DEVICE_FUNC inline void operator () (const double mass, double& fx, double& fy, double& fz ) const
    {
      fx /= mass;
      fy /= mass;
      fz /= mass;
    }
  };
}

namespace exanb
{
  template<> struct ComputeCellParticlesTraits<exaDEM::ForceToAccelFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };

}

