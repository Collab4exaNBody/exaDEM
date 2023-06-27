#pragma once

#include <exanb/compute/compute_cell_particles.h>
#include <exanb/core/basic_types_operators.h>
#include <onika/cuda/cuda.h>

namespace exaDEM
{

  struct PushToAngularVelocityFunctor
  {
    double m_dt_2;
    ONIKA_HOST_DEVICE_FUNC inline void operator () (exanb::Vec3d& vrot, const exanb::Vec3d& arot) const
    {
      vrot += arot * m_dt_2; 
    }
  };
}

namespace exanb
{
  template<> struct ComputeCellParticlesTraits<exaDEM::PushToAngularVelocityFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };

}

