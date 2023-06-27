#pragma once

#include <exanb/compute/compute_cell_particles.h>
#include <exanb/core/basic_types_operators.h>
#include <onika/cuda/cuda.h>

namespace exaDEM
{

  struct PushToAngularAccelerationFunctor
  {
    ONIKA_HOST_DEVICE_FUNC inline void operator () (const exanb::Quaternion& Q, const exanb::Vec3d& mom, const exanb::Vec3d& vrot, exanb::Vec3d& arot, const exanb::Vec3d& inertia) const
    {
      using exanb::Quaternion;
      using exanb::Vec3d;
      Quaternion Qinv = get_conjugated(Q);
      const auto omega = Qinv * vrot;  // Express omega in the body framework
      const auto M =  Qinv * mom;    // Express torque in the body framework
      Vec3d domega{
        (M.x - (inertia.z - inertia.y) * omega.y * omega.z) / inertia.x,
        (M.y - (inertia.x - inertia.z) * omega.z * omega.x) / inertia.y,
        (M.z - (inertia.y - inertia.x) * omega.x * omega.y) / inertia.z};
	arot = Q * domega;  // Express arot in the global framework
    }
  };
}

namespace exanb
{
  template<> struct ComputeCellParticlesTraits<exaDEM::PushToAngularAccelerationFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };

}

