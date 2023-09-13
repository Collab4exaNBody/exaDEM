#pragma once

#include <exanb/core/basic_types.h>
#include <exanb/compute/compute_cell_particles.h>

namespace exaDEM
{
	using namespace exanb;
	struct QuadraticForceFunctor
	{
		double cx_mu;
		ONIKA_HOST_DEVICE_FUNC inline void operator () ( double& fx, double& fy, double& fz, const double vx, const double vy, const double vz ) const
		{
			Vec3d v = {vx, vy, vz};
			double vel = exanb::norm(v);
			fx -= cx_mu * vel * vx;
			fy -= cx_mu * vel * vy;
			fz -= cx_mu * vel * vz;
		}
	};
}

namespace exanb
{
	template<> struct ComputeCellParticlesTraits<exaDEM::QuadraticForceFunctor>
	{
		static inline constexpr bool RequiresBlockSynchronousCall = false;
		static inline constexpr bool CudaCompatible = true;
	};
}
