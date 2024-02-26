#pragma once
#include <cassert>
#include <exanb/core/basic_types.h>
#include <exanb/compute/compute_cell_particles.h>

namespace exaDEM
{
  struct PolyhedraComputeVerticesFunctor
  {
    shapes& shps;
    ONIKA_HOST_DEVICE_FUNC inline void operator () (const uint8_t type, const double rx, const double ry, const double rz, const double h, const exanb::Quaternion& orient, ::onika::oarray_t<::exanb::Vec3d, 8>& vertices ) const
    {
			// h will be used in a next development
			const auto& shp = shps[type];
			const unsigned int nv = shp -> get_number_of_vertices();
			const exanb::Vec3d position = {rx, ry, rz};
			assert ( nv < 8 );
			for ( size_t i = 0 ; i < nv ; i++ )
			{
				vertices[i] = shp -> get_vertex (i, position, orient);
			} 
    }
  };
}

namespace exanb
{
  template<> struct ComputeCellParticlesTraits<exaDEM::PolyhedraComputeVerticesFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = false;
  };
}

