#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <onika/memory/allocator.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_math.h>
#include <exanb/compute/reduce_cell_particles.h>
#include <exaDEM/shapes.hpp>

namespace exaDEM
{

	using namespace exanb;
	struct DEMBackupData
	{
		using CellDEMBackupVector = onika::memory::CudaMMVector<double>;
		using DEMBackupVector = onika::memory::CudaMMVector< CellDEMBackupVector >;
		DEMBackupVector m_data;
		Mat3d m_xform;
	};

	struct ReduceMaxPolyhedronDisplacementFunctor
	{
		const DEMBackupData::CellDEMBackupVector * m_backup_data = nullptr;
		const double m_threshold_sqr = 0.0;
		const shapes& shps;

		ONIKA_HOST_DEVICE_FUNC inline void operator () (unsigned long long int & count_over_dist2 , IJK cell_loc, size_t cell, size_t j, double rx, double ry, double rz , uint8_t type, const exanb::Quaternion& orientation, reduce_thread_local_t={} ) const
		{
			const double* rb = onika::cuda::vector_data( m_backup_data[cell] );
			Quaternion old_orientation = {rb[j*7+3], rb[j*7+4], rb[j*7+5], rb[j*7+6]};
			Vec3d old_center = {rb[j*7+0], rb[j*7+1], rb[j*7+2]};
			Vec3d new_center = {rx, ry, rz};

			auto shp = shps[type];
			const int nv = shp->get_number_of_vertices();
			for( int v = 0 ; v < nv ; v++)
			{

				const Vec3d old_vertex = shp->get_vertex(v, old_center, old_orientation);
				const Vec3d new_vertex = shp->get_vertex(v, new_center, orientation);
				const Vec3d dr = new_vertex - old_vertex;
				if( exanb::dot(dr,dr) >= m_threshold_sqr )
				{
					++ count_over_dist2;
				}
			}
		}
		ONIKA_HOST_DEVICE_FUNC inline void operator () (unsigned long long int & count_over_dist2 , unsigned long long int value, reduce_thread_block_t ) const
		{
			ONIKA_CU_ATOMIC_ADD( count_over_dist2 , value );
		}

		ONIKA_HOST_DEVICE_FUNC inline void operator () (unsigned long long int & count_over_dist2 , unsigned long long int value, reduce_global_t ) const
		{
			ONIKA_CU_ATOMIC_ADD( count_over_dist2 , value );
		}
	};
}

namespace exanb
{
	template<> struct ReduceCellParticlesTraits<exaDEM::ReduceMaxPolyhedronDisplacementFunctor>
	{
		static inline constexpr bool RequiresBlockSynchronousCall = false;
		static inline constexpr bool RequiresCellParticleIndex = true;
		static inline constexpr bool CudaCompatible = true;
	};
}
