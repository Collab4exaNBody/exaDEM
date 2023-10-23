#pragma once

#include <exanb/core/basic_types.h>
#include <exanb/compute/reduce_cell_particles.h>
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/core/log.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/fields.h>
#include <exanb/core/basic_types.h>
#include <exanb/compute/reduce_cell_particles.h>
#ifdef ONIKA_CUDA_VERSION
#include <onika/cuda/cuda.h>
#endif

namespace exaDEM
{
	using namespace exanb;

	struct simulation_state_variables
	{
		Vec3d momentum = { 0. , 0. , 0. }; 
		Vec3d kinetic_energy = { 0. , 0. , 0. }; 
		double mass = 0.;
		unsigned long long int n_particles = 0;
	};

	struct ReduceSimulationStateFunctor
	{
		ONIKA_HOST_DEVICE_FUNC inline void operator () (simulation_state_variables& local_variables, double vx, double vy, double vz, double m, reduce_thread_local_t={} ) const
		{
			Vec3d v { vx, vy, vz };
			local_variables.mass += m;
			local_variables.momentum += v*m;
			local_variables.kinetic_energy += 0.5 * v * v * m;
			local_variables.n_particles += 1;
		}

		ONIKA_HOST_DEVICE_FUNC inline void operator () ( simulation_state_variables& global, simulation_state_variables local, reduce_thread_block_t ) const
		{
			ONIKA_CU_ATOMIC_ADD( global.mass , local.mass );
			ONIKA_CU_ATOMIC_ADD( global.momentum.x , local.momentum.x );
			ONIKA_CU_ATOMIC_ADD( global.momentum.y , local.momentum.y );
			ONIKA_CU_ATOMIC_ADD( global.momentum.z , local.momentum.z );
			ONIKA_CU_ATOMIC_ADD( global.kinetic_energy.x , local.kinetic_energy.x );
			ONIKA_CU_ATOMIC_ADD( global.kinetic_energy.y , local.kinetic_energy.y );
			ONIKA_CU_ATOMIC_ADD( global.kinetic_energy.z , local.kinetic_energy.z );
			ONIKA_CU_ATOMIC_ADD( global.n_particles , local.n_particles );
		}

		ONIKA_HOST_DEVICE_FUNC inline void operator () (simulation_state_variables& global , simulation_state_variables local, reduce_global_t ) const
		{
			ONIKA_CU_ATOMIC_ADD( global.mass , local.mass );
			ONIKA_CU_ATOMIC_ADD( global.momentum.x , local.momentum.x );
			ONIKA_CU_ATOMIC_ADD( global.momentum.y , local.momentum.y );
			ONIKA_CU_ATOMIC_ADD( global.momentum.z , local.momentum.z );
			ONIKA_CU_ATOMIC_ADD( global.kinetic_energy.x , local.kinetic_energy.x );
			ONIKA_CU_ATOMIC_ADD( global.kinetic_energy.y , local.kinetic_energy.y );
			ONIKA_CU_ATOMIC_ADD( global.kinetic_energy.z , local.kinetic_energy.z );
			ONIKA_CU_ATOMIC_ADD( global.n_particles , local.n_particles );
		}
	};

};
namespace exanb
{
	template<> struct ReduceCellParticlesTraits<exaDEM::ReduceSimulationStateFunctor>
	{
		static inline constexpr bool RequiresBlockSynchronousCall = false;
		static inline constexpr bool RequiresCellParticleIndex = false;
		static inline constexpr bool CudaCompatible = true;
	};
};
