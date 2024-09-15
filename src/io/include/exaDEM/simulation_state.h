/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE
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
		Vec3d rotation_energy = { 0. , 0. , 0. }; 
		Vec3d kinetic_energy = { 0. , 0. , 0. }; 
		double mass = 0.;
		unsigned long long int n_particles = 0;
	};

	struct ReduceSimulationStateFunctor
	{
		ONIKA_HOST_DEVICE_FUNC inline void operator () (simulation_state_variables& local_variables, const double vx, const double vy, const double vz, const Vec3d& vrot, const double m, reduce_thread_local_t={} ) const
		{
			Vec3d v { vx, vy, vz };
			local_variables.mass += m;
			local_variables.rotation_energy += 0.5 * vrot * vrot * m;
			local_variables.kinetic_energy  += 0.5 * v * v * m;
			local_variables.n_particles += 1;
		}

		ONIKA_HOST_DEVICE_FUNC inline void operator () ( simulation_state_variables& global, simulation_state_variables local, reduce_thread_block_t ) const
		{
			ONIKA_CU_ATOMIC_ADD( global.mass              , local.mass );
			ONIKA_CU_ATOMIC_ADD( global.rotation_energy.x , local.rotation_energy.x );
			ONIKA_CU_ATOMIC_ADD( global.rotation_energy.y , local.rotation_energy.y );
			ONIKA_CU_ATOMIC_ADD( global.rotation_energy.z , local.rotation_energy.z );
			ONIKA_CU_ATOMIC_ADD( global.kinetic_energy.x  , local.kinetic_energy.x );
			ONIKA_CU_ATOMIC_ADD( global.kinetic_energy.y  , local.kinetic_energy.y );
			ONIKA_CU_ATOMIC_ADD( global.kinetic_energy.z  , local.kinetic_energy.z );
			ONIKA_CU_ATOMIC_ADD( global.n_particles       , local.n_particles );
		}

		ONIKA_HOST_DEVICE_FUNC inline void operator () (simulation_state_variables& global , simulation_state_variables local, reduce_global_t ) const
		{
			ONIKA_CU_ATOMIC_ADD( global.mass              , local.mass );
			ONIKA_CU_ATOMIC_ADD( global.rotation_energy.x , local.rotation_energy.x );
			ONIKA_CU_ATOMIC_ADD( global.rotation_energy.y , local.rotation_energy.y );
			ONIKA_CU_ATOMIC_ADD( global.rotation_energy.z , local.rotation_energy.z );
			ONIKA_CU_ATOMIC_ADD( global.kinetic_energy.x  , local.kinetic_energy.x );
			ONIKA_CU_ATOMIC_ADD( global.kinetic_energy.y  , local.kinetic_energy.y );
			ONIKA_CU_ATOMIC_ADD( global.kinetic_energy.z  , local.kinetic_energy.z );
			ONIKA_CU_ATOMIC_ADD( global.n_particles       , local.n_particles );
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
