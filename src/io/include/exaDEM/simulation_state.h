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
#include <cuda.h>
#endif

namespace cuda_helper{
#ifdef ONIKA_CUDA_VERSION
  __device__ void atomicAdd(double* address, double val)
  {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed,
	  __double_as_longlong(val +
	    __longlong_as_double(assumed)));
    } while (assumed != old);
    return; // __longlong_as_double(old);
  }
#else
  void atomicAdd(double* address, double val)
  {
#pragma omp atomic write
    *address = val;
  }
#endif
}

namespace exaDEM
{
  using namespace exanb;

  struct simulation_state_variables
  {
    Vec3d kinetic_energy;
    Vec3d momentum; 
    double mass;
    double potential_energy;
    unsigned long long int n_particles;
  };

  struct ReduceDoubleFunctor
  {
    ONIKA_DEVICE_FUNC inline void operator () (double& local, double value, reduce_thread_local_t={} ) const
    {
      local += value;
    }

    ONIKA_DEVICE_FUNC inline void operator () ( double& global, const double local, reduce_thread_block_t ) const
    {
      // cuda_helper::atomicAdd( global , local );
      cuda_helper::atomicAdd(&global,local);
    }

    ONIKA_DEVICE_FUNC inline void operator () (double& global , double local, reduce_global_t ) const
    {
      //cuda_helper::atomicAdd( global , local );
      cuda_helper::atomicAdd(&global,local);
    }
  };

  struct ReduceSimulationStateFunctor
  {
    ONIKA_HOST_DEVICE_FUNC inline void operator () (simulation_state_variables& local_variables, double vx, double vy, double vz, double m, reduce_thread_local_t={} ) const
    {
      Vec3d v { vx, vy, vz };
      local_variables.mass += m;
      local_variables.momentum += v*m;
      local_variables.potential_energy = 0;
      local_variables.kinetic_energy += 0.5 * v * v * m;
      local_variables.n_particles += 1;
    }

    ONIKA_DEVICE_FUNC inline void operator () ( simulation_state_variables& global, simulation_state_variables local, reduce_thread_block_t ) const
    {
      cuda_helper::atomicAdd( &global.mass , local.mass );
      cuda_helper::atomicAdd( &global.momentum.x , local.momentum.x );
      cuda_helper::atomicAdd( &global.momentum.y , local.momentum.y );
      cuda_helper::atomicAdd( &global.momentum.z , local.momentum.z );
      cuda_helper::atomicAdd( &global.potential_energy , local.potential_energy );
      cuda_helper::atomicAdd( &global.kinetic_energy.x , local.kinetic_energy.x );
      cuda_helper::atomicAdd( &global.kinetic_energy.y , local.kinetic_energy.y );
      cuda_helper::atomicAdd( &global.kinetic_energy.z , local.kinetic_energy.z );
      ONIKA_CU_ATOMIC_ADD( global.n_particles , local.n_particles );
    }

    ONIKA_DEVICE_FUNC inline void operator () (simulation_state_variables& global , simulation_state_variables local, reduce_global_t ) const
    {
      cuda_helper::atomicAdd( &global.mass , local.mass );
      cuda_helper::atomicAdd( &global.momentum.x , local.momentum.x );
      cuda_helper::atomicAdd( &global.momentum.y , local.momentum.y );
      cuda_helper::atomicAdd( &global.momentum.z , local.momentum.z );
      cuda_helper::atomicAdd( &global.potential_energy , local.potential_energy );
      cuda_helper::atomicAdd( &global.kinetic_energy.x , local.kinetic_energy.x );
      cuda_helper::atomicAdd( &global.kinetic_energy.y , local.kinetic_energy.y );
      cuda_helper::atomicAdd( &global.kinetic_energy.z , local.kinetic_energy.z );
      ONIKA_CU_ATOMIC_ADD( global.n_particles , local.n_particles );
    }
  };

};
namespace exanb
{
  template<> struct ReduceCellParticlesTraits<exaDEM::ReduceDoubleFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool RequiresCellParticleIndex = false;
    static inline constexpr bool CudaCompatible = true;
  };

  template<> struct ReduceCellParticlesTraits<exaDEM::ReduceSimulationStateFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool RequiresCellParticleIndex = false;
    static inline constexpr bool CudaCompatible = true;
  };
};
