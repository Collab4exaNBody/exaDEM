#pragma once

//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

#include<exaDEM/interaction/classifier.hpp>
#include<exaDEM/itools/buffer.hpp>

#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_error.h>
#include <onika/cuda/device_storage.h>
#include <onika/soatl/field_id.h>
#include <exanb/core/grid_particle_field_accessor.h>

#include <exanb/core/parallel_grid_algorithm.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for.h>
#include <onika/parallel/parallel_for.h>
#include <exanb/compute/reduce_cell_particles.h>

// mini macro here
//#ifdef ONIKA_CUDA_VERSION
#   if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)

__device__ double atomicMin_double(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

#   define EXADEM_CU_ATOMIC_MIN(x,a,...) atomicMin_double( &x , static_cast<std::remove_reference_t<decltype(x)> >(a) )

#else
#   define EXADEM_CU_ATOMIC_MIN(x,a,...) ::onika::capture_atomic_min( x , static_cast<std::remove_reference_t<decltype(x)> >(a) )
#endif



namespace exaDEM
{
	namespace itools /* interaction tools */
	{
		using namespace onika::parallel;
    template <typename T> using VectorT =  onika::memory::CudaMMVector<T>;

		template<typename T, class FuncT, class ResultT>
			struct ReduceTFunctor
			{
				const T* const m_data;
				const FuncT m_func;
				ResultT* m_reduced_val ;

				ONIKA_HOST_DEVICE_FUNC inline void operator () ( uint64_t i ) const
				{
					ResultT local_val = ResultT();
					m_func( local_val, i, m_data,  reduce_thread_local_t{} );

					ONIKA_CU_BLOCK_SHARED onika::cuda::UnitializedPlaceHolder<ResultT> team_val_place_holder;
					ResultT& team_val = team_val_place_holder.get_ref();

					if( ONIKA_CU_THREAD_IDX == 0 ) { team_val = local_val; }
					ONIKA_CU_BLOCK_SYNC();

					if( ONIKA_CU_THREAD_IDX != 0 )
					{
						m_func( team_val, local_val , reduce_thread_block_t{} );
					}
					ONIKA_CU_BLOCK_SYNC();

					if( ONIKA_CU_THREAD_IDX == 0 )
					{
						m_func( *m_reduced_val, team_val , reduce_global_t{} );
					}
				}
			};

		struct IOSimInteractionResult
		{
			unsigned long long int n_act_interaction = 0;
			unsigned long long int n_tot_interaction = 0;
			double min_dn = 0;
      void update(IOSimInteractionResult& in)
      {
        n_act_interaction += in.n_act_interaction;
        n_tot_interaction += in.n_tot_interaction;
        min_dn             = std::min( min_dn, in.min_dn);
      }   
		};
 
		struct IOSimInteractionFunctor
		{
			const double * const dnp;
			const int coef; // 2 if sym and not an interaction between a driver and a particle

			ONIKA_HOST_DEVICE_FUNC inline void operator () ( 
					IOSimInteractionResult& local, 
					const uint64_t idx, 
					const exaDEM::Interaction* const interactions, 
					reduce_thread_local_t={} ) const
			{
				const exaDEM::Interaction& I = interactions[idx];

				// filter duplicate (mpi ghost)
				if( I.id_i < I.id_j)
				{
					const double& dn = dnp[idx];
					local.n_tot_interaction += coef;
					if( dn < 0.0 )
					{ 
						local.n_act_interaction += coef;
						local.min_dn = std::min(local.min_dn, dn);
					}
				}
			}

			ONIKA_HOST_DEVICE_FUNC inline void operator () ( 
					IOSimInteractionResult& global, 
					IOSimInteractionResult& local, 
					reduce_thread_block_t ) const
			{
				ONIKA_CU_ATOMIC_ADD(global.n_act_interaction, local.n_act_interaction);
				ONIKA_CU_ATOMIC_ADD(global.n_tot_interaction, local.n_tot_interaction);
				EXADEM_CU_ATOMIC_MIN(global.min_dn, local.min_dn);
			}

			ONIKA_HOST_DEVICE_FUNC inline void operator () ( 
					IOSimInteractionResult& global, 
					IOSimInteractionResult& local, 
					reduce_global_t ) const
			{
				ONIKA_CU_ATOMIC_ADD(global.n_act_interaction, local.n_act_interaction);
				ONIKA_CU_ATOMIC_ADD(global.n_tot_interaction, local.n_tot_interaction);
				EXADEM_CU_ATOMIC_MIN(global.min_dn, local.min_dn);
			}
		};

		template<typename T, typename Func, typename ResultT>
			static inline ParallelExecutionWrapper reduce_data(
					ParallelExecutionContext * exec_ctx, 
					const T* const data,
					Func& func, 
					uint64_t size, 
					ResultT& result)
			{
				ParallelForOptions opts;
				opts.omp_scheduling = OMP_SCHED_STATIC;
				ReduceTFunctor<T, Func, ResultT> kernel = {data, func, &result};
				return parallel_for( size, kernel, exec_ctx, opts);
			}

	}
}

namespace exanb
{
	template<> struct ReduceCellParticlesTraits<exaDEM::itools::ReduceTFunctor>
	{
		static inline constexpr bool RequiresBlockSynchronousCall = false;
		static inline constexpr bool RequiresCellParticleIndex = false;
		static inline constexpr bool CudaCompatible = true;
	};
};

namespace exaDEM
{
	namespace itools /* interaction tools */
	{
		/**
		 * @brief This function returns the total number of classifier interactions.
		 */
		template < typename GridT >
			inline uint64_t get_tot_interaction_size(const GridT& grid, const Classifier& classifier, const bool sym)
			{
				uint64_t res = 0 ;
				// interactions between particles
				if ( sym == true ) 
				{
					for( size_t i = 0 ; i < 4 ; i++ )
					{
						const auto& I = classifier.get_wave(i); 
#pragma omp parallel for reduction(+: res)
						for( size_t j = 0 ; j < I.size() ; j++ )
						{
							if ( filter_duplicates(grid, I[j]) )
							{
								res += 2;
							}
						}
					}
				}
				else /* not symetric */
				{
					for(size_t i = 0 ; i < 4 ; i++ )
					{
						const auto [ptr, size] = classifier.get_info(i);
						res += size;
					}
				}

				// particles - drivers
				for(size_t i = 4 ; i < classifier.number_of_waves() ; i++ )
				{
					const auto [ptr, size] = classifier.get_info(i);
					res += size;
				}
				ldbg << " number_of_interactions " << res << std::endl;
				return res;
			}

		/**
		 * @brief This functions returns the number of active interactions; i.e. if two particles are overlapped by themselves.
		 */
		template <typename GridT>
			uint64_t get_act_interaction_size( const GridT& grid, const Classifier& classifier, const bool sym)
			{
				// Warning, this function has to be called when the buffs buffer is set (in hooke operators).
				// TODO : Implement a GPU version
				uint64_t res = 0;

				// interactions between particles
				if ( sym == true ) 
				{
					for( size_t i = 0 ; i < 4 ; i++ )
					{
						const auto& buffs = classifier.buffers[i];
						const auto& I = classifier.get_wave(i); 
						const double* const dnp = onika::cuda::vector_data( buffs.dn ); 
						const size_t size       = onika::cuda::vector_size( buffs.dn );
#pragma omp parallel for reduction(+: res)
						for( size_t j = 0 ; j < size ; j++ )
						{
							if( dnp[j] < 0 ) /* skip void interactions */
							{ 
								if ( filter_duplicates(grid, I[j]) ) /* skip duplicated interactions in ghost area */
								{
									res += 2; 
								}
							}
						}
					}
				}
				else /* not symetric */
				{
					for( size_t i = 0 ; i < 4 ; i++ )
					{
						// number of iteractions per wave
						const auto& buffs = classifier.buffers[i];
						const double* const dnp = onika::cuda::vector_data( buffs.dn ); 
						const size_t size       = onika::cuda::vector_size( buffs.dn );
#pragma omp parallel for reduction(+: res)
						for( size_t j = 0 ; j < size ; j++ )
						{
							if( dnp[j] < 0 )
							{ 
								res += 1;
							}
						}
					}
				}

				// interactions particles - drivers
				for( size_t i = 4 ; i < classifier.number_of_waves() ; i++ )
				{
					const auto& buffs = classifier.buffers[i];
					const double* const dnp = onika::cuda::vector_data( buffs.dn ); 
					const size_t size       = onika::cuda::vector_size( buffs.dn );
#pragma omp parallel for reduction(+: res)
					for( size_t j = 0 ; j < size ; j++ )
					{
						if( dnp[j] < 0 ) res++;
					}
				}

				ldbg << " get_active_interactions " << res << std::endl;
				return res;
			}
	}
}
