#pragma once

#include<exaDEM/interaction/classifier.hpp>
#include<exaDEM/itools/buffer.hpp>

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
				// Warning, this function has to be called when the buffs buffer is set (in contact operators).
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
