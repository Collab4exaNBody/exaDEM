
#pragma once

#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>

namespace exaDEM
{
	struct Classifier
	{
		static constexpr int types = 13;
		template <typename T> using VectorT =  onika::memory::CudaMMVector<T>;
		std::vector<VectorT<exaDEM::Interaction>> waves;

		Classifier() { waves.resize(types); }

		void initialize() { waves.resize(types); }

		void reset_waves()
		{
			for(auto& wave : waves)
			{
				wave.clear();
			}
		}

		VectorT<exaDEM::Interaction>& get_wave(size_t id) {return waves[id];}

		std::pair<exaDEM::Interaction*, size_t> get_info(size_t id) 
		{
			const unsigned int  data_size = onika::cuda::vector_size( waves[id]);
			exaDEM::Interaction* const data_ptr = onika::cuda::vector_data( waves[id] );
			return {data_ptr, data_size};
		}

		size_t number_of_waves() {return waves.size();}

		void classify(GridCellParticleInteraction& ges)
		{
			reset_waves();
			// first loop to figure out the number of interactions per wave
			std::array<int, types> sizes;
			std::array<int, types> shifts;

			for(int  w = 0 ; w < types ; w++) {sizes[w] = 0 ; shifts[w]=0;}

			auto& ces = ges.m_data; // cells

//#pragma omp parallel
//			{
				std::array<int, types> ls; // local storage
				for(int  w = 0 ; w < types ; w++) {ls[w]=0;}
//#pragma omp for
				for(size_t c = 0 ; c < ces.size() ; c++)
				{
					auto& interactions = ces[c];
					const unsigned int  n_interactions_in_cell = interactions.m_data.size();
					exaDEM::Interaction* const __restrict__ data_ptr = onika::cuda::vector_data( interactions.m_data );

					for( size_t it = 0; it < n_interactions_in_cell ; it++ )
					{
						Interaction& item = data_ptr[it];
						ls[item.type]++;
					}
				}
//#pragma omp critical
				{
					for(int w = 0 ; w < types ; w++) sizes[w] += ls[w];
				}
//			}

			for(int w = 0 ; w < types ; w++) 
			{
				if(sizes[w] == 0) waves[w].clear();
				else waves[w].resize(sizes[w]);
			}

			// serial here, should be //	
			for(size_t c = 0 ; c < ces.size() ; c++)
			{
				auto& interactions = ces[c];
				const unsigned int  n_interactions_in_cell = interactions.m_data.size();
				exaDEM::Interaction* const __restrict__ data_ptr = onika::cuda::vector_data( interactions.m_data );
				for( size_t it = 0; it < n_interactions_in_cell ; it++ )
				{
					Interaction& item = data_ptr[it];
					const int t = item.type;
					auto& wave = waves[t];
					auto& shift = shifts[t];
					wave[shift++] = item;
				}
			}
		}

		void unclassify(GridCellParticleInteraction& ges)
		{
			Vec3d null = {0,0,0};
			auto& ces = ges.m_data; // cells
			for(int w = 0 ; w < types ; w++)
			{
				auto& wave = waves[w];
				const unsigned int n1 = wave.size();
#pragma omp parallel for
				for(size_t it = 0 ; it < n1 ; it++) 
				{
					exaDEM::Interaction& item1 = wave[it];
					if( item1.friction != null && item1.moment != null)
					{ 
						auto& cell = ces[item1.cell_i];
						const unsigned int  n2 = onika::cuda::vector_size( cell.m_data );
						exaDEM::Interaction* data_ptr = onika::cuda::vector_data( cell.m_data );
						for(size_t it2 = 0; it2 < n2 ; it2++)
						{
							exaDEM::Interaction& item2 = data_ptr[it2];
							if(item1 == item2)
							{
								item2.update_friction_and_moment(item1);
								break;
							}
						}
					}
				}
			}
		}
	};
}
