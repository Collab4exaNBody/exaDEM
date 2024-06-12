
#pragma once

#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>

namespace exaDEM
{
	struct interaction_collection
	{
		template <typename T> using VectorT =  onika::memory::CudaMMVector<T>;
		VectorT<VectorT<exaDEM::Interaction> waves;

		void reset_waves()
		{
			for(auto& wave : waves)
			{
				wave.clean();
			}
		}

		size_t number_of_waves() {return waves.size();}

		void classify(GridCellParticleInteraction& ges)
		{
			reset_waves();
			size_t nw = number_of_waves();
			// first loop to figure out the number of interactions per wave
			std::array<int, nw> sizes;
			std::array<int, nw> shifts;

			auto& ces = ges->m_data; // cells

#pragma omp parallel for reduction(+: sizes)
			for(size_t c = 0 ; c < ces.size() ; c++)
			{
				auto& cell = ces[c];
				for( size_t i = 0 ; i < cell.size() ;  i++)
				{
					auto& I = cell[i];
					sizes[I.type]++;
				}

				for(int w = 0 ; w < nw ; w++) 
				{
					waves[w].resize(sizes[w]);
				}

				// serial here, should be //	
				for(size_t c = 0 ; c < ces.size() ; c++)
				{
					auto& cell = ces[c];
					for( size_t i = 0 ; i < cell.size() ;  i++)
					{
						auto& I = cell[i];
						const auto tt = I.type;
						auto& wave = waves[t];
						auto& shift = shift[t];
						wave[shift++] = I;
					}
				}
			}
		}
	};
}
