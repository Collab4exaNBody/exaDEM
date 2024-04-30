#pragma once

#include <exanb/core/basic_types.h>
#include <onika/cuda/cuda.h> // mots cles specifiques
#include <onika/memory/allocator.h> // cudaMMVector
#include <onika/cuda/stl_adaptors.h>
#include <exanb/core/basic_types.h>

namespace exaDEM
{

	using namespace exanb;
	
	struct Interactions_PP
	{
		int nb_particles=0;
		onika::memory::CudaMMVector<int> pa;
		onika::memory::CudaMMVector<int> cella;
		onika::memory::CudaMMVector< onika::memory::CudaMMVector<int>> pb;
		onika::memory::CudaMMVector< onika::memory::CudaMMVector<int>> cellb;
		onika::memory::CudaMMVector< onika::memory::CudaMMVector<Vec3d>> ft_pair;
		
		
		void reset()
		{
			nb_particles= 0;
			pa.clear();
			pa.resize(0);
			cella.clear();
			cella.resize(0);
			for(int i=0; i<pb.size(); i++){
				pb[i].clear();
				pb[i].resize(0);
				cellb[i].clear();
				cellb[i].resize(0);
				ft_pair[i].clear();
				ft_pair[i].resize(0);
			}
			pb.clear();
			pb.resize(0);
			cellb.clear();
			cellb.resize(0);
			ft_pair.clear();
			ft_pair.resize(0);
		}
		
		void add_particle(int p, int cell, std::vector<std::pair<int,int>> nbh)
		{
			pa.push_back(p);
			cella.push_back(cell);
			
			nb_particles++;
			
			pb.resize(nb_particles);
			cellb.resize(nb_particles);
			ft_pair.resize(nb_particles);
			
			auto& p_b= pb[nb_particles-1];
			auto& cell_b= cellb[nb_particles-1];
			auto& ft= ft_pair[nb_particles-1];
			
			
			for(auto pair: nbh)
			{
				p_b.push_back(pair.first);
				cell_b.push_back(pair.second);
				ft.push_back({0, 0, 0});
			}
			
		}
			
	};
}

