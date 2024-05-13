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
		std::vector<int> id_a;
		
		onika::memory::CudaMMVector< onika::memory::CudaMMVector<int>> pb;
		onika::memory::CudaMMVector< onika::memory::CudaMMVector<int>> cellb;
		onika::memory::CudaMMVector< onika::memory::CudaMMVector<Vec3d>> ft_pair;
		std::vector< std::vector<int>> id_b;
		
		
		void reset()
		{
			nb_particles= 0;
			pa.clear();
			pa.resize(0);
			cella.clear();
			cella.resize(0);
			id_a.clear();
			id_a.resize(0);
			for(int i=0; i<pb.size(); i++){
				pb[i].clear();
				pb[i].resize(0);
				cellb[i].clear();
				cellb[i].resize(0);
				ft_pair[i].clear();
				ft_pair[i].resize(0);
				id_b[i].clear();
				id_b[i].resize(0);
			}
			pb.clear();
			pb.resize(0);
			cellb.clear();
			cellb.resize(0);
			ft_pair.clear();
			ft_pair.resize(0);
			id_b.clear();
			id_b.resize(0);
		}
		
		void add_particle(int p, int cell, std::vector<std::pair<int,int>> nbh, int ida, std::vector<int> idb)
		{
			pa.push_back(p);
			cella.push_back(cell);
			id_a.push_back(ida);
			
			nb_particles++;
			
			pb.resize(nb_particles);
			cellb.resize(nb_particles);
			ft_pair.resize(nb_particles);
			id_b.resize(nb_particles);
				
			
			auto& p_b= pb[nb_particles-1];
			auto& cell_b= cellb[nb_particles-1];
			auto& ft= ft_pair[nb_particles-1];
			auto& idb_ = id_b[nb_particles-1];
			
			//for(auto pair: nbh)
			for(int i = 0; i < nbh.size(); i++)
			{
				p_b.push_back(nbh[i].first);
				cell_b.push_back(nbh[i].second);
				ft.push_back({0, 0, 0});
				idb_.push_back(idb[i]);
			}
			
		}
		
		//TRI QUICK SORT
		void swap_a(int i, int j)
		{
			int temp_pa = pa[i];
			int temp_cella = cella[i];
			int temp_ida = id_a[i];
			
			int size_i = pb[i].size();
			int size_j = pb[j].size();
			
			std::vector<int> temp_pb;
			temp_pb.resize(size_i);
			std::vector<int> temp_cellb;
			temp_cellb.resize(size_i);
			std::vector<int> temp_idb;
			temp_idb.resize(size_i);
			std::vector<Vec3d> temp_ft;
			temp_ft.resize(size_i);
			
			for(int z = 0; z < size_i; z++)
			{
				temp_pb[z] = pb[i][z];
				temp_cellb[z] = cellb[i][z];
				temp_idb[z] = id_b[i][z];
				temp_ft[z] = ft_pair[i][z];
			}
			
			pa[i] = pa[j];
			pa[j] = temp_pa;
			
			cella[i] = cella[j];
			cella[j] = temp_cella;
			
			id_a[i] = id_a[j];
			id_a[j] = temp_ida;
			
			pb[i].clear();
			pb[i].resize(size_j);
			
			cellb[i].clear();
			cellb[i].resize(size_j);
			
			id_b[i].clear();
			id_b[i].resize(size_j);
			
			ft_pair[i].clear();
			ft_pair[i].resize(size_j);
			
			for(int z = 0; z < size_j; z++)
			{
				pb[i][z] = pb[j][z];
				cellb[i][z] = cellb[j][z];
				id_b[i][z] = id_b[j][z];
				ft_pair[i][z] = ft_pair[j][z];
			}
			
			pb[j].clear();
			pb[j].resize(size_i);
			
			cellb[j].clear();
			cellb[j].resize(size_i);
			
			id_b[j].clear();
			id_b[j].resize(size_i);
			
			ft_pair[j].clear();
			ft_pair[j].resize(size_i);
			
			for(int z = 0; z < size_i; z++)
			{
				pb[j][z] = temp_pb[z];
				cellb[j][z] = temp_cellb[z];
				id_b[j][z] = temp_idb[z];
				ft_pair[j][z] = temp_ft[z];
			}
		}
		
		void swap_b(int pa, int i, int j)
		{
			int temp_pb = pb[pa][i];
			int temp_cellb = cellb[pa][i];
			int temp_idb = id_b[pa][i];
			Vec3d temp_ft = ft_pair[pa][i];
			
			pb[pa][i] = pb[pa][j];
			pb[pa][j] =  temp_pb;
			
			cellb[pa][i] = cellb[pa][j];
			cellb[pa][j] = temp_cellb;
			
			id_b[pa][i] = id_b[pa][j];
			id_b[pa][j] = temp_idb;
			
			ft_pair[pa][i] = ft_pair[pa][j];
			ft_pair[pa][j] = temp_ft;
		}
		
		// Fonction pour partitionner le vecteur autour d'un pivot
		int partition_a(int low, int high)
		{
			int pivot = id_a[high];// Choix du pivot
			int i = low - 1;//Index du plus petit élément
			
			for(int j = low; j < high; j++)
			{
				if(id_a[j] < pivot)
				{
					i++;
					swap_a(i, j);
				}
			}
			
			swap_a(i + 1, high);
			return i + 1; 
		}
		
		int partition_b(int pa, int low, int high)
		{
			int pivot = id_b[pa][high];// Choix du pivot
			int i = low - 1;//Index du plus petit élément
			
			for(int j = low; j < high; j++)
			{
				if(id_b[pa][j] < pivot)
				{
					i++;
					swap_b(pa, i, j);
				}
			}
			
			swap_b(pa, i + 1, high);
			return i + 1; 
		}
		
		
		// Fonction récursive pour trier le vecteur en utilisant Quick Sort
		void quickSort_a(int low, int high)
		{
			if(low < high)
			{
				int pi = partition_a(low, high); // Partitionne le vecteur
				quickSort_a(low, pi - 1); // Trie les éléments avant le pivot
				quickSort_a(pi + 1, high); // Trie les éléments après le pivot
			}
		}
		
		void quickSort_b(int pa, int low, int high)
		{
			if(low < high)
			{
				int pi = partition_b(pa, low, high);
				quickSort_b(pa, low, pi - 1);
				quickSort_b(pa, pi + 1, high);
			}
		}
		
		void quickSort()
		{
			//printf("QS\n");
			int n = pa.size();
			quickSort_a(0, n - 1);
			
			for(int i = 0; i < nb_particles; i++)
			{
				int m = pb[i].size();
				quickSort_b(i, 0, m - 1);
			}
			//printf("QS_END\n");
		}
		//TRI QUICK_SORT
		
		void init_friction(Interactions_PP mid)
		{
			printf("FRICTION\n");
			if(nb_particles > 0 && mid.nb_particles > 0)
			{
				
				int a = 0;
				int b = 0; 
				
				while(a < nb_particles && b < mid.nb_particles)
				{
					printf("WHILE\n");
					if(id_a[a] == mid.id_a[b])
					{
						int a2 = 0;
						int b2 = 0;
						
						while(a2 < id_b[a].size() && b2 < mid.id_b[b].size())
						{
							printf("WHILE2\n");
							
							if(id_b[a][a2] == mid.id_b[b][b2])
							{
								ft_pair[a][a2] = mid.ft_pair[b][b2];
								a2++;
								b2++;
							}
							else if(id_b[a][a2] > mid.id_b[b][b2])
							{
								b2++;
							}
							else
							{
								a2++;
							}	
						}
						
						a++;
						b++; 
					}
					else if(id_a[a] > mid.id_a[b])
					{
						b++;
					}
					else
					{
						a++;
					}
				}
			}
			printf("FRICTION_END\n");
		}
		
		
			
	};
}

