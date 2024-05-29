#pragma once

#include <exanb/core/basic_types.h>
#include <onika/cuda/cuda.h> // mots cles specifiques
#include <onika/memory/allocator.h> // cudaMMVector
#include <onika/cuda/stl_adaptors.h>
#include <exanb/core/basic_types.h>

namespace exaDEM
{

	using namespace exanb;
	
	struct Interactions_PP//List of interactions between two neighboring particles
	{
		int nb_particles=0;//Number of  particles
		
		std::vector<int> pa;//List of particle postion inside the cell
		
		std::vector<int> cella;//List of cells
		
		std::vector<int> id_a;//List of particle identifiers
		
		//Neighboring particles
		std::vector<std::vector<int>> pb;
		std::vector<std::vector<int>> cellb;
		std::vector< std::vector<int>> id_b;
		
		std::vector<std::vector<Vec3d>> ft_pair;//Storage of friction between two particles
		
		std::vector<int> size_nbh;//Number of neighbors for a partices
		
		//Indexes for GPU's lists
		std::vector<int> start;
		std::vector<int> end;
		
		
		int nb_interactions=0;//Numbe of interactions
		
		//GPU Lists
		onika::memory::CudaMMVector<int> pa_GPU;
		onika::memory::CudaMMVector<int> cella_GPU;
		onika::memory::CudaMMVector<int> pb_GPU;
		onika::memory::CudaMMVector<int> cellb_GPU;
		//Friction
		onika::memory::CudaMMVector<double> ftx_GPU;
		onika::memory::CudaMMVector<double> fty_GPU;
		onika::memory::CudaMMVector<double> ftz_GPU;
		
		onika::memory::CudaMMVector<int> pa_GPU2;
		onika::memory::CudaMMVector<int> cella_GPU2;
		onika::memory::CudaMMVector<int> pb_GPU2;
		onika::memory::CudaMMVector<int> cellb_GPU2;
		//Friction
		onika::memory::CudaMMVector<double> ftx_GPU2;
		onika::memory::CudaMMVector<double> fty_GPU2;
		onika::memory::CudaMMVector<double> ftz_GPU2;
		
		//Reset the lists
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
			
			size_nbh.clear();
			size_nbh.resize(0);
			start.clear();
			start.resize(0);
			end.clear();
			end.resize(0);
			
			nb_interactions = 0;
			pa_GPU.clear();
			pa_GPU.resize(0);
			cella_GPU.clear();
			cella_GPU.resize(0);
			pb_GPU.clear();
			pb_GPU.resize(0);
			cellb_GPU.clear();
			cellb_GPU.resize(0);
			ftx_GPU.clear();
			ftx_GPU.resize(0);
			fty_GPU.clear();
			fty_GPU.resize(0);
			ftz_GPU.clear();
			ftz_GPU.resize(0);
			
			pa_GPU2.clear();
			cella_GPU2.clear();
			pb_GPU2.clear();
			cellb_GPU2.clear();
			ftx_GPU2.clear();
			fty_GPU2.clear();
			ftz_GPU2.clear();
		}
		
		int binarySearch(std::vector<int> arr, int size, int target, int cell, int p, bool b, int idx) {
		
   			int low = 0;
    			int high = size - 1;

    			while (low <= high) {
        			int mid = low + (high - low) / 2; // Utilisation de cette formule pour éviter le dépassement d'entier

        			if (arr[mid] == target) {
            				if(b){ return mid; }
            				else {  if(cellb[idx][mid] == cell && pb[idx][mid] == p) return mid; }
       				} else if (arr[mid] < target) {
           				low = mid + 1; // Ignorer la moitié gauche
       				} else {
            				high = mid - 1; // Ignorer la moitié droite
        			}
    				}
    			    return -1; // Élément non trouvé
		}
		

		//ADD PARTICLES
		void add_particle(int p, int cell, std::vector<std::pair<int,int>> nbh, int ida, std::vector<int> idb)
		{
			//printf("ICI\n");
			pa.push_back(p);
			cella.push_back(cell);
			id_a.push_back(ida);
			
			//nb_particles++;
			
			pb.resize(nb_particles+1);
			cellb.resize(nb_particles+1);
			ft_pair.resize(nb_particles+1);
			id_b.resize(nb_particles+1);
				
			
			auto& p_b= pb[nb_particles];
			auto& cell_b= cellb[nb_particles];
			auto& ft= ft_pair[nb_particles];
			auto& idb_ = id_b[nb_particles];
			
			for(int i = 0; i < nbh.size(); i++)
			{
				p_b.push_back(nbh[i].first);
				cell_b.push_back(nbh[i].second);
				ft.push_back({0, 0, 0});
				idb_.push_back(idb[i]);
			}
			nb_particles++;
		}
		
		int print()
		{
			int r = 0;
			for(int i = 0; i < nb_particles; i++)
			{
				r+= pb[i].size();
			}
			return r;
		}
		
		void set()
		{
			int nb = 0;
			std::vector<int> temp_pa;
			std::vector<int> temp_cella;
			std::vector<int> temp_ida;
			std::vector<std::vector<int>> temp_pb;
			std::vector<std::vector<int>> temp_cellb;
			std::vector<std::vector<int>> temp_idb;
			std::vector<std::vector<Vec3d>> temp_ft;
			for(int i = 0; i < nb_particles; i++)
			{
				int ida = id_a[i];
				int p_a = pa[i];
				int cell_a = cella[i];
				auto nbh_pb = pb[i];
				auto nbh_cellb = cellb[i];
				auto nbh_idb = id_b[i];
				//if(nbh_idb[0] > ida)
				//{
					//temp_pa.push_back(p_a);
					//temp_cella.push_back(cell_a);
					//temp_ida.push_back(ida);
					bool b = false;
					//int i = 0;
					//while(b && i < nbh_idb.size())
					for(int j = 0; j < nbh_idb.size(); j++)
					{
						if(nbh_idb[j] > ida)
						{
							if(temp_pb.size() == nb)
							{
								temp_pb.resize(nb+1);
								temp_cellb.resize(nb+1);
								temp_idb.resize(nb+1);
								temp_ft.resize(nb+1);
								b = true;
							}	
							temp_pb[nb].push_back(nbh_pb[j]);
							temp_cellb[nb].push_back(nbh_cellb[j]);
							temp_idb[nb].push_back(nbh_idb[j]);
							temp_ft[nb].push_back({0, 0, 0});
						}
					}
					if(b)
					{
						temp_pa.push_back(p_a);
						temp_cella.push_back(cell_a);
						temp_ida.push_back(ida);
						nb++;
					}
				//}
			}
			//nb_particles = nb;
			pa.clear();
			pa.resize(0);
			pa = temp_pa;
			cella.clear();
			cella.resize(0);
			cella = temp_cella;
			id_a.clear();
			id_a.resize(0);
			id_a = temp_ida;
			for(int i = 0; i < nb_particles; i++)
			{
				pb[i].clear();
				pb[i].resize(0);
				cellb[i].clear();
				cellb[i].resize(0);
				id_b[i].clear();
				id_b[i].resize(0);
				ft_pair[i].clear();
				ft_pair[i].resize(0);
			}
			pb = temp_pb;
			cellb = temp_cellb;
			id_b = temp_idb;
			ft_pair = temp_ft;
			printf("AVANT: %d, APRÈS: %d\n", nb_particles, nb);
			nb_particles = nb;
		}
		
		//QUICK SORT
		void swap_a(int i, int j)
		{
			//
			
			//if(id_a[i] == id_a[j] ){ printf("FLOOOOOOOOOOOOOOOOO IIIII:%d, JJJJJJJ:%d\n", i ,j); getchar(); };
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
			//if(id_a[i] == id_a[j]) printf("FLOOOOOOOOOOOOOOOOO\n");
			
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
			int n = pa.size();
			quickSort_a(0, n - 1);
			
			for(int i = 0; i < nb_particles; i++)
			{
				int m = pb[i].size();
				quickSort_b(i, 0, m - 1);
			}
		}
		//TRI QUICK_SORT
		
		//INITIALISATION DE LA FRICTION
		void init_friction(Interactions_PP mid)
		{
			if(nb_particles > 0 && mid.nb_particles > 0)
			{
				
				int a = 0;
				int b = 0; 
				
				while(a < nb_particles && b < mid.nb_particles)
				{
					if(id_a[a] == mid.id_a[b])
					{
						int a2 = 0;
						int b2 = 0;
						
						while(a2 < id_b[a].size() && b2 < mid.id_b[b].size())
						{
							
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
		}
		
				
		
		
		void erase(std::vector<int>& v, int index)
		{
			int size = v.size();
			std::vector<int> new_v;
			new_v.resize(size - 1);
			for(int i = 0; i < index; i++)
			{
				new_v[i] = v[i];
			}
			
			for(int i = index; i < size -1; i++)
			{
				new_v[i] = new_v[i+1];
			}
			
			v.clear();
			v.resize(0);
			for(int i = 0; i < new_v.size(); i++)
			{
				v.push_back(new_v[i]);
			}
			
		}
		
		void erase(std::vector<Vec3d>& v, int index)
		{
			printf("CICCINO\n");
			int size = v.size();
			std::vector<Vec3d> new_v;
			new_v.resize(size - 1);
			for(int i = 0; i < index; i++)
			{
				new_v[i] = v[i];
			}
			
			for(int i = index; i < size -1; i++)
			{
				new_v[i] = new_v[i+1];
			}
			printf("CICCINO2\n");
			v.clear();
			printf("CICCINO3\n");
			v.resize(0);
			printf("CICCINO4\n");
			for(int i = 0; i < new_v.size(); i++)
			{
				v.push_back(new_v[i]);
			}
			printf("CICCINO5\n");
		}
		
		void erase(std::vector<std::vector<int>>& v, int index)
		{
			int size = v.size();
			std::vector<std::vector<int>> new_v;
			new_v.resize(size - 1);
			for(int i = 0; i < index; i++)
			{
				for(auto j: v[i])
				{
					new_v[i].push_back(j);
				}
			}
			for(int i = index; i < size - 1; i++)
			{
				for(auto j: v[i+1])
				{
					new_v[i].push_back(j);
				}
			}
			
			for(int i = 0; i < v.size(); i++)
			{
				v[i].clear();
				v[i].resize(0);
			}
			
			v.clear();
			v.resize(0);
			v = new_v;
			
		}
		
		void erase(std::vector<std::vector<Vec3d>>& v, int index)
		{
			int size = v.size();
			std::vector<std::vector<Vec3d>> new_v;
			new_v.resize(size - 1);
			for(int i = 0; i < index; i++)
			{
				for(auto j: v[i])
				{
					new_v[i].push_back(j);
				}
			}
			for(int i = index; i < size - 1; i++)
			{
				for(auto j: v[i+1])
				{
					new_v[i].push_back(j);
				}
			}
			
			
			for(int i = 0; i < v.size(); i++)
			{
				v[i].clear();
				v[i].resize(0);
			}
			
			v.clear();
			v.resize(0);
			v = new_v;
		}
		
		
		
		/*void func()
		{
			//printf("GIT\n");
			int i = nb_particles;
			int j = 0;
			while(j < i)
			{
				int ida = id_a[j];
				//printf("SISEE: %d\n", id_b[j].size());
				printf("IDAAAAAAAAAAAAAAAAAAAA: %d PPPPPPPA: %d CEEEEEEEELA:%d\n", ida, pa[j], cella[j]);
				for(int z = 0; z < id_b[j].size(); z++)
				{
					//printf("Z: %d\n", z);
					int idb = id_b[j][z];
					printf("IDBBBBBBBBBBBBBBBBBBBBBBBBB: %d\n", idb);
					//if(idb > ida)
					//{
						int index1 = binarySearch(id_a, id_a.size(), idb);
						printf("UN\n");
						int index2 = binarySearch(id_b[index1], id_b[index1].size(), ida);
						printf("DEUX\n");
						erase(id_b[index1], index2);
						printf("TROIS\n");
						erase(pb[index1], index2);
						printf("QUATRE\n");
						erase(cellb[index1], index2);
						printf("CINQ\n");
						//erase(ft_pair[index1], index2);
						//printf("SIX\n");
						if(id_b[index1].size() == 0)
						{
							printf("OUI?\n");
							erase(id_a, index1);
							erase(pa, index1);
							erase(cella, index1);
							erase(id_b, index1);
							erase(pb, index1);
							erase(cellb, index1);
							erase(ft_pair, index1);
							nb_particles--;
							
						}
						//printf("SEPT\n");
					//}
				}
				
				i = nb_particles;
				j++;
				printf("ENDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDd\n");
			}
		}*/
		
		void test()
		{
			for(int i = 0; i < nb_particles; i++)
			{
			//	if(id_a[i] == 23){
					//printf("CELLA:%d, PA:%d\n", cella[i], pa[i]);
			//	/*}*/for(int j = 0; j < id_b[i].size(); j++){ printf("ID:%d PB:%d CELLB:%d\n", id_b[i][j], pb[i][j], cellb[i][j]);}}
			printf("ID: %d\n", id_a[i]);
			}
			getchar();
		}



		
		void maj_friction()
		{
			for(int i = 0; i < nb_particles; i++)
			{
				int z = 0;
				for(int j = start[i]; j < end[i]; j++)
				{
					Vec3d ft = {ftx_GPU2[j], fty_GPU2[j], ftz_GPU2[j]};
					ft_pair[i][z] = ft;
					z++;
				}
			}
		}
		
		
		
		//INITIALISATION DES LISTES SUR GPU
		void init_GPU()
		{
			size_nbh.resize(nb_particles);
			start.resize(nb_particles);
			end.resize(nb_particles);
			
			//#pragma omp parallel for
			for(int i = 0; i < nb_particles; i++)
			{
				size_nbh[i] = pb[i].size();
			}
			
			//#pragma omp parallel for shared(nb_interactions)
			for(int i = 0; i < nb_particles; i++)
			{
				nb_interactions+= size_nbh[i];
			}
			
			start[0] = 0;
			for(int i = 1; i < nb_particles; i++)
			{
				start[i] = size_nbh[i - 1] + start[i - 1];
			}
			
			end[nb_particles - 1] = nb_interactions;
			for(int i = 0; i < nb_particles - 1; i++)
			{
				end[i] = start[i + 1];
			} 
			
			pa_GPU.resize(nb_interactions);
			cella_GPU.resize(nb_interactions);
			pb_GPU.resize(nb_interactions);
			cellb_GPU.resize(nb_interactions);
			ftx_GPU.resize(nb_interactions);
			fty_GPU.resize(nb_interactions);
			ftz_GPU.resize(nb_interactions);
			
			//#pragma omp parallel for
			for(int i = 0; i < nb_particles; i++)
			{
				int p_a = pa[i];
				int cell_a = cella[i];
				int start_idx = start[i];
				int end_idx = end[i];
				
				int z = 0;
				
				for(int j = start_idx; j < end_idx; j++)
				{
					pa_GPU[j] = p_a;
					cella_GPU[j] = cell_a;
					pb_GPU[j] = pb[i][z];
					cellb_GPU[j] = cellb[i][z];
					Vec3d ft = ft_pair[i][z];
					ftx_GPU[j] = ft.x;
					fty_GPU[j] = ft.y;
					ftz_GPU[j] = ft.z;
					z++;
				}
			}
			
			pa_GPU2.resize(nb_interactions);
			cella_GPU2.resize(nb_interactions);
			pb_GPU2.resize(nb_interactions);
			cellb_GPU2.resize(nb_interactions);
			ftx_GPU2.resize(nb_interactions);
			fty_GPU2.resize(nb_interactions);
			ftz_GPU2.resize(nb_interactions);
		}
		
		
			
	};
}

