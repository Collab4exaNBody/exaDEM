#pragma once

#include <exanb/core/basic_types.h>

#include <onika/cuda/cuda.h> // mots cles specifiques
#include <onika/memory/allocator.h> // cudaMMVector
#include <onika/cuda/stl_adaptors.h>
#include <exanb/core/basic_types.h>

#include <exaDEM/stl_meshes.h>

//#include <exanb/core/grid.h>

namespace exaDEM
{
	using namespace exanb;
	
	
	struct Interaction_Particle
	{
		int p_i;
		size_t cell_i;
		std::vector< int > faces_idx;
		
	};
	
	
	struct Interactions
	{
		int nb_particles;//Number of particles
		
		std::vector<int> pa;
		std::vector<int> cella;
		std::vector<int> id_a;
		
		std::vector< std::vector<int>> faces_idx;
		std::vector< std::vector<double>> nx;
		std::vector< std::vector<double>> ny;
		std::vector< std::vector<double>> nz;
		std::vector< std::vector<double>> offsets;
		std::vector< std::vector<int>> num_vertices;
		
		std::vector< std::vector<double>> ftx;
		std::vector< std::vector<double>> fty;
		std::vector< std::vector<double>> ftz;
		
		onika::memory::CudaMMVector<int> pa_GPU;
		onika::memory::CudaMMVector<int> cella_GPU;
		onika::memory::CudaMMVector<int> faces_idx_GPU;
		onika::memory::CudaMMVector<int> faces_build_GPU;
		onika::memory::CudaMMVector<double> nx_GPU;
		onika::memory::CudaMMVector<double> ny_GPU;
		onika::memory::CudaMMVector<double> nz_GPU;
		onika::memory::CudaMMVector<double> offsets_GPU;
		onika::memory::CudaMMVector<int> num_vertices_GPU;
		onika::memory::CudaMMVector<double> ftx_GPU;
		onika::memory::CudaMMVector<double> fty_GPU;
		onika::memory::CudaMMVector<double> ftz_GPU;
		onika::memory::CudaMMVector<double> vx_GPU;
		onika::memory::CudaMMVector<double> vy_GPU;
		onika::memory::CudaMMVector<double> vz_GPU;
		
		onika::memory::CudaMMVector<int> pa_GPU2;
		onika::memory::CudaMMVector<int> cella_GPU2;
		onika::memory::CudaMMVector<int> faces_idx_GPU2;
		onika::memory::CudaMMVector<double> nx_GPU2;
		onika::memory::CudaMMVector<double> ny_GPU2;
		onika::memory::CudaMMVector<double> nz_GPU2;
		onika::memory::CudaMMVector<double> offsets_GPU2;
		onika::memory::CudaMMVector<int> num_vertices_GPU2;
		onika::memory::CudaMMVector<double> ftx_GPU2;
		onika::memory::CudaMMVector<double> fty_GPU2;
		onika::memory::CudaMMVector<double> ftz_GPU2;
		onika::memory::CudaMMVector<double> vx_GPU2;
		onika::memory::CudaMMVector<double> vy_GPU2;
		onika::memory::CudaMMVector<double> vz_GPU2;
		onika::memory::CudaMMVector<double> posx;
		onika::memory::CudaMMVector<double> posy;
		onika::memory::CudaMMVector<double> posz;
		
		onika::memory::CudaMMVector<int> contact;
		onika::memory::CudaMMVector<int> which_particle;
		onika::memory::CudaMMVector<int> which_particle2;
		
		onika::memory::CudaMMVector<int> add_particle;
		
		std::vector<int> start;
		std::vector<int> end;
		std::vector<int> num_faces;
		
		int nb_interactions = 0;
		
		//Reset the attributes
		void reset(){
			nb_particles = 0;
			
			
		}
		
		
		void add_particle_func(int p, int cell, std::vector<int> faces, int ida, stl_meshes meshes)
		{
			pa.push_back(p);
			cella.push_back(cell);
			id_a.push_back(ida);
			
			faces_idx.resize(nb_particles + 1);
			nx.resize(nb_particles + 1);
			ny.resize(nb_particles + 1);
			nz.resize(nb_particles + 1);
			offsets.resize(nb_particles + 1);
			num_vertices.resize(nb_particles + 1);
			ftx.resize(nb_particles + 1);
			fty.resize(nb_particles + 1);
			ftz.resize(nb_particles + 1);
			
			auto& faces2 = faces_idx[nb_particles];
			auto& nx2 = nx[nb_particles];
			auto& ny2 = ny[nb_particles];
			auto& nz2 = nz[nb_particles];
			auto& offsets2 = offsets[nb_particles];
			auto& n_vertices = num_vertices[nb_particles];
			auto& ftx2 = ftx[nb_particles];
			auto& fty2 = fty[nb_particles];
			auto& ftz2 = ftz[nb_particles];
			
			for(int i = 0; i < faces.size(); i++)
			{
				int idx = faces[i];
				faces2.push_back(idx);
				nx2.push_back(meshes.nx[idx]);
				ny2.push_back(meshes.ny[idx]);
				nz2.push_back(meshes.nz[idx]);
				offsets2.push_back(meshes.offsets[idx]);
				n_vertices.push_back(meshes.nb_vertices[idx]);
				ftx2.push_back(0);
				fty2.push_back(0);
				ftz2.push_back(0);
			}
			
			nb_particles++;
		}
		
		void swap_a(int i, int j)
		{
			int temp_pa = pa[i];
			int temp_cella = cella[i];
			int temp_ida = id_a[i];
			
			int size_i = faces_idx[i].size();
			int size_j = faces_idx[j].size();
			
			std::vector<int> temp_faces;
			temp_faces.resize(size_i);
			std::vector<double> temp_nx;
			temp_nx.resize(size_i);
			std::vector<double> temp_ny;
			temp_ny.resize(size_i);
			std::vector<double> temp_nz;
			temp_nz.resize(size_i);
			std::vector<double> temp_offsets;
			temp_offsets.resize(size_i);
			std::vector<int> temp_nbVertices;
			temp_nbVertices.resize(size_i);
			std::vector<double> temp_ftx;
			temp_ftx.resize(size_i);
			std::vector<double> temp_fty;
			temp_fty.resize(size_i);
			std::vector<double> temp_ftz;
			temp_ftz.resize(size_i);
			
			for(int z = 0; z < size_i; z++)
			{
				temp_faces[z] = faces_idx[i][z];
				temp_nx[z] = nx[i][z];
				temp_ny[z] = ny[i][z];
				temp_nz[z] = nz[i][z];
				temp_offsets[z] = offsets[i][z];
				temp_nbVertices[z] = num_vertices[i][z];
				temp_ftx[z] = ftx[i][z];
				temp_fty[z] = fty[i][z];
				temp_ftz[z] = ftz[i][z];
			}
			
			pa[i] = pa[j];
			pa[j] = temp_pa;
			
			cella[i] = cella[j];
			cella[j] = temp_cella;
			
			id_a[i] = id_a[j];
			id_a[j] = temp_ida;
			
			faces_idx[i].clear();
			faces_idx[i].resize(size_j);
			
			nx[i].clear();
			nx[i].resize(size_j);
			
			ny[i].clear();
			ny[i].resize(size_j);
			
			nz[i].clear();
			nz[i].resize(size_j);
			
			offsets[i].clear();
			offsets[i].resize(size_j);
			
			num_vertices[i].clear();
			num_vertices[i].resize(size_j);
			
			ftx[i].clear();
			ftx[i].resize(size_j);
			
			fty[i].clear();
			fty[i].resize(size_j);
			
			ftz[i].clear();
			ftz[i].resize(size_j);
			
			for(int z = 0; z < size_j; z++)
			{
				faces_idx[i][z] = faces_idx[j][z];
				nx[i][z] = nx[j][z];
				ny[i][z] = ny[j][z];
				nz[i][z] = nz[j][z];
				offsets[i][z] = offsets[j][z];
				num_vertices[i][z] = num_vertices[j][z];
				ftx[i][z] = ftx[j][z];
				fty[i][z] = fty[j][z];
				ftz[i][z] = ftz[j][z];
			}
			
			faces_idx[j].clear();
			faces_idx[j].resize(size_i);
			
			nx[j].clear();
			nx[j].resize(size_i);
			
			ny[j].clear();
			ny[j].resize(size_i);
			
			nz[j].clear();
			nz[j].resize(size_i);
			
			offsets[j].clear();
			offsets[j].resize(size_i);
			
			num_vertices[j].clear();
			num_vertices[j].resize(size_i);
			
			ftx[j].clear();
			ftx[j].resize(size_i);
			
			fty[j].clear();
			fty[j].resize(size_i);
			
			ftz[j].clear();
			ftz[j].resize(size_i);
			
			
			for(int z = 0; z < size_i; z++)
			{
				faces_idx[j][z] = temp_faces[z];
				nx[j][z] = temp_nx[z];
				ny[j][z] = temp_ny[z];
				nz[j][z] = temp_nz[z];
				offsets[j][z] = temp_offsets[z];
				num_vertices[j][z] = temp_nbVertices[z];
				ftx[j][z] = temp_ftx[z];
				fty[j][z] = temp_fty[z];
				ftz[j][z] = temp_ftz[z];
			}
		}
		
		int partition_a(int low, int high)
		{
			int pivot = id_a[high];
			int i = low - 1;
			
			for(int j = low; j < high; j++)
			{
				if(id_a[j] < pivot)
				{
					i++;
					swap_a(i ,j);
				}
			}
			
			swap_a(i + 1, high);
			return i + 1;
		}
		
		void quickSort_a(int low, int high)
		{
			if(low < high)
			{
				int pi = partition_a(low, high);
				quickSort_a(low, pi - 1);
				quickSort_a(pi + 1, high);
			}
		}
		
		void quickSort()
		{
			int n = nb_particles;
			quickSort_a(0, n - 1);
			
		}
		
		void init_friction(Interactions old)
		{
			if(nb_particles > 0 && old.nb_particles > 0)
			{
				int a = 0;
				int b = 0;
				
				while(a < nb_particles && b < old.nb_particles)
				{
					if(id_a[a] == old.id_a[b])
					{
						int a2 = 0;
						int b2 = 0;
						
						while(a2 < faces_idx[a].size() && b2 < old.faces_idx[b].size())
						{
							if(faces_idx[a][a2] == old.faces_idx[b][b2])
							{
								ftx[a][a2] = old.ftx[b][b2];
								fty[a][a2] = old.fty[b][b2];
								ftz[a][a2] = old.ftz[b][b2];
								a2++;
								b2;;
							} else if(faces_idx[a][a2] > old.faces_idx[b][b2])
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
					else if(id_a[a] > old.id_a[b])
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
		
		void maj_friction()
		{
			for(int i = 0; i < nb_particles; i++)
			{
				int z = 0;
				for(int j = start[z]; j < end[z]; j++)
				{
					ftx[i][z] = ftx_GPU[j];
					fty[i][z] = fty_GPU[j];
					ftz[i][z] = ftz_GPU[j];
					z++;
				}
			}
		}
		
		
		void vertices(stl_meshes meshes)
		{
			faces_build_GPU.resize(nb_interactions);
			std::vector<int> index;
			std::vector<int> start_meshes;
			for(int i = 0; i < nb_interactions; i++)
			{
				int idx = faces_idx_GPU[i];
				int real_index;
				bool find = false;
				int j = 0;
				
				while(j < index.size() && find == false)
				{
					if(index[j] == idx)
					{
						real_index = j;
						find = true;
					}
					
					j++;
				}
				
				if(find)
				{
					faces_build_GPU[i] = start_meshes[real_index];
				}
				else
				{
					index.push_back(idx);
					start_meshes.push_back(vx_GPU.size());
					faces_build_GPU.push_back(vx_GPU.size());
					int start = meshes.start[idx];
					int end = meshes.end[idx];
					
					for(int j = start; j < end; j++)
					{
						vx_GPU.push_back(meshes.vx[j]);
						vy_GPU.push_back(meshes.vy[j]);
						vz_GPU.push_back(meshes.vz[j]);
					}
				}
			} 
		}
		
		void init_GPU(stl_meshes meshes)
		{
			start.resize(nb_particles);
			end.resize(nb_particles);
			num_faces.resize(nb_particles);
			
			for(int i = 0; i < nb_particles; i++)
			{
				num_faces[i] = faces_idx[i].size();
			}
			
			for(int i = 0; i < nb_particles; i++)
			{
				nb_interactions+= num_faces[i];
			}
			
			start[0] = 0;
			for(int i = 1; i < nb_particles; i++)
			{
				start[i] = num_faces[i - 1] + start[i - 1];
			}
			
			end[nb_particles - 1] = nb_interactions;
			for(int i = 0; i < nb_particles - 1; i++)
			{
				end[i] = start[i + 1];
			}
			
			
			pa_GPU.resize(nb_interactions);
			cella_GPU.resize(nb_interactions);
			faces_idx_GPU.resize(nb_interactions);
			nx_GPU.resize(nb_interactions);
			ny_GPU.resize(nb_interactions);
			nz_GPU.resize(nb_interactions);
			offsets_GPU.resize(nb_interactions);
			num_vertices_GPU.resize(nb_interactions);
			ftx_GPU.resize(nb_interactions);
			fty_GPU.resize(nb_interactions);
			ftz_GPU.resize(nb_interactions);
			which_particle.resize(nb_interactions);
			
			
			for(int i= 0; i < nb_particles; i++)
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
					faces_idx_GPU[j] = faces_idx[i][z];
					nx_GPU[j] = nx[i][z];
					ny_GPU[j] = ny[i][z];
					nz_GPU[j] = nz[i][z];
					offsets_GPU[j] = offsets[i][z];
					int nb = num_vertices[i][z];
					num_vertices_GPU[j] = nb;
					ftx_GPU[j] = ftx[i][z];
					fty_GPU[j] = fty[i][z];
					ftz_GPU[j] = ftz[i][z];
					which_particle[j] = i;
					z++;
				}
			}
			
			vertices(meshes);
			
			pa_GPU2.resize(nb_interactions);
			cella_GPU2.resize(nb_interactions);
			faces_idx_GPU2.resize(nb_interactions);
			nx_GPU2.resize(nb_interactions);
			ny_GPU2.resize(nb_interactions);
			nz_GPU2.resize(nb_interactions);
			offsets_GPU2.resize(nb_interactions);
			num_vertices_GPU2.resize(nb_interactions);
			ftx_GPU2.resize(nb_interactions);
			fty_GPU2.resize(nb_interactions);
			ftz_GPU2.resize(nb_interactions);
			contact.resize(nb_interactions);
			which_particle2.resize(nb_interactions);
			posx.resize(nb_interactions);
			posy.resize(nb_interactions);
			posz.resize(nb_interactions);
			
			add_particle.resize(nb_particles);
			
			vx_GPU2.resize(vx_GPU.size());
			vy_GPU2.resize(vy_GPU.size());
			vz_GPU2.resize(vz_GPU.size());
			
		}
		
	};
		
	
};
