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
		int nb_particles_flow;
		
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
		onika::memory::CudaMMVector<int> potentiels;
		
		std::vector<int> start;
		std::vector<int> end;
		std::vector<int> num_faces;
		
		int nb_interactions = 0;
		
		//Reset the attributes
		void reset(){
			
			nb_particles = 0;
	
			nb_interactions = 0;
			
			
			pa.clear();
			pa.shrink_to_fit();
			
			cella.clear();
			cella.shrink_to_fit();
			
			id_a.clear();
			id_a.shrink_to_fit();
			
			faces_idx.clear();
			faces_idx.shrink_to_fit();
			
			nx.clear();
			nx.shrink_to_fit();
			
			ny.clear();
			ny.shrink_to_fit();
			
			nz.clear();
			nz.shrink_to_fit();
			
			offsets.clear();
			offsets.shrink_to_fit();
			
			num_vertices.clear();
			num_vertices.shrink_to_fit();
			
			ftx.clear();
			ftx.shrink_to_fit();
			
			fty.clear();
			fty.shrink_to_fit();
			
			ftz.clear();
			ftz.shrink_to_fit();
			
			pa_GPU.clear();
			pa_GPU.shrink_to_fit();
			
			cella_GPU.clear();
			cella_GPU.shrink_to_fit();
			
			faces_idx_GPU.clear();
			faces_idx_GPU.shrink_to_fit();
			
			faces_build_GPU.clear();
			faces_build_GPU.shrink_to_fit();
			
			nx_GPU.clear();
			nx_GPU.shrink_to_fit();
			
			ny_GPU.clear();
			ny_GPU.shrink_to_fit();
			
			nz_GPU.clear();
			nz_GPU.shrink_to_fit();
			
			offsets_GPU.clear();
			offsets_GPU.shrink_to_fit();
			
			num_vertices_GPU.clear();
			num_vertices_GPU.shrink_to_fit();
			
			ftx_GPU.clear();
			ftx_GPU.shrink_to_fit();
			
			fty_GPU.clear();
			fty_GPU.shrink_to_fit();
			
			ftz_GPU.clear();
			ftz_GPU.shrink_to_fit();
			
			vx_GPU.clear();
			vx_GPU.shrink_to_fit();
			
			vy_GPU.clear();
			vy_GPU.shrink_to_fit();
			
			vz_GPU.clear();
			vz_GPU.shrink_to_fit();
			
			pa_GPU2.clear();
			pa_GPU2.shrink_to_fit();
			
			cella_GPU2.clear();
			cella_GPU2.shrink_to_fit();
			
			faces_idx_GPU2.clear();
			faces_idx_GPU2.shrink_to_fit();
			
			nx_GPU2.clear();
			nx_GPU2.shrink_to_fit();
			
			ny_GPU2.clear();
			ny_GPU2.shrink_to_fit();
			
			nz_GPU2.clear();
			nz_GPU2.shrink_to_fit();
			
			offsets_GPU2.clear();
			offsets_GPU2.shrink_to_fit();
			
			num_vertices_GPU2.clear();
			num_vertices_GPU2.shrink_to_fit();
			
			ftx_GPU2.clear();
			ftx_GPU2.shrink_to_fit();
			
			fty_GPU2.clear();
			fty_GPU2.shrink_to_fit();
			
			ftz_GPU2.clear();
			ftz_GPU2.shrink_to_fit();
			
			vx_GPU2.clear();
			vx_GPU2.shrink_to_fit();
			
			vy_GPU2.clear();
			vy_GPU2.shrink_to_fit();
			
			vz_GPU2.clear();
			vz_GPU2.shrink_to_fit();
			
			posx.clear();
			posx.shrink_to_fit();
			
			posy.clear();
			posy.shrink_to_fit();
			
			posz.clear();
			posz.shrink_to_fit();
			
			contact.clear();
			contact.shrink_to_fit();
			
			which_particle.clear();
			which_particle.shrink_to_fit();
			
			which_particle2.clear();
			which_particle2.shrink_to_fit();
			
			add_particle.clear();
			add_particle.shrink_to_fit();
			
			start.clear();
			start.shrink_to_fit();
			
			end.clear();
			end.shrink_to_fit();
			
			num_faces.clear();
			num_faces.shrink_to_fit();
			
			potentiels.clear();
			potentiels.shrink_to_fit();
			
		}
		
		void resize(int rs)
		{
			nb_particles =rs;
			
			pa.resize(rs);
			cella.resize(rs);
			id_a.resize(rs);
			
			faces_idx.resize(rs);
			nx.resize(rs);
			ny.resize(rs);
			nz.resize(rs);
			offsets.resize(rs);
			num_vertices.resize(rs);
			ftx.resize(rs);
			fty.resize(rs);
			ftz.resize(rs);
		}
		
		
		void add_particle_func(int p, int cell, std::vector<int> faces, int ida, stl_meshes meshes)
		{
			pa.push_back(p);
			//pa[nb_particles_flow] = p;
			cella.push_back(cell);
			//cella[nb_particles_flow] = cell;
			id_a.push_back(ida);
			//id_a[nb_particles_flow] = ida;
			
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
			
			int fsize = faces.size();
			faces2.resize(fsize);
			nx2.resize(fsize);
			ny2.resize(fsize);
			nz2.resize(fsize);
			offsets2.resize(fsize);
			n_vertices.resize(fsize);
			ftx2.resize(fsize);
			fty2.resize(fsize);
			ftz2.resize(fsize);
			
			for(int i = 0; i < fsize; i++)
			{
				int idx = faces[i];
				faces2[i] = idx;
				nx2[i] = meshes.nx[idx];
				ny2[i] = meshes.ny[idx];
				nz2[i] = meshes.nz[idx];
				offsets2[i] = meshes.offsets[idx];
				n_vertices[i] = meshes.nb_vertices[idx];
				ftx2[i] = 0;
				fty2[i] = 0;
				ftz2[i] = 0;
			}
			
			nb_particles++;
			//printf("ADD END\n");
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
								//printf("INIT\n");
								ftx[a][a2] = old.ftx[b][b2];
								fty[a][a2] = old.fty[b][b2];
								ftz[a][a2] = old.ftz[b][b2];
								a2++;
								b2++;
								//printf("INIT END\n");
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
				for(int j = start[i]; j < end[i]; j++)
				{
					ftx[i][z] = ftx_GPU2[j];
					fty[i][z] = fty_GPU2[j];
					ftz[i][z] = ftz_GPU2[j];
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
					faces_build_GPU[i] = vx_GPU.size();
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
			if(nb_particles > 0)
			{
			start.resize(nb_particles);
			end.resize(nb_particles);
			num_faces.resize(nb_particles);
			
			for(int i = 0; i < nb_particles; i++)
			{
				num_faces[i] = faces_idx[i].size();
			}
			
			//printf("SLT\n");
			
			for(int i = 0; i < nb_particles; i++)
			{
				nb_interactions+= num_faces[i];
			}
			//printf("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII : %d\n", nb_interactions);
			//getchar();
			
			//printf("SLT2\n");
			//printf("NB PARTICULES : %d\n", nb_particles);
			start[0] = 0;
			//printf("NB PARTICULES : %d\n", nb_particles);
			for(int i = 1; i < nb_particles; i++)
			{
				//printf("III : %d\n", i);
				start[i] = num_faces[i - 1] + start[i - 1];
			}
			
			//printf("SLT3\n");
			
			end[nb_particles - 1] = nb_interactions;
			for(int i = 0; i < nb_particles - 1; i++)
			{
				end[i] = start[i + 1];
			}
			
			//printf("SLT4\n");
			
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
			potentiels.resize(nb_particles);
			
			vx_GPU2.resize(vx_GPU.size());
			vy_GPU2.resize(vy_GPU.size());
			vz_GPU2.resize(vz_GPU.size());
			}
			
		}
		
	};
		
	
};
