#pragma once

#include <exanb/core/basic_types.h>

#include <onika/cuda/cuda.h> // mots cles specifiques
#include <onika/memory/allocator.h> // cudaMMVector
#include <onika/cuda/stl_adaptors.h>
#include <exanb/core/basic_types.h>

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
		onika::memory::CudaMMVector<int> p_i;//List of Particle's ids
		onika::memory::CudaMMVector<size_t> cell_i;//List of Cell's ids
		onika::memory::CudaMMVector<int> faces;//Index of the face in the vectors associated to the faces
		onika::memory::CudaMMVector<int> start_particle;//First face associated to a given particle in the faces vector
		onika::memory::CudaMMVector<int> end_particle;//Last face associated a given particle in the faces vector
		onika::memory::CudaMMVector<int> num_faces;//List of number of faces associated to each particle
		
		//Attributes associated to the faces
		
		//Normal Vector
		onika::memory::CudaMMVector<double> nx;//List of X coordinates of the faces's normal vectors 
		onika::memory::CudaMMVector<double> ny;//List of Y coordiantes of the faces's normal vectors
		onika::memory::CudaMMVector<double> nz;//List of Z coordinates of the faces's normal vectors
		
		onika::memory::CudaMMVector<double> offsets;//List of offsets
		
		//Vertices
		onika::memory::CudaMMVector<int> start_face;//First vertex associated to a given face in the vertices's coordinates list
		onika::memory::CudaMMVector<int> end_face;//Last vertex associated to a given face in the vertices's coordinates list
		onika::memory::CudaMMVector<int> num_vertices;//List of number of vertices associated to each face
		onika::memory::CudaMMVector<double> vertex_x;//List of X coordiantes of the vertices
		onika::memory::CudaMMVector<double> vertex_y;//List of Y coordinates of the vertices
		onika::memory::CudaMMVector<double> vertex_z;//List of Z coordinates of the vertices
		
		//Reset the attributes
		void reset(){
			nb_particles = 0;
			p_i.resize(0);
			cell_i.resize(0);
			nx.resize(0);
			ny.resize(0);
			nz.resize(0);
			start_particle.resize(0);
			end_particle.resize(0);
			offsets.resize(0);
			start_face.resize(0);
			end_face.resize(0);
			vertex_x.resize(0);
			vertex_y.resize(0);
			vertex_z.resize(0);
			num_faces.resize(0);
			num_vertices.resize(0);
			faces.resize(0);
		}
		
		
		//Mathod used to fill the attributes associated to the particles
		void add_particle(Interaction_Particle i){
			p_i.push_back(i.p_i);
			cell_i.push_back(i.cell_i);
			num_faces.push_back(i.faces_idx.size());
			if(nb_particles == 0){
				start_particle.push_back(0);
			} else {
				start_particle.push_back(end_particle.back());
			}
			end_particle.push_back(start_particle.back() + num_faces.back());			
			for(int face = 0; face < num_faces.back(); face++){
				faces.push_back(i.faces_idx[face]);
			}
			nb_particles++;
		}
		
		//Method used to fill the attributes associated to the faces 
		void add_mesh(int nf,
				std::vector<double> nx_,
				std::vector<double> ny_,
				std::vector<double> nz_,
				std::vector<double> offsets_,
				std::vector<int> start_face_,
				std::vector<int> end_face_,
				std::vector<int> num_vertices_,
				std::vector<double> vx,
				std::vector<double> vy,
				std::vector<double> vz)
		{
			for(int i = 0; i < nf; i++){
				nx.push_back(nx_[i]);
				ny.push_back(ny_[i]);
				nz.push_back(nz_[i]);
				offsets.push_back(offsets_[i]);
				int start = start_face_[i];
				start_face.push_back(start_face_[i]);
				end_face.push_back(end_face_[i]);
				int size = num_vertices_[i];
				num_vertices.push_back(size);
				for(int j = 0; j < size; j++){
					vertex_x.push_back(vx[start+j]);
					vertex_y.push_back(vy[start+j]);
					vertex_z.push_back(vz[start+j]);
				}
			}
		}
	
	};
		
	
};
