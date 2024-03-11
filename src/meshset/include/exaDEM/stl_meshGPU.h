#pragma once

#include <exaDEM/face.h>
#include <sstream>
#include <exanb/core/basic_types.h>

#include <onika/cuda/cuda.h> // mots cles specifiques
#include <onika/memory/allocator.h> // cudaMMVector

#include <cfloat>

namespace exaDEM
{
	using namespace exanb;
	/**
	* @brief Represents a collection of 3D meshes consisting of faces
	*
	* The 'stl_meshes' struct represents a collection of 3D meshes composed of faces. Is an optimization of the 'stl_mesh' data layout to fit GPU computing.
	* It provides methods fro adding faces to the mesh, accessing the mesh data, and retrieving individual faces.
	
	*/			

	/**
	 * @brief Represents a 3D mesh consisting of faces.
	 *
	 * The `stl_mesh` struct represents a 3D mesh composed of faces. It provides methods for adding faces to the mesh,
	 * accessing the mesh data, and retrieving individual faces.
	 	 
	 */
	 
	
	
	struct stl_mesh
	{
		std::vector< double > vx;/**< x coordinate of the faces's vertices.*/
		std::vector< double > vy;/**< y coordinate of the faces's vertices.*/
		std::vector< double > vz;/**< z coordinate of the faces's vertices.*/
		std::vector< double > nx;/**< x coordinate of the faces's normal vector.*/
		std::vector< double > ny;/**< y coordinate of the faces's normal vector.*/
		std::vector< double > nz;/**< z coordiante of the faces's normal vector.*/
		std::vector< double > offsets;/**< Faces's offset.*/
		std::vector< int > start;/**< Vector used to keep track of the vertices's indexes*/
		std::vector< int > end;/**< Vector used to keep track of the vertices's indexes.*/
		std::vector< Box > m_boxes;/**< Faces's boxes.*/
		std::vector<std::vector<int>> indexes;/**< Indexes to construct the GPU data layout. */
		
		//GPU DATA
		std::vector< int > offs_cells;/**< Vector used to construct the GPU data/*/
		std::vector< int > nb_max_meshes;/**< Vector used to construct the GPU data*/
		std::vector< int > nb_meshes;/**< Vector used to construct the GPU data.*/
		
		onika::memory::CudaMMVector<int> cells_GPU;/**< Cells that intersperse with at least of face.*/
		onika::memory::CudaMMVector< int > nb_faces_GPU;/**< Number of faces for each cell in the cells_GPU vector/*/
		onika::memory::CudaMMVector< int > offs_faces_GPU;/**< Vector used to keep track of the track of the indexes for each cell's faces.*/
		onika::memory::CudaMMVector< double > nx_GPU;/**< x coordinate of the faces's normal vector for the faces that intersperse with a cell.*/
		onika::memory::CudaMMVector< double > ny_GPU;/**< y coordinate of the faces's normal vector for the faces that intersperse with a cell.*/
		onika::memory::CudaMMVector< double > nz_GPU;/**< z coordinate of the faces's normal vector for the faces that intersperse with a cell.*/
		onika::memory::CudaMMVector< double > offsets_GPU;/**< Faces's offset for the faces that intersperse with a cell.*/
		onika::memory::CudaMMVector< double > vx_GPU;/**< x coordinate of the faces's vertices for the faces that interperse with a cell.*/
		onika::memory::CudaMMVector< double > vy_GPU;/**< y coordinate of the faces's vertices for the faces that intersperse with a cell.*/
		onika::memory::CudaMMVector< double > vz_GPU;/**< z coordinate of the faces's vertices for the faces that intersperse with a cell.*/
		onika::memory::CudaMMVector< int > offs_vertices_GPU;/**< Vector used to keep track of the indexes for each cell's faces's vertices.*/
		onika::memory::CudaMMVector< int > nb_vertices_GPU;/**< Number of vertices for each face taht intersperse with a cell.*/
		
		//offs_mesh_GPU
		//nb_meshes_GPU
		/**
		 * @brief Reads mesh data from an STL file and populates the mesh.
		 *
		 * The `read_stl` function reads mesh data from an STL file specified by `file_name` and populates the mesh with
		 * vertices and faces. It also calculates the number of vertices and faces in the mesh and provides information about
		 * the mesh's characteristics.
		 *
		 * @param file_name The name of the STL file to read.
		 */
		void read_stl(std::string file_name)
		{
			std::ifstream input( file_name.c_str() );
			std::string first;
			std::vector<Vec3d> vertices;
			Vec3d vertex;
			int nv = 0;
			int nf = 0;
			
			for( std::string line; getline( input, line ); )
			{
				input >> first;
				if(first == "outer")
				{
					int nv_2=0;
					start.push_back(nv);
					bool build_face = true;
					while(build_face)
					{
						
						getline(input, line);
						input >> first;
						if(first == "vertex")
						{
							input >> vertex.x >> vertex.y >> vertex.z;
							vertices.push_back(vertex);	
							nv++;
							nv_2++;
						}
						else if (first != "endloop")
						{
							std::cout << "error when reading stl file, it should be endloop and not " << first << std::endl;
							build_face = false;
						}
						else { 
							build_face = false;
						}
					}
					auto [_normal, _offset, _exist] = compute_normal_and_offset(vertices);
					offsets.push_back(_offset);
					nx.push_back(_normal.x);
					ny.push_back(_normal.y);
					nz.push_back(_normal.z);
					for(auto v: vertices){
						vx.push_back(v.x);
						vy.push_back(v.y);
						vz.push_back(v.z);
					}
					end.push_back(nv);
					vertices.clear();
					nf++;
					nb_meshes.push_back(nv_2);
				}
			}
			std::cout << "Mesh: " << file_name << " - number of vertices: " << nv << " - number of faces: " << nf << std::endl;
		}
		
		std::tuple<Vec3d, double, bool> compute_normal_and_offset(std::vector<Vec3d> vertices) 
		{
			Vec3d _normal;
			double dist = 0;
			if (vertices.size() < 3) {
				// need three vertices at least
				return std::make_tuple(_normal, dist, false);
			}

			Vec3d v1 = vertices[1] - vertices[0];
			Vec3d v2 = vertices[2] - vertices[0];
			_normal = cross(v1, v2);
			_normal = _normal / exanb::norm(_normal);
			dist = dot(_normal, vertices[0]);
			return std::make_tuple(_normal, dist, true);
		}

		/**
		 * @brief Builds bounding boxes for all faces in the mesh.
		 *
		 * The `build_boxes` function calculates and stores bounding boxes for all faces in the mesh. It ensures that the mesh
		 * has at least one face and performs this computation in parallel when possible.
		 */
		void build_boxes()
		{
			const int size = start.size();
			assert(size > 0);
			m_boxes.resize(size);
#pragma omp parallel for
			for(int i = 0 ; i < size ; i++)
			{
				std::vector<Vec3d> vertices;
				for(int j=start[i]; j<end[i]; j++){
					Vec3d v= {vx[j], vy[j], vz[j]};
					vertices.push_back(v);
				}	
				m_boxes[i]= create_box(vertices);
				vertices.clear();
			}
		}
		
		Box create_box(std::vector<Vec3d> vertices)
		{
			Vec3d inf = vertices[0];
			Vec3d sup = inf;
			for(auto vertex : vertices)
			{
				inf.x = std::min(inf.x,vertex.x);
				inf.y = std::min(inf.y,vertex.y);
				inf.z = std::min(inf.z,vertex.z);
				sup.x = std::max(sup.x,vertex.x);
				sup.y = std::max(sup.y,vertex.y);
				sup.z = std::max(sup.z,vertex.z);
			}
			Box res = {inf,sup};
			return res;
		}
		
		

		/**
		 * @brief Updates the indexes based on the bounding box of a face.
		 *
		 * The `update_indexes` function updates the indexes based on the bounding box of a face specified by `idBox` and `b1`.
		 * It iterates through the mesh's bounding boxes, compares them with `b1`, and adds relevant indexes to the specified `id`.
		 *
		 * @param id The index to update with relevant indexes.
		 * @param b1 The bounding box of the face for comparison.
		 */
		void update_indexes(const int id, Box& b1)
		{
			for( size_t idBox = 0 ; idBox < m_boxes.size() ; idBox++ )
			{
				auto& b2 = m_boxes[idBox];
				if (( b1.sup.x >= b2.inf.x && b1.inf.x <= b2.sup.x) &&
						(b1.sup.y >= b2.inf.y && b1.inf.y <= b2.sup.y) &&
						(b1.sup.z >= b2.inf.z && b1.inf.z <= b2.sup.z))
				{
					indexes[id].push_back(idBox);
				}
			}
		}
		
	       /**
	        * @brief Constructs the data layout optimized for the GPU computing from the indexes vector.
	        *
	        * The 'update_GPU' fuctions updates the vectors used by ApplyHookeSTLMeshesFunctor following the update of the indexes vector.
	        *
	        */
		void update_GPU(){
		
			/**WHICH CELLS*/
			int nb_max_faces=0;
			int nb_faces= 0;
			int nb_cells= 0;
			cells_GPU.resize(0);
			nb_faces_GPU.resize(0);
			for(size_t i = 0; i < indexes.size(); i++){
				if(indexes[i].size() > 0){
					nb_faces+= indexes[i].size();
					nb_cells++;
					cells_GPU.push_back(i);
					nb_faces_GPU.push_back(indexes[i].size());
					if(indexes[i].size() > nb_max_faces) nb_max_faces = indexes[i].size();
				}
			}
			
			/**WHICH FACES*/
			offs_cells.clear();
			offs_cells.resize(nb_cells);
			offs_faces_GPU.clear();
			offs_faces_GPU.resize(nb_faces);
			nx_GPU.resize(0);
			ny_GPU.resize(0);
			nz_GPU.resize(0);
			offsets_GPU.resize(0);
			nb_max_meshes.clear();
			nb_max_meshes.resize(nb_max_faces);
			int nb_vertices=0;
			for(int i = 0; i < nb_max_faces; i++){
				int max_meshes=0;
				for(int j = 0; j < nb_cells; j++){
					if(i < nb_faces_GPU[j]){
						nx_GPU.push_back(nx[indexes[cells_GPU[j]][i]]);
						ny_GPU.push_back(ny[indexes[cells_GPU[j]][i]]);
						nz_GPU.push_back(nz[indexes[cells_GPU[j]][i]]);
						offsets_GPU.push_back(offsets[indexes[cells_GPU[j]][i]]);
						nb_vertices_GPU.push_back(nb_meshes[indexes[cells_GPU[j]][i]]);
						if(nb_meshes[indexes[cells_GPU[j]][i]] > max_meshes) max_meshes= nb_meshes[indexes[cells_GPU[j]][i]];
						if( i > 0 ){
							offs_faces_GPU[offs_cells[j]]= nx_GPU.size() - 1;
						}
						offs_cells[j]= nx_GPU.size() - 1;
						nb_vertices+= nb_meshes[indexes[cells_GPU[j]][i]];
					} 
				}
				nb_max_meshes[i] = max_meshes;
			}
			
			/**WHICH VERTICES.*/
			vx_GPU.resize(0);
			vy_GPU.resize(0);
			vz_GPU.resize(0);
			offs_vertices_GPU.clear();
			offs_vertices_GPU.resize(nb_vertices);
			for(int i = 0; i < nb_max_faces; i++){
				for(int j = 0; j < nb_max_meshes[i]; j++){
					for(int z = 0; z < nb_cells; z++){
						if(i < nb_faces_GPU[z] && j < nb_meshes[indexes[cells_GPU[z]][i]]){
							vx_GPU.push_back(vx[start[indexes[cells_GPU[z]][i]]+j]);
							vy_GPU.push_back(vy[start[indexes[cells_GPU[z]][i]]+j]);
							vz_GPU.push_back(vz[start[indexes[cells_GPU[z]][i]]+j]);
							if(i > 0 || j > 0){
								offs_vertices_GPU[offs_cells[z]]= vx_GPU.size() - 1;
							}
							offs_cells[z]= vx_GPU.size() - 1;
						} 
					}
				}
			}
		}
		
		
	};
}
