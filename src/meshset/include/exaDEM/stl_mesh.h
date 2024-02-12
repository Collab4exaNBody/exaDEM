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
	
	struct stl_meshes  
	{
		/**<The number of meshes. */
		int nb_meshes= 0;
		/**<A collection of meshes with for each mesh: the number of faces, and for each face of the meshes: the normal vector of the face, the offset of the face, the number of vertices, and the coordinates of each vertex. */
		onika::memory::CudaMMVector< onika::memory::CudaMMVector < double > > m_meshes;
		/**<A collection of Box objects bounding the mesh for each mesh. */
		onika::memory::CudaMMVector< onika::memory::CudaMMVector< Box > > m_boxes;
		/**<Indexes for mesh data for each mesh. */
		onika::memory::CudaMMVector< onika::memory::CudaMMVector< onika::memory::CudaMMVector <int> > > indexes;
		
		/**
		 * @brief Reads mesh data from an STL file and populates the meshes.
		 *
		 * The `read_stl` function reads mesh data from an STL file specified by `file_name` and populates the mesh with
		 * vertices and faces. It also calculates the number of vertices and faces in the mesh and provides information about
		 * the mesh's characteristics.
		 *
		 * @param file_name The name of the STL file to read.
		 */
		void read2_stl(std::string file_name)
		{
			std::ifstream input( file_name.c_str() );
			std::string first;
			std::vector<Vec3d> vertices;
			//onika::memory::CudaMMVector <Vec3d> vertices;
			nb_meshes++; /**On incrémente le nombre de meshes. */
			m_meshes.resize(nb_meshes);/**Maj du nombre de meshes pour le vector m_meshes. */
			m_boxes.resize(nb_meshes);/**Maj du nombre de meshes pour le vector m_boxes. */
			indexes.resize(nb_meshes);/**Maj du nombre de meshes pour le vector indexes. */
			Vec3d vertex;
			int nv = 0;
			int nf = 0;
			/**L'index correspondant au nombre de faces pour ce mesh. */
			m_meshes[nb_meshes - 1].push_back(nf);
			
			/**L'index correspondant au nombre de vertex pour une face donnée. */
			//m_meshes[nb_meshes - 1].push_back(nv);
			//int nb_vertex_pos = m_meshes[nb_meshes-1].size() -1;
			int nb_vertex_pos = 0;
			
			/**The normal vector of the face. */
			Vec3d normal;
			/**The offset of the face. */
			double offset;
			for( std::string line; getline( input, line ); )
			{
				input >> first;
				if(first == "outer")
				{
					m_meshes[nb_meshes - 1].push_back(0);
					nb_vertex_pos = m_meshes[nb_meshes - 1].size() -1;
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
					//Face tmp (vertices);
					//this->add_face(tmp);
					auto [_normal, _offset, _exist] = compute_normal_and_offset(vertices);
					normal = _normal;			
					offset = _offset;					
					if(!_exist) std::cout << " error when filling this Face " << std::endl;
					/**On commence par ajouter les coordonnées du vectuer normal pour la face correspondante. */
					m_meshes[nb_meshes - 1].push_back(normal.x);
					m_meshes[nb_meshes - 1].push_back(normal.y);
					m_meshes[nb_meshes - 1].push_back(normal.z);
					/**Ensuite on ajoute le offset pour la face correspondante. */
					m_meshes[nb_meshes - 1].push_back(offset);
					/**Pour finir on ajoute les vertices pour la face correspondante. */
					for(auto v: vertices)
					{
						/**Maj du nombre de vertices pour la face correspondante. */
						m_meshes[nb_meshes - 1][nb_vertex_pos]++;
						/**On ajoute les cordonnées du vertex. */
						m_meshes[nb_meshes - 1].push_back(v.x);
						m_meshes[nb_meshes - 1].push_back(v.y);
						m_meshes[nb_meshes - 1].push_back(v.z);
					}
					vertices.clear();
					/**On met à jour le nombre de faces pour le mesh correspondant. */
					m_meshes[nb_meshes - 1][0]++;
					nf++;
					/**On prépare le terrain pour la prochaine face. */
					//m_meshes[nb_meshes - 1].push_back(0);
					//nb_vertex_pos = m_meshes[nb_meshes - 1].size() -1;
				}
			}
			printf("SIZE: %d\n", m_meshes[nb_meshes-1].size());
			std::cout << "Mesh: " << file_name << " - number of vertices: " << nv << " - number of faces: " << nf << std::endl;
		}
		
		/**
		 * @brief Computes the normal vector and offset for a face defined by its vertices.
		 *
		 * This function calculates the normal vector and offset for a face based on its vertices. It returns the computed
		 * normal vector, offset, and a boolean indicating success or failure.
		 *
		 * @return A tuple containing three values:
		 *         - `Vec3d` normal: The computed normal vector.
		 *         - `double` offset: The computed offset.
		 *         - `bool` success: Indicates whether the calculation was successful (true) or not (false).
		 */
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
		 * @brief Builds bounding boxes for all faces in the meshes.
		 *
		 * The `build_boxes` function calculates and stores bounding boxes for all faces in the meshes. It ensures that the mesh
		 * has at least one face and performs this computation in parallel when possible.
		 */
		void build2_boxes(int mesh)
		{
			//printf("BUILD\n");
			//const int size = m_data.size();
			const int size = m_meshes[/**nb_meshes*/mesh][0];
			//printf("SIZE: %d\n", size);
			//const int size = m_data_size;
			assert(size > 0);
			m_boxes[/**nb_meshes*/mesh].resize(size);
			int j = 1;
			//std::vector<Vec3d> vertices;
//#pragma omp parallel for private(vertices)
			//printf("BUILD2\n");
			for(int i = 0 ; i < size ; i++)
			{
				//printf("IIIIIIIII: %d\n", i);
				std::vector<Vec3d> vertices;
				int nb_Vertices = m_meshes[/**nb_meshes*/mesh][j]; 
				//printf("NB_VERTICES: %d\n", nb_Vertices);
				j+=5;
				int j_2 = j;
				/**On récupère tous les vertices de la face dans un vector. */
				for(int z = j_2; z < j_2 + nb_Vertices*3; z+=3){
				//	printf("JJJJJJJJ: %d\n", j);
					Vec3d v = {m_meshes[/**nb_meshes*/mesh][z], m_meshes[/**nb_meshes*/mesh][z + 1], m_meshes[/**nb_meshes*/mesh][z + 2]};
					vertices.push_back(v);
					j+=3; 
				}
				m_boxes[/**nb_meshes*/mesh][i] = create_box(vertices);
				//vertices.clear();
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
		void update_indexes(int mesh, const int id, Box& b1)
		{
			for( size_t idBox = 0 ; idBox < m_boxes[mesh].size() ; idBox++ )
			{
				auto& b2 = m_boxes[mesh][idBox];
				if (( b1.sup.x >= b2.inf.x && b1.inf.x <= b2.sup.x) &&
						(b1.sup.y >= b2.inf.y && b1.inf.y <= b2.sup.y) &&
						(b1.sup.z >= b2.inf.z && b1.inf.z <= b2.sup.z))
				{
					indexes[mesh][id].push_back(idBox);
				}
			}
		}
		
		
		
	};	
	



	/**
	 * @brief Represents a 3D mesh consisting of faces.
	 *
	 * The `stl_mesh` struct represents a 3D mesh composed of faces. It provides methods for adding faces to the mesh,
	 * accessing the mesh data, and retrieving individual faces.
	 	 
	 */
	struct stl_mesh
	{
		//std::vector <Face> m_data; /**< A collection of Face objects representing the mesh. */
		onika::memory::CudaMMVector< Face > m_data;
		//Face* m_data;
		//long unsigned int m_data_size = 0;
		//std::vector <Box> m_boxes;  /**< A collection of Box objects bounding the mesh. */
		onika::memory::CudaMMVector< Box > m_boxes;
		onika::memory::CudaMMVector< OBB > m_obbs;
		//std::vector <std::vector<int>> indexes; /**< Indexes for mesh data. */
		//onika::memory::CudaMMVector <std::vector<int>> indexes;
		onika::memory::CudaMMVector <onika::memory::CudaMMVector <int> > indexes;
		// indexes;
		//long unsigned int indexes_size = 0;

		/**
		 * @brief Adds a Face to the mesh.
		 * @param face The Face to be added to the mesh.
		 */
		void add_face(Face& face)
		{
			m_data.push_back(face);
			//m_data[m_data_size] = face;
			//m_data_size++;
		}

		/**
		 * @brief Gets a reference to the mesh data.
		 * @return A reference to the vector of Face objects representing the mesh.
		 */
		 //std::vector <Face> & get_data()
		onika::memory::CudaMMVector <Face>& get_data()
		{
			return m_data;
		}
		
		/**Face* get_data()
		{
			return m_data;
		}*/

		/**
		 * @brief Gets a reference to a specific Face in the mesh.
		 * @param idx The index of the Face to retrieve.
		 * @return A reference to the specified Face.
		 */
		Face& get_data(const int idx)
		{
			return m_data[idx];
		}
		
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
			//onika::memory::CudaMMVector <Vec3d> vertices;
			Vec3d vertex;
			int nv = 0;
			int nf = 0;
			
			for( std::string line; getline( input, line ); )
			{
				input >> first;
				if(first == "outer")
				{
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
					Face tmp (vertices);
					this->add_face(tmp);
					vertices.clear();
					nf++;
				}
			}
			std::cout << "Mesh: " << file_name << " - number of vertices: " << nv << " - number of faces: " << nf << std::endl;
		}

		/**
		 * @brief Builds bounding boxes for all faces in the mesh.
		 *
		 * The `build_boxes` function calculates and stores bounding boxes for all faces in the mesh. It ensures that the mesh
		 * has at least one face and performs this computation in parallel when possible.
		 */
		void build_boxes()
		{
			const int size = m_data.size();
			//const int size = m_data_size;
			assert(size > 0);
			m_boxes.resize(size);
#pragma omp parallel for
			for(int i = 0 ; i < size ; i++)
			{
				m_boxes[i] = m_data[i].create_box();
			}
		}
		
		void build_obbs()
		{
			const int size = m_data.size();
			assert(size > 0);
			m_obbs.resize(size);
#pragma omp parallel for
			for(int i = 0; i < size; i++)
			{
				m_obbs[i] = m_data[i].create_OBB();
			}
		}
		
		/**ONIKA_DEVICE_KERNEL_FUNC
		void build_boxes_kernel()
		{
			const int size = m_data.size();
			assert(size > 0);
			m_boxes.resize(size);
			long nt = ONIKA_CU_GRID_SIZE*ONIKA_CU_BLOCK_SIZE;
			long i = ONIKA_CU_BLOCK_IDX*ONIKA_CU_BLOCK_SIZE + ONIKA_CU_THREAD_IDX;
			for( ; i < size ; i += nt )
			{M
				m_boxes[i] = m_data[i].create_box();
			}
		}*/

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
			//printf("ICIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n");
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
		
		bool intersect_OBB( const OBB &b1, const OBB &b2 )
		{
			double ra, rb;
			//Mat3d R, AbsR;
			
			Mat3d R = make_mat3d({0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.});
			Mat3d AbsR = make_mat3d({0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.});
			
			//Compute rotation matrix expressing b2 in b1's coordinate frame
			for( int i = 0; i < 3; i++){
				for(int j = 0; j < 3; j++){
					set_Mat3d(R, i, j, exanb::dot(b1.axis[i], b2.axis[j]));
				}
			}
			
			//Compute translation vector t
			Vec3d t = b2.center - b1.center;
			
			//Bring translation vector into a's coordinate form
			t = {exanb::dot(t, b1.axis[0]), exanb::dot(t, b1.axis[1]), exanb::dot(t, b1.axis[2])};
			
			//Compute common subexpressions. Add in an epsilon term to counteract arithmetic errors when two edges are parallel and thei cross product is near null
			for( int i = 0; i < 3; i++){
				for( int j = 0; j < 3; j++){
					double set = std::abs(get_Mat3d(R, i, j)) + FLT_EPSILON;
					set_Mat3d(AbsR, i, j, set);
				}
			}
			
			
			//Test axes L = A0, L = A1, L = A2
			ra = b1.extents.x;
			rb = b2.extents.x * get_Mat3d(AbsR, 0, 0) + b2.extents.y * get_Mat3d(AbsR, 0, 1) + b2.extents.z * get_Mat3d(AbsR, 0, 2);
			if(std::abs(t.x) > ra + rb) return false;
			
			ra = b1.extents.y;
			rb = b2.extents.x * get_Mat3d(AbsR, 1, 0) + b2.extents.y * get_Mat3d(AbsR, 1, 1) + b2.extents.z * get_Mat3d(AbsR, 1, 2);
			if(std::abs(t.x) > ra + rb) return false;
			
			ra = b1.extents.z;
			rb = b2.extents.x * get_Mat3d(AbsR, 2, 0) + b2.extents.y * get_Mat3d(AbsR, 2, 1) + b2.extents.z * get_Mat3d(AbsR, 2, 2);
			if(std::abs(t.x) > ra + rb) return false;
			
			//Test axes L = B0, L = B1, L = B2
			ra = b1.extents.x * get_Mat3d(AbsR, 0, 0) + b1.extents.y * get_Mat3d(AbsR, 1, 0) + b1.extents.z * get_Mat3d(AbsR, 2, 0);
			rb = b2.extents.x;
			if(std::abs(t.x * get_Mat3d(R, 0, 0) + t.y * get_Mat3d(R, 1, 0) + t.z * get_Mat3d(R, 2, 0)) > ra + rb) return false;
			
			ra = b1.extents.x * get_Mat3d(AbsR, 0, 1) + b1.extents.y * get_Mat3d(AbsR, 1, 1) + b1.extents.z * get_Mat3d(AbsR, 2, 1);
			rb = b2.extents.y;
			if(std::abs(t.x * get_Mat3d(R, 0, 1) + t.y * get_Mat3d(R, 1, 1) + t.z * get_Mat3d(R, 2, 1)) > ra + rb) return false;
			
			ra = b1.extents.x * get_Mat3d(AbsR, 0, 2) + b1.extents.y * get_Mat3d(AbsR, 1, 2) + b1.extents.z * get_Mat3d(AbsR, 2, 2);
			rb = b2.extents.z;
			if(std::abs(t.x * get_Mat3d(R, 0, 2) + t.y * get_Mat3d(R, 1, 2) + t.z * get_Mat3d(R, 2, 2)) > ra + rb) return false;
			
			//Test axis L = A0 * B0
			ra = b1.extents.y * get_Mat3d(AbsR, 2, 0) + b1.extents.z * get_Mat3d(AbsR, 1, 0);
			rb = b2.extents.y * get_Mat3d(AbsR, 0, 2) + b2.extents.z * get_Mat3d(AbsR, 0, 1);
			if( std::abs(t.z * get_Mat3d(R, 1, 0) - t.y * get_Mat3d(R, 2, 0)) > ra + rb) return false;
			
			//Test axis L = A0 * B1
			ra = b1.extents.y * get_Mat3d(AbsR, 2, 1) + b1.extents.z * get_Mat3d(AbsR, 1, 1);
			rb = b2.extents.x * get_Mat3d(AbsR, 0, 2) + b2.extents.z * get_Mat3d(AbsR, 0, 0);
			if( std::abs(t.z * get_Mat3d(R, 1, 1) - t.y * get_Mat3d(R, 2, 1)) > ra + rb) return false;
			
			//Test axis L = A0 * B2
			ra = b1.extents.y * get_Mat3d(AbsR, 2, 2) + b1.extents.z * get_Mat3d(AbsR, 1, 2);
			rb = b2.extents.x * get_Mat3d(AbsR, 0, 1) + b2.extents.y * get_Mat3d(AbsR, 0, 0);
			if( std::abs(t.z * get_Mat3d(R, 1, 2) - t.y * get_Mat3d(R, 2, 2)) > ra + rb) return false;
			
			//Test axis L = A1 * B0
			ra = b1.extents.x * get_Mat3d(AbsR, 2, 0) + b1.extents.z * get_Mat3d(AbsR, 0, 0);
			rb = b2.extents.y * get_Mat3d(AbsR, 1, 2) + b2.extents.z * get_Mat3d(AbsR, 1, 1);
			if( std::abs(t.x * get_Mat3d(R, 2, 0) - t.z * get_Mat3d(R, 0, 0)) > ra + rb) return false;
			
			//Test axis L = A1 * B1
			ra = b1.extents.x * get_Mat3d(AbsR, 2, 1) + b1.extents.z * get_Mat3d(AbsR, 0, 1);
			rb = b2.extents.x * get_Mat3d(AbsR, 1, 2) + b2.extents.z * get_Mat3d(AbsR, 1, 0);
			if( std::abs(t.x * get_Mat3d(R, 2, 1) - t.z * get_Mat3d(R, 0, 1)) > ra + rb) return false;
			
			//Test axis L = A1 * B2
			ra = b1.extents.x * get_Mat3d(AbsR, 2, 2) + b1.extents.z * get_Mat3d(AbsR, 0, 2);
			rb = b2.extents.x * get_Mat3d(AbsR, 1, 1) + b2.extents.y * get_Mat3d(AbsR, 1, 0);
			if( std::abs(t.x * get_Mat3d(R, 2, 2) - t.z * get_Mat3d(R, 0, 2)) > ra + rb) return false;
			
			//Test axis L = A2 * B0
			ra = b1.extents.x * get_Mat3d(AbsR, 1, 0) + b1.extents.y * get_Mat3d(AbsR, 0, 0);
			rb = b2.extents.y * get_Mat3d(AbsR, 2, 2) + b2.extents.z * get_Mat3d(AbsR, 2, 1);
			if( std::abs(t.y * get_Mat3d(R, 0, 0) - t.x * get_Mat3d(R, 1, 0)) > ra + rb) return false;
			
			//Test axis L = A2 * B1
			ra = b1.extents.x * get_Mat3d(AbsR, 1, 1) + b1.extents.y * get_Mat3d(AbsR, 0, 1);
			rb = b2.extents.x * get_Mat3d(AbsR, 2, 2) + b2.extents.z * get_Mat3d(AbsR, 2, 2);
			if( std::abs(t.y * get_Mat3d(R, 0, 1) - t.x * get_Mat3d(R, 1, 1)) > ra + rb) return false;
			
			//Test axis L = A2 * B2
			ra = b1.extents.x * get_Mat3d(AbsR, 1, 2) + b1.extents.y * get_Mat3d(AbsR, 0, 2);
			rb = b2.extents.x * get_Mat3d(AbsR, 2, 1) + b2.extents.y * get_Mat3d(AbsR, 2, 0);
			if( std::abs(t.y * get_Mat3d(R, 0, 2) - t.x * get_Mat3d(R, 1, 2)) > ra + rb) return false;
			
			return true;
					
		}
		
		void update_indexes_obb(const int id, Box& b1)
		{
		
			//Convert AABB into OBB
			Vec3d extent = b1.sup - b1.inf;
			Vec3d center = b1.center();
			OBB test = {center, {{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}}, extent}; 
			for( size_t idBox = 0 ; idBox < m_obbs.size() ; idBox++ )
			{
				auto& b2 = m_obbs[idBox];
				if(intersect_OBB( b2, test )) indexes[id].push_back(idBox);
			}
		}
	};
}
