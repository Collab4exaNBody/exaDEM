#pragma once

#include <exaDEM/face.h>
#include <sstream>
#include <exanb/core/basic_types.h>

namespace exaDEM
{
	using namespace exanb;

	/**
	 * @brief Represents a 3D mesh consisting of faces.
	 *
	 * The `stl_mesh` struct represents a 3D mesh composed of faces. It provides methods for adding faces to the mesh,
	 * accessing the mesh data, and retrieving individual faces.
	 */
	struct stl_mesh
	{
		std::vector <Face> m_data; /**< A collection of Face objects representing the mesh. */
		std::vector <Box> m_boxes;  /**< A collection of Box objects bounding the mesh. */
		std::vector <std::vector<int>> indexes; /**< Indexes for mesh data. */

		/**
		 * @brief Adds a Face to the mesh.
		 * @param face The Face to be added to the mesh.
		 */
		void add_face(Face& face)
		{
			m_data.push_back(face);
		}

		/**
		 * @brief Gets a reference to the mesh data.
		 * @return A reference to the vector of Face objects representing the mesh.
		 */
		std::vector <Face>& get_data()
		{
			return m_data;
		}

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
			assert(size > 0);
			m_boxes.resize(size);
#pragma omp parallel for
			for(int i = 0 ; i < size ; i++)
			{
				m_boxes[i] = m_data[i].create_box();
			}
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
	};
}
