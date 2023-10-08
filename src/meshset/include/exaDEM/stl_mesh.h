#pragma once

#include <exaDEM/face.h>
#include <sstream>
#include <exanb/core/basic_types.h>

namespace exaDEM
{
	using namespace exanb;
	struct stl_mesh
	{
		std::vector <Face> m_data;

		void add_face(Face& face)
		{
			m_data.push_back(face);
		}

		std::vector <Face>& get_data()
		{
			return m_data;
		}

		Face& get_data(const int idx)
		{
			return m_data[idx];
		}

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
	};
}
