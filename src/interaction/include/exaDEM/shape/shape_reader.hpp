/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
#pragma once
#include <exaDEM/shape/shapes.hpp>
#include <cassert>
#include <fstream>
#include <regex>

namespace exaDEM
{
  // this function 
  inline void write_shp(const shape& shp, std::stringstream& output)
	{
    int nv = shp.get_number_of_vertices();
    int ne = shp.get_number_of_edges();
    int nf = shp.get_number_of_faces();

    output << std::endl << "<" << std::endl;
    output << "name " << shp.m_name.c_str() << std::endl;
    output << "radius " << shp.m_radius  << std::endl;
		output << "nv " << nv << std::endl;
    for(int i = 0; i < nv ; i++)
    {
      auto& v = shp.m_vertices[i] ;
      output << v.x << " " << v.y << " " << v.z << std::endl;
    }
		output << "ne " << ne << std::endl;
    for(int i = 0; i < ne ; i++) 
    {
      output << shp.m_edges[2*i] << " " << shp.m_edges[2*i+1] << std::endl;
    }
    output << "nf " << nf << std::endl;
    for(int i = 0; i < nf ; i++) 
    {
      auto [ptr, size] = shp.get_face(i);
      output << size;
      for(int j = 0 ; j < size ; j++)
      {
        output << " " << ptr[j];
      }
      output << std::endl;
    }
    output << "obb.extent " << shp.obb.extent << std::endl;
    output << "obb.e1 " << shp.obb.e1 << std::endl;
    output << "obb.e2 " << shp.obb.e2 << std::endl;
    output << "obb.e3 " << shp.obb.e3 << std::endl;
    output << "obb.center " << shp.obb.center << std::endl;
    output << "orientation 1.0 0.0 0.0 0.0" << std::endl;
    output << "volume " << shp.m_volume << std::endl;
    auto& Im = shp.m_inertia_on_mass;
    output << "I/m " << Im.x << Im.y << Im.z << std::endl;
    output << ">" << std::endl;
  }


	inline void read_shp(shapes& sphs, const std::string file_name)
	{
		std::ifstream input( file_name.c_str() );
		std::string first;
		std::string key;
		std::vector<int> tmp_face;
    //int count = 0;
		for( std::string line; getline( input, line ); )
		{
      //std::cout << count++ << std::endl;
			input >> first;
			if(first == "<")
			{
				OBB obb;

				first = ""; 
				shape new_shape;
				bool do_fill = true;
				while(do_fill)
				{
					input >> key;
          //std::cout << "size: " << key.size() << " " << key << std::endl;

					if(key == "name")
					{
						input >> new_shape.m_name;
					}
					else if(key == "obb.center") // === keys relative to the OBB
					{
						input >> obb.center.x >> obb.center.y >> obb.center.z;
					}
					else if(key == "obb.extent")
					{
						input >> obb.extent.x >> obb.extent.y >> obb.extent.z;
					}
					else if(key == "obb.e1")
					{
						input >> obb.e1.x >> obb.e1.y >> obb.e1.z;
					}
					else if(key == "obb.e2")
					{
						input >> obb.e2.x >> obb.e2.y >> obb.e2.z;
					}
					else if(key == "obb.e3")
					{
						input >> obb.e3.x >> obb.e3.y >> obb.e3.z;
					}
					else if(key == "radius")
					{
						input >> new_shape.m_radius;
					}
					else if(key == "volume")
					{
						input >> new_shape.m_volume;
					}
					else if(key == "I/m")
					{
						exanb::Vec3d Im;
						input >> Im.x >> Im.y >> Im.z; 
						new_shape.m_inertia_on_mass = Im;
					}
					else if(key == "nv")
					{
						int nv = 0;
						input >> nv;
						if( nv > EXADEM_MAX_VERTICES ) 
						{
							lout << "=== EXADEM ERROR ===" << std::endl;
							lout << "=== Please, increase the maximum number of vertices: cmake ${Path_To_ExaDEM} -DEXADEM_MAX_VERTICES=" << nv << std::endl;
							lout << "=== ABORT ===" << std::endl;     
							std::abort();
						}
						assert( nv != 0 );
						for(int i = 0; i < nv ; i++)
						{
							getline(input, line);
							exanb::Vec3d vertex;
							input >> vertex.x >> vertex.y >> vertex.z;
							new_shape.add_vertex(vertex);
						}
					}
					else if(key == "ne")
					{
						int ne = 0;
						input >> ne;
						//assert(ne!=0);
						for(int i = 0; i < ne ; i++)
						{
							getline(input, line);
							int e1, e2;
							input >> e1 >> e2;
							new_shape.add_edge(e1,e2);
						}
					}
					else if(key == "nf")
					{
						int nf = 0;
						input >> nf;
						//assert(nf!=0);
						for(int i = 0; i < nf ; i++)
						{
							getline(input, line);
							int n = 0;
							input >> n;
							assert(n != 0);
							tmp_face.resize(n);
							for(int j = 0; j < n ; j++)
							{
								input>>tmp_face[j];
							}
							new_shape.add_face(tmp_face.size(), tmp_face.data());
						}
					}
					else if(key == ">")
					{
						new_shape.obb = obb;
						do_fill = false;
					}
				}
				if(do_fill == false)
				{
					new_shape.print();
					new_shape.write_paraview();
					new_shape.pre_compute_obb_edges(Vec3d{0,0,0}, Quaternion{1,0,0,0});
					new_shape.pre_compute_obb_faces(Vec3d{0,0,0}, Quaternion{1,0,0,0});
					sphs.add_shape(&new_shape);
					do_fill = true;
				}
			}
		}
	}
}
