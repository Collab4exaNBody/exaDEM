#pragma once
#include <exaDEM/shape/shapes.hpp>
#include <cassert>

namespace exaDEM
{
	void add_shape_from_file_shp(shapes& sphs, const std::string file_name)
	{
		std::ifstream input( file_name.c_str() );
		std::string first;
		std::string key;
		std::vector<int> tmp_face;
		for( std::string line; getline( input, line ); )
		{
			input >> first;
			if(first == "<")
			{
				OBB obb;

				first = ""; // reset first for next shape
				shape new_shape;
				bool do_fill = true;
				while(do_fill)
				{
					input >> key;
					if(key == "name")
					{
						input >> new_shape.m_name;
					}

					// === keys relative to the OBB
					if(key == "obb.center")
					{
						input >> obb.center.x >> obb.center.y >> obb.center.z;
					}

					if(key == "obb.extent")
					{
						input >> obb.extent.x >> obb.extent.y >> obb.extent.z;
					}

					if(key == "obb.e1")
					{
						input >> obb.e[0].x >> obb.e[0].y >> obb.e[0].z;
					}

					if(key == "obb.e2")
					{
						input >> obb.e[1].x >> obb.e[1].y >> obb.e[1].z;
					}

					if(key == "obb.e3")
					{
						input >> obb.e[2].x >> obb.e[2].y >> obb.e[2].z;
					}

					// 
					if(key == "radius")
					{
						input >> new_shape.m_radius;
					}

					if(key == "volume")
					{
						input >> new_shape.m_volume;
					}

					if(key == "I/m")
					{
						exanb::Vec3d Im;
						input >> Im.x >> Im.y >> Im.z;
						new_shape.m_inertia_on_mass = Im;
					}

					if(key == "nv")
					{
						int nv = 0;
						input >> nv;
						if( nv >= EXADEM_MAX_VERTICES ) 
						{
							lout << "=== EXADEM ERROR ===
							lout << "=== Please, increase the maximum number of vertices: cmake ${Path_To_ExaDEM} -DEXADEM_MAX_VERTICES=" << nv << std::endl;
							lout << "=== ABORT ===     
							std::abort();
						}
						assert( nv !=0 );
						for(int i = 0; i < nv ; i++)
						{
							getline(input, line);
							exanb::Vec3d vertex;
							input >> vertex.x >> vertex.y >> vertex.z;
							new_shape.add_vertex(vertex);
						}
						continue;
					}

					if(key == "ne")
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
						continue;
					}

					if(key == "nf")
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
						continue;
					}

					if(key == ">")
					{
						new_shape.obb = obb;
						do_fill = false;
					}
				}
				if(do_fill == false)
				{
					new_shape.print();
					new_shape.write_paraview();
					new_shape.pre_compute_obb_edges();
					new_shape.pre_compute_obb_faces();
					sphs.add_shape(&new_shape);
					do_fill = true;
				}
			}
		}
	}
}
