#pragma once

#include <exaDEM/shape.hpp>
#include <iostream>

namespace exaDEM
{
	void build_buffer (const exanb::Vec3d& pos, const shape* shp, const exanb::Quaternion& orient,  size_t& count_point_size, size_t& count_polygon_size, size_t& count_polygon_table_size, std::stringstream& buff_position, std::stringstream& buff_faces)
	{
		auto writer_v = [] (const exanb::Vec3d& v, std::stringstream& out, const exanb::Vec3d& p, const exanb::Quaternion& q)
    {
      exanb::Vec3d new_pos = p + q * v;
      out << new_pos.x << " " << new_pos.y << " " << new_pos.z << std::endl;
    };
		shp->for_all_vertices(writer_v, buff_position, pos, orient);

		size_t n_faces = shp->get_number_of_faces();
		count_polygon_size += n_faces;
			
    int* ptr = shp->get_faces() + 1;
    for(size_t it = 0 ; it < n_faces ; it++)
    {
      count_polygon_table_size += ptr[0] + 1; // number of vertices + vertex idexes
      ptr += ptr[0] + 1; // -> next face 
    }
		auto writer_f = [] (const size_t size, const int * data, std::stringstream& out, int off)
    {
      out << size;
      for (size_t it = 0 ; it < size ; it++) out << " " << data[it] + off;
      out << std::endl;
    };
		shp->for_all_faces(writer_f, buff_faces, count_point_size);
		count_point_size += shp->get_number_of_vertices();

	}

	void build_vtk (std::string name, size_t count_point_size, size_t count_polygon_size, size_t count_polygon_table_size, std::stringstream& buff_vertices, std::stringstream& buff_faces )
	{
		name = name + ".vtk";
    std::ofstream outFile(name);
    if (!outFile) {
      std::cerr << "Erreur : impossible de créer le fichier de sortie !" << std::endl;
      return;
    }
    outFile << "# vtk DataFile Version 3.0" << std::endl;
    outFile << "Spheres" << std::endl;
    outFile << "ASCII" << std::endl;
    outFile << "DATASET POLYDATA"<<std::endl;
    outFile << "POINTS " << count_point_size << " float" << std::endl;
		outFile << buff_vertices.rdbuf() << std::endl;
		outFile << "POLYGONS "<< count_polygon_size << " " << count_polygon_table_size << std::endl;
		outFile << buff_faces.rdbuf() << std::endl;
	}
}
