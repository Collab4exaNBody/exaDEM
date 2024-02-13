#pragma once

#include <exaDEM/shape.hpp>
#include <iostream>

namespace exaDEM
{
/*
	// This function displays information about the shape
	inline void shape::print()
	{
		std::cout << "======= Shape Configuration =====" << std::endl;
		std::cout << "Shape Name: " << this->m_name << std::endl;
		std::cout << "Shape Radius: " << this->m_radius << std::endl;
		std::cout << "Shape I/m: [" << this->m_inertia_on_mass << "]" << std::endl;
		std::cout << "Shape Volume: " << this->m_volume << std::endl;
		print_vertices();
		print_edges();
		print_faces();
		std::cout << "=================================" << std::endl << std::endl;
	}

	inline void shape::write_paraview()
	{
		std::cout << " writting paraview for shape " << this->m_name << std::endl;
		std::string name = m_name + ".vtk";
		std::ofstream outFile(name);
		if (!outFile) {
			std::cerr << "Erreur : impossible de créer le fichier de sortie !" << std::endl;
			return;
		}
		outFile << "# vtk DataFile Version 3.0" << std::endl;
		outFile << "Spheres" << std::endl;
		outFile << "ASCII" << std::endl;
		outFile << "DATASET POLYDATA"<<std::endl;
		outFile << "POINTS " << this->get_number_of_vertices() << " float" << std::endl;
		auto writer_v = [] (exanb::Vec3d& v, std::ofstream& out) 
		{
			out << v.x << " " << v.y << " " << v.z << std::endl;
		};

		for_all_vertices(writer_v, outFile);

		outFile << std::endl;
		int count_polygon_size = this->get_number_of_faces();
		int count_polygon_table_size = 0;
		int* ptr = this->m_faces.data() + 1;
		for(int it = 0 ; it < count_polygon_size ; it++)
		{
			count_polygon_table_size += ptr[0] + 1; // number of vertices + vertex idexes
			ptr += ptr[0] + 1; // -> next face 
		}

		outFile << "POLYGONS "<<count_polygon_size<< " " << count_polygon_table_size << std::endl;
		auto writer_f = [] (const size_t size, const int * data,  std::ofstream& out)
		{
			out << size;
			for (size_t it = 0 ; it < size ; it++) out << " " << data[it];
			out << std::endl;
		};
		for_all_faces(writer_f, outFile);
	}
*/
	void build_buffer (const exanb::Vec3d& pos, const shape* shp, const exanb::Quaternion& orient,  size_t& polygon_offset_in_stream,
			size_t& n_vertices, 	size_t& n_polygon, 			std::stringstream& buff_position, std::stringstream& buff_faces, std::stringstream& buff_offset)
	{
		auto writer_v = [] (const exanb::Vec3d& v, std::stringstream& out, const exanb::Vec3d& p, const exanb::Quaternion& q)
		{
			exanb::Vec3d new_pos = p + q * v;
			out << " " << new_pos.x << " " << new_pos.y << " " << new_pos.z;
		};

		shp->for_all_vertices(writer_v, buff_position, pos, orient);

		size_t n_faces = shp->get_number_of_faces();
		n_polygon += n_faces;

		// faces 
		auto writer_f = [] (const size_t size, const int * data, std::stringstream& sface, std::stringstream& soffset, size_t& offset, size_t point_off)
		{
			soffset << offset + size << " ";
			offset += size;
			for (size_t it = 0 ; it < size ; it++) sface << " " << data[it] + point_off;
		};
		shp->for_all_faces(writer_f, buff_faces, buff_offset, polygon_offset_in_stream, n_vertices);
		n_vertices += shp->get_number_of_vertices();
	}

	void write_vtp (std::string name, size_t n_vertices, size_t n_polygons, 
			std::stringstream& buff_vertices, std::stringstream& buff_faces, std::stringstream& buff_offsets )
	{
		name = name + ".vtp";
		std::ofstream outFile(name);
		if (!outFile) {
			std::cerr << "Erreur : impossible de créer le fichier de sortie !" << std::endl;
			return;
		}

		outFile << "<VTKFile type=\"PolyData\">"  << std::endl;
		outFile << " <PolyData>"  << std::endl;
		outFile << "   <Piece NumberOfPoints=\"" << n_vertices << "\" NumberOfPolys=\"" << n_polygons << "\">"  << std::endl;
		outFile << "   <Points>"  << std::endl;
		outFile << "     <DataArray type=\"Float64\" Name=\"\"  NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
		outFile << buff_vertices.rdbuf() << std::endl;
		outFile << "     </DataArray>"  << std::endl;
		outFile << "   </Points>"  << std::endl;
		outFile << "   <Polys>"  << std::endl;
		outFile << "     <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">"  << std::endl;
		outFile << buff_faces.rdbuf() << std::endl;
		outFile << "     </DataArray>"  << std::endl;
		outFile << "     <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">"  << std::endl;
		outFile << buff_offsets.rdbuf() << std::endl;
		outFile << "     </DataArray>"  << std::endl;
		outFile << "   </Polys>"  << std::endl;
		outFile << "  </Piece>"  << std::endl;
		outFile << " </PolyData>"  << std::endl;
		outFile << "</VTKFile>"  << std::endl;
	}
	
	void write_pvtp
	{
		
	}
	/*
		 void build_buffer_old (const exanb::Vec3d& pos, const shape* shp, const exanb::Quaternion& orient,  size_t& count_point_size, 
		 size_t& count_line_size, size_t& count_line_table_size,
		 size_t& count_polygon_size, size_t& count_polygon_table_size,
		 std::stringstream& buff_position, std::stringstream& buff_edges, std::stringstream& buff_faces)
		 {
		 auto writer_v = [] (const exanb::Vec3d& v, std::stringstream& out, const exanb::Vec3d& p, const exanb::Quaternion& q)
		 {
		 exanb::Vec3d new_pos = p + q * v;
		 out << new_pos.x << " " << new_pos.y << " " << new_pos.z << std::endl;
		 };
		 shp->for_all_vertices(writer_v, buff_position, pos, orient);

		 size_t n_edges = shp->get_number_of_edges();
		 size_t n_faces = shp->get_number_of_faces();
		 count_polygon_size += n_faces;
		 count_line_size += n_edges;
		 count_line_table_size += 3 * n_edges;  // (number of vertices + vertex 1 + vertex 2) * number of edges 

		 int* ptr = shp->get_faces() + 1;
		 for(size_t it = 0 ; it < n_faces ; it++)
		 {
		 count_polygon_table_size += ptr[0] + 1; // number of vertices + vertex idexes
		 ptr += ptr[0] + 1; // -> next face 
		 }

	// edge
	auto writer_e = [] (const size_t v1, const size_t v2, std::stringstream& out, int off)
	{
	out << 2 << " " << off + v1 << " " << off + v2 << std::endl;
	};

	shp->for_all_edges(writer_e, buff_edges, count_point_size);

	// faces 
	auto writer_f = [] (const size_t size, const int * data, std::stringstream& out, int off)
	{
	out << size;
	for (size_t it = 0 ; it < size ; it++) out << " " << data[it] + off;
	out << std::endl;
	};
	shp->for_all_faces(writer_f, buff_faces, count_point_size);

	count_point_size += shp->get_number_of_vertices(); // at the end
	}

	void build_buffer (const exanb::Vec3d& pos, const shape* shp, const exanb::Quaternion& orient,  size_t& count_point_size, 
	size_t& count_line_size, size_t& count_line_table_size,
	size_t& count_polygon_size, size_t& count_polygon_table_size,
	std::stringstream& buff_position, std::stringstream& buff_edges, std::stringstream& buff_faces)
	{
	auto writer_v = [] (const exanb::Vec3d& v, std::stringstream& out, const exanb::Vec3d& p, const exanb::Quaternion& q)
	{
	exanb::Vec3d new_pos = p + q * v;
	out << new_pos.x << " " << new_pos.y << " " << new_pos.z << std::endl;
	};
	shp->for_all_vertices(writer_v, buff_position, pos, orient);

	size_t n_edges = shp->get_number_of_edges();
	size_t n_faces = shp->get_number_of_faces();
	count_polygon_size += n_faces;
	count_line_size += n_edges;
	count_line_table_size += 3 * n_edges;  // (number of vertices + vertex 1 + vertex 2) * number of edges 

	int* ptr = shp->get_faces() + 1;
	for(size_t it = 0 ; it < n_faces ; it++)
	{
	count_polygon_table_size += ptr[0] + 1; // number of vertices + vertex idexes
	ptr += ptr[0] + 1; // -> next face 
	}

	// edge
	auto writer_e = [] (const size_t v1, const size_t v2, std::stringstream& out, int off)
	{
		out << 2 << " " << off + v1 << " " << off + v2 << std::endl;
	};

	shp->for_all_edges(writer_e, buff_edges, count_point_size);

	// faces 
	auto writer_f = [] (const size_t size, const int * data, std::stringstream& out, int off)
	{
		out << size;
		for (size_t it = 0 ; it < size ; it++) out << " " << data[it] + off;
		out << std::endl;
	};
	shp->for_all_faces(writer_f, buff_faces, count_point_size);

	count_point_size += shp->get_number_of_vertices(); // at the end
}

void build_vtk_old (std::string name, size_t count_point_size, 
		size_t count_line_size, size_t count_line_table_size, 
		size_t count_polygon_size, size_t count_polygon_table_size, 
		std::stringstream& buff_vertices, std::stringstream& buff_edges, std::stringstream& buff_faces )
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

	// only if the number of face = 0
	if( count_polygon_size == 0 )
	{
		outFile << "LINES " << count_line_size  << " " << count_line_table_size << std::endl;
		outFile << buff_edges.rdbuf() << std::endl;
	}
	else
	{
		outFile << "POLYGONS "<< count_polygon_size << " " << count_polygon_table_size << std::endl;
		outFile << buff_faces.rdbuf() << std::endl;
	}
}
*/
}
