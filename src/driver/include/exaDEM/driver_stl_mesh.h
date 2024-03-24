#pragma once
#include <exanb/core/basic_types.h>
#include <exaDEM/driver_base.h>
#include <exaDEM/shape/shape.hpp>

namespace exaDEM
{
	using namespace exanb;
	
	template<typename T> using vector_t = std::vector<T>;

	struct list_of_elements
	{
		vector_t<int> vertices;
		vector_t<int> edges;
		vector_t<int> faces;
	};

  struct Stl_mesh
	{
		exanb::Vec3d center; // normal * offset
		exanb::Vec3d vel;   // 0,0,0
		exanb::Vec3d vrot;
    shape shp;
		vector_t<list_of_elements> grid_indexes;


		constexpr DRIVER_TYPE get_type() {return DRIVER_TYPE::STL_MESH;}

		void print()
		{
			std::cout << "Driver Type: MESH STL" << std::endl;
			std::cout << "center : " << center   << std::endl;
			std::cout << "Vel    : " << vel << std::endl;
			std::cout << "AngVel : " << vrot << std::endl;
			std::cout << "Number of faces    : " << shp.get_number_of_faces() << std::endl;
			std::cout << "Number of edges    : " << shp.get_number_of_edges() << std::endl;
			std::cout << "Number of vertices : " << shp.get_number_of_vertices() << std::endl;
		}

		inline void initialize ()
		{
			// checks
		}

		inline void update_position ( const double t )
		{
			center = center + t * vel; 
		}

		inline void grid_indexes_summary()
		{
			const size_t size = grid_indexes.size();
			size_t nb_fill_cells(0), nb_v (0), nb_e(0), nb_f(0), max_v(0), max_e(0), max_f(0);

#pragma omp parallel for reduction(+: nb_fill_cells, nb_v, nb_e, nb_f) reduction(max: max_v, max_e, max_f)
			for(size_t i = 0 ; i < size ; i++)
			{
				auto& list = grid_indexes[i];
				if ( list.vertices.size() == 0 && list.edges.size() == 0 && list.faces.size()) continue; 
				nb_fill_cells++;
				nb_v += list.vertices.size();
				nb_e += list.edges.size();
				nb_f += list.faces.size();
				max_v = std::max( max_v, list.vertices.size() );
				max_e = std::max( max_e, list.edges.size() );
				max_f = std::max( max_f, list.faces.size() );
			}

			lout << "========= STL Grid summary ======"                            << std::endl;
      lout << "Number of emplty cells = " << nb_fill_cells << " / " << size  << std::endl;
			lout << "Vertices (Total/Max)   = " << nb_v          << " / " << max_v << std::endl;
			lout << "Edges    (Total/Max)   = " << nb_e          << " / " << max_e << std::endl;
			lout << "Faces    (Total/Max)   = " << nb_f          << " / " << max_f << std::endl;
			lout << "================================="                            << std::endl;
		}
	};
}
