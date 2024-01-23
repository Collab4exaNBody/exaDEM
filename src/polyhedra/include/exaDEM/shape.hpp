#pragma once
#include <fstream>  
#include <vector>
#include <cassert>
#include <math.h>
#include <exanb/core/quaternion_operators.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_operators.h>


namespace exaDEM
{
	struct shape
	{
		shape() {
			m_faces.push_back(0); // init
		}

		void clear()
		{
			m_vertices.clear(); 
			m_faces.clear();
			m_faces.push_back(0);
			m_edges.clear();
			m_name = "undefined";
		}

		std::vector<exanb::Vec3d> m_vertices; ///<  
		exanb::Vec3d m_inertia_on_mass;
		std::vector<int> m_edges; ///<  
		std::vector<int> m_faces; ///<  
		double m_radius; ///< use for detection
		double m_volume; ///< use for detection
		std::string m_name = "undefined";

		inline
			const double get_volume() const
			{
				assert(m_volume != 0 && "wrong initialisation");
				return m_volume;
			}

		inline const exanb::Vec3d get_Im()
		{
			return m_inertia_on_mass;
		}

		inline const exanb::Vec3d get_Im() const
		{
			return m_inertia_on_mass;
		}

		inline const int get_number_of_vertices()
		{
			return m_vertices.size();
		}

		inline const int get_number_of_vertices() const
		{
			return m_vertices.size();
		}

		inline const int get_number_of_edges()
		{
			return m_edges.size() / 2;
		}

		inline const int get_number_of_edges() const
		{
			return m_edges.size() / 2;
		}

		inline const int get_number_of_faces()
		{
			return m_faces[0];
		}

		inline const int get_number_of_faces() const
		{
			return m_faces[0];
		}

		inline exanb::Vec3d& get_vertex(const int i)
		{
			return m_vertices[i];
		}

		inline const exanb::Vec3d& get_vertex(const int i) const
		{
			return m_vertices[i];
		}

		inline exanb::Vec3d get_vertex(const int i, const exanb::Vec3d& p, const exanb::Quaternion& orient)
		{
			return p + orient * m_vertices[i];
		}

		inline exanb::Vec3d get_vertex(const int i, const exanb::Vec3d& p, const exanb::Quaternion& orient) const
		{
			return p + orient * m_vertices[i];
		}

		inline const std::pair<int,int> get_edge(const int i)
		{
			return {m_edges[2*i], m_edges[2*i+1]};
		}

		inline const std::pair<int,int> get_edge(const int i) const
		{
			return {m_edges[2*i], m_edges[2*i+1]};
		}

		inline int* get_faces() const
		{
			return (int*)m_faces.data();
		}

		// returns the pointor on data and the number of vertex in the faces
		const std::pair<int*, int> get_face(const int i)
		{
			assert(i < m_faces[0]);
			int * ptr = this->get_faces() + 1;
			for(int it = i ; it > 0 ; it --)
			{
				ptr += ptr[0]+1;
			}
			return { ptr + 1, ptr[0] };
		}

		// returns the pointor on data and the number of vertex in the faces
		const std::pair<int*, int> get_face(const int i) const
		{
			assert(i < m_faces[0]);
			int * ptr = this->get_faces() + 1;
			for(int it = i ; it > 0 ; it --)
			{
				ptr += ptr[0]+1;
			}
			return { ptr + 1, ptr[0] };
		}

		void add_vertex(const exanb::Vec3d& vertex)
		{
			m_vertices.push_back(vertex);
		}

		void add_edge(const int i, const int j)
		{
			assert(i>=0 && "add negatif vertex");
			assert(j>=0 && "add negatif vertex");
			m_edges.push_back(i);
			m_edges.push_back(j);
		}

		void add_face(const size_t n, const int* data)
		{
			assert(n != 0);
			m_faces[0]++;
			const size_t old_size = m_faces.size();
			m_faces.resize(old_size + n + 1); // number of vertex + 1 storage to this number
			m_faces[old_size] = n;
			for(size_t it = 0 ; it < n ; it++)
			{
				m_faces[old_size + 1 + it] = data[it];
			} 
		}

		void add_radius(const double radius)
		{
			m_radius = radius;
		}

		double compute_max_rcut() const	
		{
			const size_t n = this->get_number_of_vertices();
			double rcut = 0;
			// try every combinaisons  --> old and not useful	

			const exanb::Vec3d origin = {0.0,0.0,0.0};		
			for(size_t it = 0 ; it < n ; it++ )
			{
				auto& vertex = this->get_vertex(it);
				const exanb::Vec3d dov = vertex - origin;
				const double d = exanb::norm(dov) + 2 * m_radius;
				std::cout <<"rcut = " << d << std::endl;
				rcut = std::max(rcut, d);
	
				/*for(size_t it2 = 0 ; it2 < n ; it2++ )
				{
					if(it1 == it2) continue;
					auto& vertex2 = this->get_vertex(it2);
				}*/		
			}

			assert(rcut != 0);
			return rcut;
		}

		template<typename Func, typename... Args>
			void for_all_vertices(Func& func, Args&&... args)
			{
				const size_t n = this->get_number_of_vertices();
				for(size_t it = 0 ; it < n ; it++ )
				{
					auto& vertex = this->get_vertex(it);
					func(vertex, std::forward<Args>(args)...);
				}
			}

		template<typename Func, typename... Args>
			void for_all_vertices(Func& func, Args&&... args) const
			{
				const size_t n = this->get_number_of_vertices();
				for(size_t it = 0 ; it < n ; it++ )
				{
					auto& vertex = this->get_vertex(it);
					func(vertex, std::forward<Args>(args)...);
				}
			}
		template<typename Func, typename... Args>
			void for_all_edges(Func& func, Args&&... args)
			{
				const size_t n = this->get_number_of_faces();
				for(size_t it = 0 ; it < n ; it++ )
				{
					auto [first, second] = this->get_edge(it);
					func(first, second, std::forward<Args>(args)...);
				}
			}

		template<typename Func, typename... Args>
			void for_all_faces(Func& func, Args&&... args)
			{
				const size_t n = this->get_number_of_faces();
				for(size_t it = 0 ; it < n ; it++ )
				{
					auto [data, size] = this->get_face(it);
					func(size, data, std::forward<Args>(args)...);
				}
			}

		template<typename Func, typename... Args>
			void for_all_faces(Func& func, Args&&... args) const
			{
				const size_t n = this->get_number_of_faces();
				for(size_t it = 0 ; it < n ; it++ )
				{
					auto [data, size] = this->get_face(it);
					func(size, data, std::forward<Args>(args)...);
				}
			}

		void print_vertices()
		{
			int idx = 0;
			auto printer = [&idx](exanb::Vec3d& vertex) {
				std::cout << "Vertex[" << idx++ << "]: [" << vertex.x << "," << vertex.y << "," << vertex.z << "]" << std::endl;
			};

			std::cout << "Number of vertices: " << this->get_number_of_vertices() << std::endl;
			for_all_vertices(printer);
		}

		void print_edges()
		{
			int idx = 0;
			auto printer = [&idx](int first, int second) {
				std::cout << "Edge["<<idx++ <<"]: ["<< first << "," << second << "]";
			};

			if(this->get_number_of_edges() == 0)
			{
				std::cout << "No edge" << std::endl;
			}
			else
			{
				std::cout << "Number of edges: " << this->get_number_of_edges() << std::endl;
				for_all_edges(printer);
			}
		}

		void print_faces()
		{
			int idx = 0;
			auto printer = [&idx](int vertices, int* data) {
				std::cout << "Number of vertices in face [" << idx++ << "]: " << vertices << std::endl;
				std::cout << "Vertices :";
				for(int it = 0; it < vertices - 1 ; it++)
				{
					std::cout << data[it] << ", ";
				}
				std::cout << data[vertices-1] << std::endl;
			};
			if(this->get_number_of_faces() == 0)
			{
				std::cout << "No face" << std::endl;
			}
			else
			{
				std::cout << "Number of faces: " << this->get_number_of_faces() << std::endl;
				for_all_faces(printer);
			}
		}

		// This function displays information about the shape
		void print()
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

		void write_paraview()
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
	};
};
