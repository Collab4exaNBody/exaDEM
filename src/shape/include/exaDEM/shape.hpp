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

#include <fstream>
#include <vector>
#include <cassert>
#include <cmath>
#include <onika/log.h>
#include <exaDEM/basic_types.hpp>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <onika/cuda/stl_adaptors.h>
#include <onika/math/basic_types.h>

//#include <exaDEM/shape_printer.hpp>

namespace exaDEM
{
  using namespace onika;
  struct shape
  {

    template <typename T> using VectorT = onika::memory::CudaMMVector<T>;

    shape()
    {
      m_faces.push_back(0); // init
    }

    void clear()
    {
      m_vertices.clear();
      m_faces.clear();
      m_faces.resize(1);
      m_faces[0] = 0;
      m_edges.clear();
      m_name = "undefined";
    }

    VectorT<exanb::Vec3d> m_vertices; ///<
    exanb::Vec3d m_inertia_on_mass;
    VectorT<OBB> m_obb_vertices; ///< only used for stl meshes
    VectorT<OBB> m_obb_edges;
    VectorT<OBB> m_obb_faces;
    OBB obb;
    VectorT<int> m_edges; ///<
    VectorT<int> m_faces; ///<
    VectorT<int> m_offset_faces; ///<
    double m_radius;      ///< use for detection
    double m_volume;      ///< use for detection
    std::string m_name = "undefined";
    inline void pre_compute_obb_edges(const exanb::Vec3d &particle_center, const exanb::Quaternion &particle_quat);
    inline void pre_compute_obb_faces(const exanb::Vec3d &particle_center, const exanb::Quaternion &particle_quat);
    inline void pre_compute_obb_vertices(const exanb::Vec3d * scratch);
    inline void pre_compute_obb_edges(const exanb::Vec3d * scratch);
    inline void pre_compute_obb_faces(const exanb::Vec3d * scratch);
    inline void increase_obb(const double value);
    void compute_prepro_obb(exanb::Vec3d * scratch, const exanb::Vec3d &particle_center, const exanb::Quaternion &particle_quat);

    ONIKA_HOST_DEVICE_FUNC
      inline double get_volume() const
      {
        assert(m_volume != 0 && "wrong initialisation");
        return m_volume;
      }

    ONIKA_HOST_DEVICE_FUNC
      inline const exanb::Vec3d &get_Im() { return m_inertia_on_mass; }

    ONIKA_HOST_DEVICE_FUNC
      inline const exanb::Vec3d &get_Im() const { return m_inertia_on_mass; }

    ONIKA_HOST_DEVICE_FUNC
      inline int get_number_of_vertices() { return onika::cuda::vector_size(m_vertices); }

    ONIKA_HOST_DEVICE_FUNC
      inline int get_number_of_vertices() const { return onika::cuda::vector_size(m_vertices); }

    ONIKA_HOST_DEVICE_FUNC
      inline int get_number_of_edges() { return onika::cuda::vector_size(m_edges) / 2; }

    ONIKA_HOST_DEVICE_FUNC
      inline int get_number_of_edges() const { return onika::cuda::vector_size(m_edges) / 2; }

    ONIKA_HOST_DEVICE_FUNC
      inline int get_number_of_faces()
      {
        const int *__restrict__ faces = onika::cuda::vector_data(m_faces);
        return faces[0];
      }

    ONIKA_HOST_DEVICE_FUNC
      inline int get_number_of_faces() const
      {
        const int *__restrict__ faces = onika::cuda::vector_data(m_faces);
        return faces[0];
      }

    ONIKA_HOST_DEVICE_FUNC
      inline exanb::Vec3d &get_vertex(const int i)
      {
        Vec3d *__restrict__ vertices = onika::cuda::vector_data(m_vertices);
        return vertices[i];
      }

    ONIKA_HOST_DEVICE_FUNC
      inline const exanb::Vec3d &get_vertex(const int i) const
      {
        const Vec3d *__restrict__ vertices = onika::cuda::vector_data(m_vertices);
        return vertices[i];
      }

    ONIKA_HOST_DEVICE_FUNC
      inline exanb::Vec3d get_vertex(const int i, const exanb::Vec3d &p, const exanb::Quaternion &orient) 
      {
        const Vec3d *__restrict__ vertices = onika::cuda::vector_data(m_vertices); 
        return p + orient * vertices[i]; 
      }

    ONIKA_HOST_DEVICE_FUNC
      inline exanb::Vec3d get_vertex(const int i, const exanb::Vec3d &p, const exanb::Quaternion &orient) const
      {
        const Vec3d *__restrict__ vertices = onika::cuda::vector_data(m_vertices);
        return p + orient * vertices[i];
      }

    ONIKA_HOST_DEVICE_FUNC
      inline std::pair<int, int> get_edge(const int i)
      {
        const int *__restrict__ edges = onika::cuda::vector_data(m_edges);
        return {edges[2 * i], edges[2 * i + 1]};
      }

    ONIKA_HOST_DEVICE_FUNC
      inline std::pair<int, int> get_edge(const int i) const
      {
        const int *__restrict__ edges = onika::cuda::vector_data(m_edges);
        return {edges[2 * i], edges[2 * i + 1]};
      }

    ONIKA_HOST_DEVICE_FUNC
      inline int *get_faces() const
      {
        const int *faces = onika::cuda::vector_data(m_faces);
        return (int *)faces; // just get a copy of the pointer
      }

    ONIKA_HOST_DEVICE_FUNC
      inline int *get_faces()
      {
        int *faces = onika::cuda::vector_data(m_faces);
        return faces;
      }

    inline void compute_offset_faces()
    {
      int n = this->get_number_of_faces();
      m_offset_faces.resize(n);
      int *start = this->get_faces() ;
#     pragma omp parallel for
      for(int i = 0 ; i < n ; i++)
      {
        int * ptr = start + 1; // the first element is the total number of face
        int acc = 1;
        for (int it = i; it > 0; it--)
        {
          acc += ptr[0] + 1; // ptr[0] contains the number of vertices of this face
          ptr += ptr[0] + 1;
        }
        m_offset_faces[i] = acc; 
      } 
    }


    // returns the pointor on data and the number of vertex in the faces
    ONIKA_HOST_DEVICE_FUNC
      const std::pair<int *, int> get_face(const int i)
      {
        auto * __restrict__ data =  onika::cuda::vector_data(m_offset_faces);
        int *ptr = this->get_faces();
        int index = data[i];
        return {ptr + index + 1, ptr[index]};
      }

    // returns the pointor on data and the number of vertex in the faces
    ONIKA_HOST_DEVICE_FUNC
      const std::pair<int *, int> get_face(const int i) const
      {
        auto * __restrict__ data =  onika::cuda::vector_data(m_offset_faces);
        int *ptr = this->get_faces();
        int index = data[i];
        return {ptr + index + 1, ptr[index]};
      }

    ONIKA_HOST_DEVICE_FUNC
      inline OBB get_obb_edge(const exanb::Vec3d &position, const size_t index, const exanb::Quaternion& orientation) const
      {
        OBB res = m_obb_edges[index];
        res.rotate(conv_to_quat(orientation));
        res.translate(conv_to_vec3r(position));
        return res;
      }

    ONIKA_HOST_DEVICE_FUNC
      inline OBB get_obb_face(const exanb::Vec3d &position, const size_t index, const exanb::Quaternion& orientation) const
      {
        OBB res = m_obb_faces[index];
        res.rotate(conv_to_quat(orientation));
        res.translate(conv_to_vec3r(position));
        return res;
      }

    void add_vertex(const exanb::Vec3d &vertex) { m_vertices.push_back(vertex); }

    void add_edge(const int i, const int j)
    {
      assert(i >= 0 && "add negatif vertex");
      assert(j >= 0 && "add negatif vertex");
      m_edges.push_back(i);
      m_edges.push_back(j);
    }

    void add_face(const size_t n, const int *data)
    {
      assert(n != 0);
      m_faces[0]++;
      const size_t old_size = m_faces.size();
      m_faces.resize(old_size + n + 1); // number of vertex + 1 storage to this number
      m_faces[old_size] = n;
      for (size_t it = 0; it < n; it++)
      {
        m_faces[old_size + 1 + it] = data[it];
      }
    }

    void add_radius(const double radius) { m_radius = radius; }

    double compute_max_rcut() const
    {
      const size_t n = this->get_number_of_vertices();
      double rcut = 0;
      for (size_t it = 0; it < n; it++)
      {
        auto &vertex = this->get_vertex(it);
        const double d = exanb::norm(vertex) + m_radius; // 2 * m_radius;
        rcut = std::max(rcut, d);
      }

      assert(rcut != 0);
      return rcut;
    }

    template <typename Func, typename... Args> void for_all_vertices(Func &func, Args &&...args)
    {
      const size_t n = this->get_number_of_vertices();
      for (size_t it = 0; it < n; it++)
      {
        auto &vertex = this->get_vertex(it);
        func(vertex, std::forward<Args>(args)...);
      }
    }

    template <typename Func, typename... Args> void for_all_vertices(Func &func, Args &&...args) const
    {
      const size_t n = this->get_number_of_vertices();
      for (size_t it = 0; it < n; it++)
      {
        auto &vertex = this->get_vertex(it);
        func(vertex, std::forward<Args>(args)...);
      }
    }

    template <typename Func, typename... Args> void for_all_edges(Func &func, Args &&...args)
    {
      const size_t n = this->get_number_of_edges();
      for (size_t it = 0; it < n; it++)
      {
        auto [first, second] = this->get_edge(it);
        func(first, second, std::forward<Args>(args)...);
      }
    }

    template <typename Func, typename... Args> void for_all_edges(Func &func, Args &&...args) const
    {
      const size_t n = this->get_number_of_edges();
      for (size_t it = 0; it < n; it++)
      {
        auto [first, second] = this->get_edge(it);
        func(first, second, std::forward<Args>(args)...);
      }
    }

    template <typename Func, typename... Args> void for_all_faces(Func &func, Args &&...args)
    {
      const size_t n = this->get_number_of_faces();
      for (size_t it = 0; it < n; it++)
      {
        auto [data, size] = this->get_face(it);
        func(size, data, std::forward<Args>(args)...);
      }
    }

    template <typename Func, typename... Args> void for_all_faces(Func &func, Args &&...args) const
    {
      const size_t n = this->get_number_of_faces();
      for (size_t it = 0; it < n; it++)
      {
        auto [data, size] = this->get_face(it);
        func(size, data, std::forward<Args>(args)...);
      }
    }

    void rescale(const double scale)
    {
      auto scale_vertices = [] (exanb::Vec3d& v, double s) { v = s * v; };
      for_all_vertices(scale_vertices, scale);
      m_radius *= scale;

    }

    void print_vertices()
    {
      int idx = 0;
      auto printer = [&idx](exanb::Vec3d &vertex) { lout << "Vertex[" << idx++ << "]: [" << vertex.x << "," << vertex.y << "," << vertex.z << "]" << std::endl; };

      lout << "Number of vertices= " << this->get_number_of_vertices() << std::endl;
      for_all_vertices(printer);
    }

    void print_edges()
    {
      int idx = 0;
      auto printer = [&idx](int first, int second) { lout << "Edge[" << idx++ << "]: [" << first << "," << second << "]" << std::endl; };

      if (this->get_number_of_edges() == 0)
      {
        lout << "No edge" << std::endl;
      }
      else
      {
        lout << "Number of edge  = " << this->get_number_of_edges() << std::endl;
        for_all_edges(printer);
      }
    }

    void print_faces()
    {
      int idx = 0;
      auto printer = [&idx](int vertices, int *data)
      {
        lout << "Face [" << idx++ << "]: ";
        for (int it = 0; it < vertices - 1; it++)
        {
          lout << data[it] << ", ";
        }
        lout << data[vertices - 1] << std::endl;
      };
      if (this->get_number_of_faces() == 0)
      {
        lout << "No face" << std::endl;
      }
      else
      {
        lout << "Number of faces  = " << this->get_number_of_faces() << std::endl;
        for_all_faces(printer);
      }
    }

    double compute_surface() const
    {
      double surface = 0.0;
      const size_t n_faces = this->get_number_of_faces();
#pragma omp parallel for reduction(+: surface)
      for (size_t it = 0; it < n_faces; it++)
      {
        auto [data, size] = this->get_face(it);
        const Vec3d& vi = m_vertices[data[0]];
        if( size == 3 )
        {
          for(int j = 1; j < size - 1; j++)
          {
            const size_t k = j + 1;
            const Vec3d vij = m_vertices[data[j]] - vi;
            const Vec3d vik = m_vertices[data[k]] - vi;
            const Vec3d cross = exanb::cross(vij, vik);
            surface += 0.5 * exanb::norm(cross);
          }
        }
      }
      return surface;
    }

    inline void print()
    {
      lout << std::endl;
      lout << "======= Shape Configuration =====" << std::endl;
      lout << "Shape Name        = " << this->m_name << std::endl;
      lout << "Shape Radius      = " << this->m_radius << std::endl;
      lout << "Shape I/m         = [" << this->m_inertia_on_mass << "]" << std::endl;
      lout << "Shape Volume      = " << this->m_volume << std::endl;
      print_vertices();
      print_edges();
      print_faces();
      lout << "=================================" << std::endl << std::endl;
    }

    inline void write_paraview()
    {
      ldbg << " writting paraview for shape " << this->m_name << std::endl;
      std::string name = m_name + ".vtk";
      std::ofstream outFile(name);
      if (!outFile)
      {
        std::cerr << "[ERROR] Impossible to create an output file!" << std::endl;
        std::cerr << "[ERROR] Impossible to open the file: " << name << std::endl;
        return;
      }
      outFile << "# vtk DataFile Version 3.0" << std::endl;
      outFile << "Spheres" << std::endl;
      outFile << "ASCII" << std::endl;
      outFile << "DATASET POLYDATA" << std::endl;
      outFile << "POINTS " << this->get_number_of_vertices() << " float" << std::endl;
      auto writer_v = [](exanb::Vec3d &v, std::ofstream &out) { out << v.x << " " << v.y << " " << v.z << std::endl; };

      for_all_vertices(writer_v, outFile);

      outFile << std::endl;

      outFile << "LINES " << this->get_number_of_edges() << " " << 3*this->get_number_of_edges() << std::endl;

      auto writer_e = [] (int a, int b, std::ofstream &out)
      {
        out << "2 " << a << " " << b << std::endl;
      };

      for_all_edges(writer_e, outFile);

      int count_polygon_size = this->get_number_of_faces();
      int count_polygon_table_size = 0;
      int *ptr = this->m_faces.data() + 1;
      for (int it = 0; it < count_polygon_size; it++)
      {
        count_polygon_table_size += ptr[0] + 1; // number of vertices + vertex idexes
        ptr += ptr[0] + 1;                      // -> next face
      }
      outFile << std::endl;

      outFile << "POLYGONS " << count_polygon_size << " " << count_polygon_table_size << std::endl;
      auto writer_f = [](const size_t size, const int *data, std::ofstream &out)
      {
        out << size;
        for (size_t it = 0; it < size; it++)
          out << " " << data[it];
        out << std::endl;
      };
      for_all_faces(writer_f, outFile);
    }

    inline void write_move_paraview(std::string path, int timestep, Vec3d &center, Quaternion &quat)
    {
      std::string time = std::to_string(timestep);
      ldbg << " writting paraview for shape " << this->m_name << " timestep: " << time << std::endl;
      std::string name = path + m_name + "_" + time + ".vtk";
      std::ofstream outFile(name);
      if (!outFile)
      {
        std::cerr << "[ERROR] Impossible to create the output file: " << name << std::endl;
        return;
      }
      outFile << "# vtk DataFile Version 3.0" << std::endl;
      outFile << "Spheres" << std::endl;
      outFile << "ASCII" << std::endl;
      outFile << "DATASET POLYDATA" << std::endl;
      outFile << "POINTS " << this->get_number_of_vertices() << " float" << std::endl;
      auto writer_v = [](const exanb::Vec3d &v, const exanb::Vec3d &center, const exanb::Quaternion &Q, std::ofstream &out)
      {
        exanb::Vec3d Vertex = center + Q * v;
        out << Vertex.x << " " << Vertex.y << " " << Vertex.z << std::endl;
      };

      for_all_vertices(writer_v, center, quat, outFile);

      outFile << std::endl;
      int count_polygon_size = this->get_number_of_faces();
      int count_polygon_table_size = 0;
      int *ptr = this->m_faces.data() + 1;
      for (int it = 0; it < count_polygon_size; it++)
      {
        count_polygon_table_size += ptr[0] + 1; // number of vertices + vertex idexes
        ptr += ptr[0] + 1;                      // -> next face
      }

      outFile << "POLYGONS " << count_polygon_size << " " << count_polygon_table_size << std::endl;
      auto writer_f = [](const size_t size, const int *data, std::ofstream &out)
      {
        out << size;
        for (size_t it = 0; it < size; it++)
          out << " " << data[it];
        out << std::endl;
      };
      for_all_faces(writer_f, outFile);
    }
  };

  inline int contact_possibilities(const shape *s1, const shape *s2)
  {
    const int nv1 = s1->get_number_of_vertices();
    const int ne1 = s1->get_number_of_edges();
    const int nf1 = s1->get_number_of_faces();
    const int nv2 = s2->get_number_of_vertices();
    const int ne2 = s2->get_number_of_edges();
    const int nf2 = s2->get_number_of_faces();
    return nv1 * (nv2 + ne2 + nf2) + ne1 * ne2 + nv2 * (ne1 + nf1);
  }

}; // namespace exaDEM

#include <exaDEM/shape_prepro.hpp>
