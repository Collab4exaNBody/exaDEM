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
#include <exanb/core/basic_types.h>
#include <exaDEM/driver_base.h>
#include <exaDEM/shape/shape.hpp>
#include <exaDEM/shape/shape_reader.hpp>
#include <exaDEM/interaction/interaction.hpp>
#include <filesystem>

namespace exaDEM
{
  using namespace exanb;

  template <typename T> using vector_t = std::vector<T>;

  /**
   * @brief Struct representing a list of elements( vertex, edge, or face).
   */
  struct list_of_elements
  {
    vector_t<int> vertices; /**< List of vertex indices. */
    vector_t<int> edges;    /**< List of edge indices. */
    vector_t<int> faces;    /**< List of face indices. */
  };

  /**
   * @brief Struct representing a STL mesh in the exaDEM simulation.
   */
  struct Stl_mesh
  {
    exanb::Vec3d center;    /**< Center position of the STL mesh. */
    exanb::Vec3d vel;       /**< Velocity of the STL mesh. */
    exanb::Vec3d vrot;      /**< Angular velocity of the STL mesh. */
    exanb::Quaternion quat; /**< Quaternion of the STL mesh. */
    shape shp;              /**< Shape of the STL mesh. */
    vector_t<Vec3d> vertices;      /**< Collection of vertices (computed from shp, quat and center). */
    vector_t<list_of_elements> grid_indexes; /**< Grid indices of the STL mesh. */

    /**
     * @brief Get the type of the driver (in this case, STL_MESH).
     * @return The type of the driver.
     */
    constexpr DRIVER_TYPE get_type() { return DRIVER_TYPE::STL_MESH; }

    /**
     * @brief Print information about the STL mesh.
     */
    void print()
    {
      lout << "Driver Type: MESH STL" << std::endl;
      lout << "Name   : " << shp.m_name << std::endl;
      lout << "Center : " << center << std::endl;
      lout << "Vel    : " << vel << std::endl;
      lout << "AngVel : " << vrot << std::endl;
      lout << "Quat   : " << quat.w << " " << quat.x << " " << quat.y << " " << quat.z << std::endl;
      lout << "Number of faces    : " << shp.get_number_of_faces() << std::endl;
      lout << "Number of edges    : " << shp.get_number_of_edges() << std::endl;
      lout << "Number of vertices : " << shp.get_number_of_vertices() << std::endl;
    }

    /**
     * @brief Print information about the STL mesh.
     */
    inline void initialize()
    {
      // checks
      if( shp.get_number_of_faces() == 0 
          && shp.get_number_of_edges() == 0 
          && shp.get_number_of_vertices() == 0)
      {
        lout << "Your shape is not correctly defined, no vertex, no edge, and no face" << std::endl;
        std::abort();
      }

      // resize and initialize vertices
      vertices.resize(shp.get_number_of_vertices());  
#pragma omp parallel for schedule(static)
      for(int i = 0; i < shp.get_number_of_vertices() ; i++)
      {
        this->update_vertex(i);
      }

      // remove relative paths
      std::filesystem::path full_name = this->shp.m_name;
      this->shp.m_name = full_name.filename();
    }

    ONIKA_HOST_DEVICE_FUNC 
      inline void update_vertex(int i)
      {
        vertices[i] = shp.get_vertex(i, this->center, this->quat);
      }

    /**
     * @brief return driver velocity
     */
    ONIKA_HOST_DEVICE_FUNC inline Vec3d &get_vel() { return vel; }

    /**
     * @brief return driver velocity
     */
    ONIKA_HOST_DEVICE_FUNC inline exanb::Quaternion &get_quat() { return quat; }

    ONIKA_HOST_DEVICE_FUNC inline bool stationary()
    {
      constexpr exanb::Vec3d null = {0,0,0};
      return (this->vel == null && this->vrot == null);
    }

    void dump_driver(int id, std::string path, std::stringstream &stream)
    {
      std::string filename = path + this->shp.m_name + ".shp";
      stream << "  - add_stl_mesh:" << std::endl;
      stream << "     id: " << id << std::endl;
      stream << "     center: [" << this->center << "]" << std::endl;
      stream << "     filename: " << filename << std::endl;
      stream << "     minskowski: " << this->shp.m_radius << std::endl;
      stream << "     velocity: [" << this->vel << "]" << std::endl;
      stream << "     angular_velocity: [" << this->vrot << "]" << std::endl;
      stream << "     orientation: [" << quat.w << "," << quat.x << "," << quat.y << "," << quat.z << "]" << std::endl;
      write_shp(this->shp, filename);
    }

    // angular velocity
    inline void push_av_to_quat(double dt)
    {
      using namespace exanb;
      // std::cout << dt << " " << vrot << std::endl;
      this->quat = this->quat + dot(this->quat, this->vrot) * dt;
      this->quat = normalize(this->quat);
      ldbg << "Quat[stl mesh]: " << this->quat.w << " " << this->quat.x << " " << this->quat.y << " " << this->quat.z << std::endl;
    }

    inline void push_v_to_r(double dt)
    {
      using namespace exanb;
      this->center += this->vel * dt;
      // this->center += this->vel * dt + this->acc * dt * dt * 0.5; // full
    }

    /**
     * @brief Prints a summary of grid indices for the STL mesh.
     * @details This function prints the number of elements in the grid indexes for vertices, edges, and faces.
     */
    inline void grid_indexes_summary()
    {
      const size_t size = grid_indexes.size();
      size_t nb_fill_cells(0), nb_v(0), nb_e(0), nb_f(0), max_v(0), max_e(0), max_f(0);

#     pragma omp parallel for reduction(+: nb_fill_cells, nb_v, nb_e, nb_f) reduction(max: max_v, max_e, max_f)
      for (size_t i = 0; i < size; i++)
      {
        auto &list = grid_indexes[i];
        if (list.vertices.size() == 0 && list.edges.size() == 0 && list.faces.size())
          continue;
        nb_fill_cells++;
        nb_v += list.vertices.size();
        nb_e += list.edges.size();
        nb_f += list.faces.size();
        max_v = std::max(max_v, list.vertices.size());
        max_e = std::max(max_e, list.edges.size());
        max_f = std::max(max_f, list.faces.size());
      }

      lout << "========= STL Grid summary ======" << std::endl;
      lout << "Number of emplty cells = " << nb_fill_cells << " / " << size << std::endl;
      lout << "Vertices (Total/Max)   = " << nb_v << " / " << max_v << std::endl;
      lout << "Edges    (Total/Max)   = " << nb_e << " / " << max_e << std::endl;
      lout << "Faces    (Total/Max)   = " << nb_f << " / " << max_f << std::endl;
      lout << "=================================" << std::endl;
    }

    // rVerletMax = rVerlet + sphere radius
    std::vector<exaDEM::Interaction> detection_sphere_driver(const size_t cell, const size_t p, const uint64_t id, const size_t drv_id, const double rx, const double ry, const double rz, const double rVerletMax)
    {
      std::vector<exaDEM::Interaction> res;
      exaDEM::Interaction item;
      item.cell_i = cell;
      item.p_i = p;
      item.id_i = id;
      item.id_j = drv_id;
      item.sub_i = 0; // not used

      auto &list = grid_indexes[cell];
      const size_t stl_nv = list.vertices.size();
      const size_t stl_ne = list.edges.size();
      const size_t stl_nf = list.faces.size();

      OBB *__restrict__ stl_obb_vertices = onika::cuda::vector_data(shp.m_obb_vertices);
      OBB *__restrict__ stl_obb_edges = onika::cuda::vector_data(shp.m_obb_edges);
      OBB *__restrict__ stl_obb_faces = onika::cuda::vector_data(shp.m_obb_faces);

      vec3r v = {rx, ry, rz};
      // vertex - vertex
      item.type = 7;
      for (size_t j = 0; j < stl_nv; j++)
      {
        size_t idx = list.vertices[j];
        OBB obb = stl_obb_vertices[idx];
        obb.enlarge(rVerletMax);
        if (obb.intersect(v))
        {
          item.sub_j = idx;
          res.push_back(item);
        }
      }
      // vertex - edge
      item.type = 8;
      for (size_t j = 0; j < stl_ne; j++)
      {
        size_t idx = list.edges[j];
        OBB obb = stl_obb_edges[idx];
        obb.enlarge(rVerletMax);
        if (obb.intersect(v))
        {
          item.sub_j = idx;
          res.push_back(item);
        }
      }
      // vertex - face
      item.type = 9;
      for (size_t j = 0; j < stl_nf; j++)
      {
        size_t idx = list.faces[j];
        OBB obb = stl_obb_faces[idx];
        obb.enlarge(rVerletMax);
        if (obb.intersect(v))
        {
          item.sub_j = idx;
          res.push_back(item);
        }
      }
      return res;
    }
  };
} // namespace exaDEM
