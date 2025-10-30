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
#include <exaDEM/shapes.hpp>
#include <exanb/core/particle_type_id.h>
#include <cassert>
#include <fstream>
#include <filesystem>
#include <regex>

namespace exaDEM
{
  /**
   * @brief Read a single shape from an input stream.
   *
   * @param input       Input file stream (should be open and valid).
   * @param big_shape   Optional flag to handle very large shapes (default: false).
   *
   * @return shape      Fully populated shape object.
   */
  inline shape read_shp(
      std::ifstream &input, 
      bool big_shape = false)
  {
    shape shp;
    std::string key, line;
    std::vector<int> face_indices_buffer;
    exanb::Vec3d position = {0, 0, 0};
    while (1)
    {
      input >> key;
      
      if (key == "name")
      {
        input >> shp.m_name;
      }
      else if (key == "obb.center") // === keys relative to the OBB
      {
        input >> shp.obb.center.x >> shp.obb.center.y >> shp.obb.center.z;
      }
      else if (key == "obb.extent")
      {
        input >> shp.obb.extent.x >> shp.obb.extent.y >> shp.obb.extent.z;
      }
      else if (key == "obb.e1")
      {
        input >> shp.obb.e1.x >> shp.obb.e1.y >> shp.obb.e1.z;
      }
      else if (key == "obb.e2")
      {
        input >> shp.obb.e2.x >> shp.obb.e2.y >> shp.obb.e2.z;
      }
      else if (key == "obb.e3")
      {
        input >> shp.obb.e3.x >> shp.obb.e3.y >> shp.obb.e3.z;
      }
      else if (key == "radius")
      {
        input >> shp.m_radius;
      }
      else if (key == "volume")
      {
        input >> shp.m_volume;
      }
      else if (key == "I/m")
      {
        exanb::Vec3d Im;
        input >> Im.x >> Im.y >> Im.z;
        shp.m_inertia_on_mass = Im;
      }
      else if(key == "position")
      {
        input >> position.x >> position.y >> position.z;
      }
      else if (key == "nv")
      {
        int num_vertices = 0;
        input >> num_vertices;
        assert(num_vertices != 0);
        for (int i = 0; i < num_vertices; i++)
        {
          getline(input, line);
          exanb::Vec3d vertex;
          input >> vertex.x  >> vertex.y >> vertex.z;
          vertex = {vertex.x, vertex.y, vertex.z};
          shp.add_vertex(vertex);
        }
      }
      else if (key == "ne")
      {
        int num_edges = 0;
        input >> num_edges;
        // assert(ne!=0);
        for (int i = 0; i < num_edges; i++)
        {
          getline(input, line);
          int e1, e2;
          input >> e1 >> e2;
          shp.add_edge(e1, e2);
        }
      }
      else if (key == "nf")
      {
        int num_faces = 0;
        input >> num_faces;
        for (int i = 0; i < num_faces; i++)
        {
          getline(input, line);
          int n = 0;
          input >> n;
          assert(n != 0);
          face_indices_buffer.resize(n);
          for (int j = 0; j < n; j++)
          {
            input >> face_indices_buffer[j];
          }
          shp.add_face(face_indices_buffer.size(), face_indices_buffer.data());
        }
        shp.compute_offset_faces();
      }
      else if (key == ">")
      {
//        shp.obb.center = {shp.obb.center.x - position.x, shp.obb.center.y - position.y, shp.obb.center.z - position.z};
        shp.obb.center = {0,0,0}; // {shp.obb.center.x - position.x, shp.obb.center.y - position.y, shp.obb.center.z - position.z};
 //       shp.shift_vertices(position);
        shp.pre_compute_obb_edges(Vec3d{0, 0, 0}, Quaternion{1, 0, 0, 0});
        shp.pre_compute_obb_faces(Vec3d{0, 0, 0}, Quaternion{1, 0, 0, 0});
        return shp;
      }
    }
  }

/**
 * @brief Reads multiple shapes from a file and stores them in a container.
 *
 * @param file_name  Path to the input file.
 * @param big_shape  Flag to indicate handling of large shapes (default: false).
 * @return A vector containing the shapes read from the file.
 */
  inline std::vector<shape> read_shps(
      const std::string file_name, 
      bool big_shape = false)
  {
    std::ifstream input(file_name.c_str());
    std::vector<shape> res;
    if (!input.is_open()) 
    {
      color_log::error("write_shapes", "Impossible to create the output file: " + file_name); 
    }

    for (std::string line; getline(input, line);)
    {
      if (line == "<")
      {
        shape shp = read_shp(input, big_shape);
        /* too much verbosity        
           if (!big_shape)
           shp.print();
         */
        shp.write_paraview();
        res.push_back(shp);
      }
    }
    return res;
  }

  /**
   * @brief Registers a collection of shapes into the particle type map and shape container.
   *
   * @param ptm   Reference to the particle type map.
   * @param shps  Reference to the shape container.
   * @param shp   Vector of shapes to register.
   */
  inline void register_shapes(ParticleTypeMap& ptm, shapes& shps, std::vector<shape>& shp)
  {
    for(auto& s: shp)
    {
      if( ptm.find(s.m_name) != ptm.end() )
      {
        s.m_name = s.m_name + "X";
        color_log::warning("read_shape", "[read_shape, WARNING] This polyhedron name is already taken, exaDEM has renamed it to: " + s.m_name);
      }
      ptm[s.m_name] = shps.size();
      shps.add_shape(&s);
    }
  }
  /**
   * @brief Read multiple shapes from a file.
   *
   * @param file_name  Path to the input shape file.
   * @param big_shape  Optional flag for large shapes (default: false).
   */
  inline shape read_shp(
      const std::string file_name, 
      bool big_shape = false)
  {
    std::ifstream input(file_name.c_str());
    for (std::string line; getline(input, line);)
    {
      if (line == "<")
      {
        return read_shp(input, big_shape);
      }
    }
    color_log::warning("read_shape", "No shape find into the file " + file_name + ".");
    color_log::warning("read_shape", "This file is ignored" + file_name + ".");
    return shape();
  }

  /**
   * @brief Read multiple shapes from a file and store them in a shapes container.
   *
   * @param shps       Container to store the parsed shapes.
   * @param file_name  Path to the input shape file.
   * @param big_shape  Optional flag for large shapes (default: false).
   */
  inline void read_shp(
      shapes &shps, 
      const std::string file_name, 
      bool big_shape = false)
  {
    std::ifstream input(file_name.c_str());
    for (std::string line; getline(input, line);)
    {
      if (line == "<")
      {
        shape shp = read_shp(input, big_shape);
        shps.add_shape(&shp);
        if (!big_shape)
          shp.print();
        shp.write_paraview();
      }
    }
  }

  /**
   * @brief Reads shape definitions from a "rockable"-style input file.
   *
   * @param file_name Path to the file (std::string_view for efficiency).
   * @return Map associating each shape name with its Vec3d position.
   */
  inline std::map<std::string, Vec3d> scan_shape_position(std::string_view file_name)
  {
    std::map<std::string, Vec3d> res;
    std::ifstream input(file_name.data());
    for (std::string line; getline(input, line);)
    {
      if (line == "<")
      {
        Vec3d position = {0,0,0}; // default value
        std::string type_name = "error";
        std::string key;
        // Read until closing '>'
        while(std::getline(input, line) &&  line != ">" )
        {
          std::istringstream iss(line);
          iss >> key;

          if( key == "name" ) 
          {
            iss >> type_name;
          }

          if( key == "position")
          {
            iss >> position.x >> position.y >> position.z;
            std::cout << position << std::endl;
          }
        }

        if( type_name == "error" ) 
        {
          color_log::error("read_shape::scan_shape_position", "A shape is defined without name");
        }
        else
        {
          res[type_name] = position;
        }
      }
    }
    return res;
  }
} // namespace exaDEM
