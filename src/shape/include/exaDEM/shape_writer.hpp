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
   * @brief Write the description of a shape into a stringstream.
   *
   * This function serializes the geometric properties of a shape 
   * (vertices, edges, faces, OBB, volume, inertia, etc.).
   *
   * @param shp     Input shape to serialize.
   * @param output  Output stringstream where the shape data will be written.
   */
  inline void write_shp(
      const shape &shp, 
      std::stringstream &output)
  {
    int n_vertices = shp.get_number_of_vertices();
    int n_edges = shp.get_number_of_edges();
    int n_faces = shp.get_number_of_faces();

    output << std::endl << "<" << std::endl;
    output << "name " << shp.m_name.c_str() << std::endl;
    output << "radius " << shp.m_radius << std::endl;
    output << "preCompDone y" << std::endl;
    output << "nv " << n_vertices << std::endl;
    for (int i = 0; i < n_vertices; i++)
    {
      auto &vertex = shp.get_vertex(i);
      output << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
    }
    output << "ne " << n_edges << std::endl;
    for (int i = 0; i < n_edges; i++)
    {
      output << shp.m_edges[2 * i] << " " << shp.m_edges[2 * i + 1] << std::endl;
    }
    output << "nf " << n_faces << std::endl;
    for (int i = 0; i < n_faces; i++)
    {
      auto [ptr, size] = shp.get_face(i);
      output << size;
      for (int j = 0; j < size; j++)
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
    auto &inertia_over_mass = shp.m_inertia_on_mass;
    output << "I/m " << inertia_over_mass.x << " " << inertia_over_mass.y << " " << inertia_over_mass.z << std::endl;
    output << ">" << std::endl;
  }

  /**
   * @brief Write a shape to a file.
   * This function serializes a shape (via write_shp(const shape&, std::stringstream&))
   *
   * @param shp      Input shape to serialize.
   * @param filename Path to the output file.
   */
  inline void write_shp(
      const shape &shp, 
      std::string filename)
  {
    std::stringstream stream;
    stream.precision(16);
    write_shp(shp, stream);
    std::ofstream file(filename.c_str());
    if (!file.is_open())
    {
      throw std::runtime_error("Failed to open file: " + filename);
    }
    file << stream.rdbuf();
  }

  /**
   * @brief Writes shape data from the given shapes container to a file.
   *
   * @param shps The container of shapes to write.
   * @param filename The output file path.
   * @param precision The numeric precision for floating-point output (default 16).
   */
  inline void write_shps(shapes& shps, std::string filename, int precision = 16)
  {
    std::stringstream stream;
    stream << std::setprecision(precision);
    // creating directory if it does not already exist
    const std::filesystem::path fspath(filename);
    std::filesystem::create_directories(fspath.parent_path());
    // open output file
    std::ofstream outFile(filename);
    if (!outFile)
    {
      std::cerr << "[write_shapes, ERROR] Impossible to create the output file: " << filename << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // fill stream with shape data
    for (size_t i = 0; i < shps.size(); i++)
    {
      const shape *shp = shps[i];
      exaDEM::write_shp(*shp, stream);
    }
    // fill output file
    outFile << std::setprecision(16);
    outFile << stream.rdbuf();
  }
} // namespace exaDEM
