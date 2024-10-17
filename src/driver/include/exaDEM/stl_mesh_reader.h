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

#include <exaDEM/face.h>
#include <sstream>
#include <exanb/core/basic_types.h>

namespace exaDEM
{
  using namespace exanb;

  /**
   * @brief Represents a 3D mesh consisting of faces.
   *
   * The `stl_mesh` struct represents a 3D mesh composed of faces. It provides methods for adding faces to the mesh,
   * accessing the mesh data, and retrieving individual faces.
   */
  struct stl_mesh_reader
  {
    std::vector<Face> m_data; /**< A collection of Face objects representing the mesh. */

    /**
     * @brief Adds a Face to the mesh.
     * @param face The Face to be added to the mesh.
     */
    void add_face(Face &face) { m_data.push_back(face); }

    /**
     * @brief Gets a reference to the mesh data.
     * @return A reference to the vector of Face objects representing the mesh.
     */
    std::vector<Face> &get_data() { return m_data; }

    /**
     * @brief Gets a reference to a specific Face in the mesh.
     * @param idx The index of the Face to retrieve.
     * @return A reference to the specified Face.
     */
    Face &get_data(const int idx) { return m_data[idx]; }

    /**
     * @brief Reads mesh data from an STL file and populates the mesh.
     *
     * The `read_stl` function reads mesh data from an STL file specified by `file_name` and populates the mesh with
     * vertices and faces. It also calculates the number of vertices and faces in the mesh and provides information about
     * the mesh's characteristics.
     *
     * @param file_name The name of the STL file to read.
     */
    void operator()(std::string file_name)
    {
      std::ifstream input(file_name.c_str());
      std::string first;
      std::vector<Vec3d> vertices;
      Vec3d vertex;
      int nv = 0;
      int nf = 0;
      for (std::string line; getline(input, line);)
      {
        input >> first;
        if (first == "outer")
        {
          bool build_face = true;
          while (build_face)
          {
            getline(input, line);
            input >> first;
            if (first == "vertex")
            {
              input >> vertex.x >> vertex.y >> vertex.z;
              vertices.push_back(vertex);
              nv++;
            }
            else if (first != "endloop")
            {
              std::cout << "error when reading stl file, it should be endloop and not " << first << std::endl;
              build_face = false;
            }
            else
            {
              build_face = false;
            }
          }
          Face tmp(vertices);
          this->add_face(tmp);
          vertices.clear();
          nf++;
        }
      }
      lout << "========= STL Mesh ==============" << std::endl;
      lout << "Name     = " << file_name << std::endl;
      ldbg << "Vertices = " << nv << std::endl;
      ldbg << "Faces    = " << nf << std::endl;
    }
  };
} // namespace exaDEM
