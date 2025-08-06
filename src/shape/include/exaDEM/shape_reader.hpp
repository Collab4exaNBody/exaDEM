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
#include <regex>

namespace exaDEM
{
  // this function
  inline void write_shp(const shape &shp, std::stringstream &output)
  {
    int nv = shp.get_number_of_vertices();
    int ne = shp.get_number_of_edges();
    int nf = shp.get_number_of_faces();

    output << std::endl << "<" << std::endl;
    output << "name " << shp.m_name.c_str() << std::endl;
    output << "radius " << shp.m_radius << std::endl;
    output << "nv " << nv << std::endl;
    for (int i = 0; i < nv; i++)
    {
      auto &v = shp.m_vertices[i];
      output << v.x << " " << v.y << " " << v.z << std::endl;
    }
    output << "ne " << ne << std::endl;
    for (int i = 0; i < ne; i++)
    {
      output << shp.m_edges[2 * i] << " " << shp.m_edges[2 * i + 1] << std::endl;
    }
    output << "nf " << nf << std::endl;
    for (int i = 0; i < nf; i++)
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
    auto &Im = shp.m_inertia_on_mass;
    output << "I/m " << Im.x << " " << Im.y << " " << Im.z << std::endl;
    output << ">" << std::endl;
  }

  inline void write_shp(const shape &shp, std::string filename)
  {
    std::stringstream stream;
    stream.precision(16);
    write_shp(shp, stream);
    std::ofstream file(filename.c_str());
    file << stream.rdbuf();
  }

  inline shape read_shp(std::ifstream &input, bool big_shape = false)
  {
    shape shp;
    std::string key, line;
    std::vector<int> tmp_face;
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
        int nv = 0;
        input >> nv;
        if (nv > EXADEM_MAX_VERTICES && !big_shape)
        {
          lout << "=== EXADEM ERROR ===" << std::endl;
          lout << "=== Please, increase the maximum number of vertices: cmake ${Path_To_ExaDEM} -DEXADEM_MAX_VERTICES=" << nv << std::endl;
          lout << "=== ABORT ===" << std::endl;
          std::abort();
        }
        assert(nv != 0);
        for (int i = 0; i < nv; i++)
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
        int ne = 0;
        input >> ne;
        // assert(ne!=0);
        for (int i = 0; i < ne; i++)
        {
          getline(input, line);
          int e1, e2;
          input >> e1 >> e2;
          shp.add_edge(e1, e2);
        }
      }
      else if (key == "nf")
      {
        int nf = 0;
        input >> nf;
        // assert(nf!=0);
        for (int i = 0; i < nf; i++)
        {
          getline(input, line);
          int n = 0;
          input >> n;
          assert(n != 0);
          tmp_face.resize(n);
          for (int j = 0; j < n; j++)
          {
            input >> tmp_face[j];
          }
          shp.add_face(tmp_face.size(), tmp_face.data());
        }
        shp.compute_offset_faces();
      }
      else if (key == ">")
      {
        shp.obb.center = {shp.obb.center.x - position.x, shp.obb.center.y - position.y, shp.obb.center.z - position.z};
        shp.pre_compute_obb_edges(Vec3d{0, 0, 0}, Quaternion{1, 0, 0, 0});
        shp.pre_compute_obb_faces(Vec3d{0, 0, 0}, Quaternion{1, 0, 0, 0});
        printf("NUMBER OF VERTICES: %d\n", shp.get_number_of_vertices());
        printf("NUMBER OF EDGES: %d\n", shp.get_number_of_edges());
        printf("NUMBER OF FACES: %d\n", shp.get_number_of_faces());
        return shp;
      }
    }
  }
  inline shape read_shp(shape &shps, const std::string file_name, bool big_shape = false)
  {
    std::ifstream input(file_name.c_str());
    std::string first;
    for (std::string line; getline(input, line);)
    {
      input >> first;
      if (first == "<")
      {
        return read_shp(input, big_shape);
      }
    }
    lout << "Warning, no shape find into the file " << file_name << "." << std::endl;
    lout << "Warning, this file is ignored." << file_name << std::endl;
    return shape();
  }

  inline void read_shp(ParticleTypeMap& ptm, shapes& shps, const std::string file_name, bool big_shape = false)
  {
    std::ifstream input(file_name.c_str());
    std::string first;
    for (std::string line; getline(input, line);)
    {
      if (line == "<")
      {
        first = ""; // reset key
        shape shp = read_shp(input, big_shape);
/* too much verbosity        
        if (!big_shape)
          shp.print();
*/
        shp.write_paraview();
        if( ptm.find(shp.m_name) != ptm.end() )
        {
          shp.m_name = shp.m_name + "X";
          lout << "Warning, this polyhedron name is already taken, exaDEM has renamed it to: " << shp.m_name << std::endl;
        } 
        ptm[shp.m_name] = shps.get_size();
        shps.add_shape(&shp);
      }
    }
  }

  inline void read_shp(shapes &shps, const std::string file_name, bool big_shape = false)
  {
    std::ifstream input(file_name.c_str());
    std::string first;
    for (std::string line; getline(input, line);)
    {
      if (line == "<")
      {
        first = ""; // reset key
        shape shp = read_shp(input, big_shape);
        shps.add_shape(&shp);
        if (!big_shape)
          shp.print();
        shp.write_paraview();
      }
    }
  }
} // namespace exaDEM
