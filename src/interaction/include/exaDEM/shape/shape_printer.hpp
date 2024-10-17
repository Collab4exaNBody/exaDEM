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

#include <exaDEM/shape/shape.hpp>
#include <iostream>

namespace exaDEM
{
  void build_buffer(const exanb::Vec3d &pos, const shape *shp, const exanb::Quaternion &orient, size_t &polygon_offset_in_stream, size_t &n_vertices, size_t &n_polygon, std::stringstream &buff_position, std::stringstream &buff_faces, std::stringstream &buff_offset)
  {
    auto writer_v = [](const exanb::Vec3d &v, std::stringstream &out, const exanb::Vec3d &p, const exanb::Quaternion &q)
    {
      exanb::Vec3d new_pos = p + q * v;
      out << " " << new_pos.x << " " << new_pos.y << " " << new_pos.z;
    };

    shp->for_all_vertices(writer_v, buff_position, pos, orient);

    size_t n_faces = shp->get_number_of_faces();
    n_polygon += n_faces;

    // faces
    auto writer_f = [](const size_t size, const int *data, std::stringstream &sface, std::stringstream &soffset, size_t &offset, size_t point_off)
    {
      soffset << offset + size << " ";
      offset += size;
      for (size_t it = 0; it < size; it++)
        sface << " " << data[it] + point_off;
    };
    shp->for_all_faces(writer_f, buff_faces, buff_offset, polygon_offset_in_stream, n_vertices);
    n_vertices += shp->get_number_of_vertices();
  }

  void write_vtp(std::string name, size_t n_vertices, size_t n_polygons, std::stringstream &buff_vertices, std::stringstream &buff_faces, std::stringstream &buff_offsets)
  {
    name = name + ".vtp";
    std::ofstream outFile(name);
    if (!outFile)
    {
      std::cerr << "Erreur : impossible de créer le fichier de sortie suivant: " << name << std::endl;
      return;
    }

    outFile << "<VTKFile type=\"PolyData\">" << std::endl;
    outFile << " <PolyData>" << std::endl;
    outFile << "   <Piece NumberOfPoints=\"" << n_vertices << "\" NumberOfPolys=\"" << n_polygons << "\">" << std::endl;
    outFile << "   <Points>" << std::endl;
    outFile << "     <DataArray type=\"Float64\" Name=\"\"  NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
    if (n_vertices != 0)
      outFile << buff_vertices.rdbuf() << std::endl;
    outFile << "     </DataArray>" << std::endl;
    outFile << "   </Points>" << std::endl;
    outFile << "   <Polys>" << std::endl;
    outFile << "     <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << std::endl;
    if (n_polygons != 0)
      outFile << buff_faces.rdbuf() << std::endl;
    outFile << "     </DataArray>" << std::endl;
    outFile << "     <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << std::endl;
    if (n_polygons != 0)
      outFile << buff_offsets.rdbuf() << std::endl;
    outFile << "     </DataArray>" << std::endl;
    outFile << "   </Polys>" << std::endl;
    outFile << "  </Piece>" << std::endl;
    outFile << " </PolyData>" << std::endl;
    outFile << "</VTKFile>" << std::endl;
  }

  void write_pvtp(std::string basedir, std::string basename, size_t number_of_files)
  {

    std::string name = basedir + "/" + basename + ".pvtp";
    std::ofstream outFile(name);
    if (!outFile)
    {
      std::cerr << "Erreur : impossible de créer le fichier de sortie suivant: " << name << std::endl;
      return;
    }
    outFile << " <VTKFile type=\"PPolyData\"> " << std::endl;
    outFile << "   <PPolyData GhostLevel=\"0\">" << std::endl;
    outFile << "     <PPoints>" << std::endl;
    outFile << "       <PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>" << std::endl;
    outFile << "     </PPoints> " << std::endl;
    for (size_t i = 0; i < number_of_files; i++)
    {
      std::string subfile = basename + "/" + basename + "_" + std::to_string(i) + ".vtp";
      outFile << "     <Piece Source=\"" << subfile << "\"/>" << std::endl;
    }
    outFile << "   </PPolyData>" << std::endl;
    outFile << " </VTKFile>" << std::endl;
  }
} // namespace exaDEM
