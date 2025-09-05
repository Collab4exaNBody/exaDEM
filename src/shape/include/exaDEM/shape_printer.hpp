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
software DISTributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
#pragma once

#include <exaDEM/shape.hpp>
#include <onika/string_utils.h>
#include <iostream>

namespace exaDEM
{
  struct par_obb_helper
  {
    int count = 0;  // use to count the number of corners
    int off = 0;    // use for lines
    std::stringstream corners;
    std::stringstream lines;
    std::stringstream offsets;
    std::stringstream ids;
    std::stringstream types;
  };

  struct par_poly_helper
  {
    bool mpi_rank;
    int n_vertices = 0;
    int n_lines = 0;
    int n_faces = 0;
    int n_polygons = 0;
    uint64_t incr_offset = 0;
    uint64_t incr_line_offset = 0;
    std::stringstream vertices; 
    std::stringstream lines;
    std::stringstream faces;
    std::stringstream line_offsets;
    std::stringstream tube_size;
    std::stringstream offsets;
    std::stringstream ids;
    std::stringstream types;
    std::stringstream velocities;
    std::stringstream ranks; 
  };

  inline void build_buffer_polyhedron(const exanb::Vec3d &pos, const shape *shp, const exanb::Quaternion &orient, uint64_t id, uint16_t type, double vx, double vy, double vz, par_poly_helper& buffers)
  {
    auto writer_v = [&buffers](const exanb::Vec3d &v, const exanb::Vec3d &p, const exanb::Quaternion &q)
    {
      exanb::Vec3d new_pos = p + q * v;
      buffers.vertices << " " << new_pos.x << " " << new_pos.y << " " << new_pos.z;
    };

    auto writer_components = [&buffers] (const exanb::Vec3d &v, uint64_t i, uint16_t t, double v_x, double v_y, double v_z)
    {
      buffers.ids   << " " << i;
      buffers.types << " " << t;
      buffers.velocities << " " << v_x << " " << v_y << " " << v_z;
    };

    shp->for_all_vertices(writer_v, pos, orient);
    shp->for_all_vertices(writer_components, id, type, vx, vy, vz);

    size_t n_faces = shp->get_number_of_faces();
    buffers.n_polygons += n_faces;

    if(n_faces == 0)
    {
      auto writer_e = [&buffers, &shp](const int first, const int second)
      {
        buffers.n_lines ++;
        buffers.lines << " " << buffers.n_vertices + first << " " << buffers.n_vertices + second;
        buffers.tube_size << " " << shp->m_radius;
        buffers.line_offsets << " " << buffers.incr_line_offset; buffers.incr_line_offset += 2; 
      };
      shp->for_all_edges(writer_e);
    }

    // faces
    auto writer_f = [&buffers](const size_t size, const int *data)
    {
      buffers.offsets << buffers.incr_offset + size << " ";
      buffers.incr_offset += size;
      for (size_t it = 0; it < size; it++)
        buffers.faces << " " << data[it] + buffers.n_vertices;
    };
    shp->for_all_faces(writer_f);
    buffers.n_vertices += shp->get_number_of_vertices(); // warning, increment n_vertices after all_faces
  }


  inline void build_buffer_obb(const exanb::Vec3d &pos, const uint64_t id, const uint16_t type, const shape *shp, const exanb::Quaternion &orient, par_obb_helper& buffers)
  {
    // add [1] corner [2] lines [3] id [4] type

    double homothety = 1;
    vec3r p = conv_to_vec3r(pos);
    OBB obbi = shp->obb;
    quat Q = conv_to_quat(orient);
    obbi.rotate(Q);
    obbi.extent *= homothety;
    obbi.center *= homothety;
    obbi.center += p;
    vec3r corner;


    auto add_corner = [&buffers] (vec3r& c) { buffers.corners << " " << c[0] << " " << c[1] << " " << c[2]; };
    auto add_offset = [&buffers] (int incr) { buffers.offsets << " " << buffers.off; buffers.off += incr; };

    // [1] Add Corners

    vec3r e0 = obbi.e1; // 
    vec3r e1 = obbi.e2; // 
    vec3r e2 = obbi.e3; // 

    corner = obbi.center - obbi.extent[0] * e0 - obbi.extent[1] * e1 - obbi.extent[2] * e2;
    add_corner(corner);
    corner += 2.0 * obbi.extent[0] * e0;
    add_corner(corner);
    corner += 2.0 * obbi.extent[1] * e1;
    add_corner(corner);
    corner -= 2.0 * obbi.extent[0] * e0;
    add_corner(corner);

    corner = obbi.center - obbi.extent[0] * e0 - obbi.extent[1] * e1 - obbi.extent[2] * e2;
    corner += 2.0 * obbi.extent[2] * e2;
    add_corner(corner);
    corner += 2.0 * obbi.extent[0] * e0;
    add_corner(corner);
    corner += 2.0 * obbi.extent[1] * e1;
    add_corner(corner);
    corner -= 2.0 * obbi.extent[0] * e0;
    add_corner(corner);

    // [2] add lines
    int count = buffers.count;
    add_offset(5); // 1
    add_offset(5); // 2
    add_offset(2); // 3
    add_offset(2); // 4
    add_offset(2); // 5 
    add_offset(2); // 6
    buffers.lines << " " << count // 1
                  << " " << count + 1 
                  << " " << count + 2 
                  << " " << count + 3 
                  << " " << count;
    buffers.lines << " " << count + 4 // 2
                  << " " << count + 5 
                  << " " << count + 6 
                  << " " << count + 7 
                  << " " << count + 4;
    buffers.lines << " " << count << " " << count + 4; // 3
    buffers.lines << " " << count + 1 << " " << count + 5; // 4
    buffers.lines << " " << count + 2 << " " << count + 6; // 5
    buffers.lines << " " << count + 3 << " " << count + 7; // 6
    buffers.count += 8; // number of corners
    // [3] add id
    for(int i = 0 ; i < 8 ; i++)
    {
      buffers.ids << " " << id;
    }

    // [4] add type
    for(int i = 0 ; i < 8 ; i++)
    {
      buffers.types << " " << int(type);
    }
  }


  inline void write_vtp_polyhedron(std::string name, par_poly_helper& buffers)
  {
    std::ofstream outFile(name);
    if (!outFile)
    {
      std::cerr << "[ERROR] impossible to open the file: " << name << std::endl;
      return;
    }

    outFile << "<?xml version=\"1.0\"?>" << std::endl;
    outFile << "<VTKFile type=\"PolyData\">" << std::endl;
    outFile << "  <PolyData>" << std::endl;
    outFile << "    <Piece NumberOfPoints=\"" << buffers.n_vertices << "\" NumberOfLines=\"" << buffers.n_lines << "\" NumberOfPolys=\"" << buffers.n_polygons << "\">" << std::endl;
    outFile << "    <PointData>" << std::endl;

    /// PARTICLE FIELDS
    outFile << "      <DataArray type=\"Int64\" Name=\"Id\"  NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
    outFile << buffers.ids.rdbuf() << std::endl;
    outFile << "      </DataArray>" << std::endl;
    outFile << "      <DataArray type=\"Int32\" Name=\"Type\"  NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
    outFile << buffers.types.rdbuf() << std::endl;
    outFile << "      </DataArray>" << std::endl;
    outFile << "      <DataArray type=\"Float64\" Name=\"Velocity\"  NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
    outFile << buffers.velocities.rdbuf() << std::endl;
    outFile << "      </DataArray>" << std::endl;
    if(buffers.mpi_rank) // MPI  rank - optional
    {
      outFile << "      <DataArray type=\"Int32\" Name=\"MPI rank\"  NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
      if (buffers.n_vertices != 0) outFile << buffers.ranks.rdbuf() << std::endl;
      outFile << "      </DataArray>" << std::endl;
    }
    outFile << "    </PointData>" << std::endl;
    outFile << "    <Points>" << std::endl;
    outFile << "      <DataArray type=\"Float64\" Name=\"\"  NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
    if (buffers.n_vertices != 0)
      outFile << buffers.vertices.rdbuf() << std::endl;
    outFile << "      </DataArray>" << std::endl;
    outFile << "    </Points>" << std::endl;
    if (buffers.n_lines != 0)
    { 
      outFile << "    <Lines>" << std::endl;
      outFile << "      <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << std::endl;
      outFile << buffers.lines.rdbuf() << std::endl;
      outFile << "      </DataArray>" << std::endl;
      outFile << "      <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << std::endl;
      outFile << buffers.line_offsets.rdbuf() << std::endl;
      outFile << "      </DataArray>" << std::endl;
      outFile << "      <DataArray type=\"Float64\" Name=\"TubeSize\" format=\"ascii\">" << std::endl;
      outFile << buffers.tube_size.rdbuf() << std::endl;
      outFile << "      </DataArray>" << std::endl;
      outFile << "    </Lines>" << std::endl;
    }
    /// PARTICLE FACES
    if (buffers.n_polygons != 0)
    {
      outFile << "    <Polys>" << std::endl;
      outFile << "      <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << std::endl;
      outFile << buffers.faces.rdbuf() << std::endl;
      outFile << "      </DataArray>" << std::endl;
      outFile << "      <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << std::endl;
      if (buffers.n_polygons != 0)
        outFile << buffers.offsets.rdbuf() << std::endl;
      outFile << "      </DataArray>" << std::endl;
      outFile << "    </Polys>" << std::endl;
    }
    outFile << "    </Piece>" << std::endl;
    outFile << "  </PolyData>" << std::endl;
    outFile << "</VTKFile>" << std::endl;
  }

  inline void write_vtp_obb(std::string name, par_obb_helper& buffers)
  {
    std::ofstream outFile(name);
    if (!outFile)
    {
      std::cerr << "[ERROR] Impossible to open the file: " << name << std::endl;
      return;
    }

    int n_obb = buffers.count / 8; // count = number of corners 

    outFile << "<?xml version=\"1.0\"?>" << std::endl;
    outFile << "<VTKFile type=\"PolyData\" version=\"1.0\" header_type=\"UInt64\">" << std::endl;
    outFile << "  <PolyData>" << std::endl;
    outFile << "    <Piece NumberOfPoints=\"" << n_obb * 8 << "\" NumberOfLines=\"" << n_obb * 6 << "\" NumberOfPolys=\"" << 0 << "\">" << std::endl;
    outFile << "    <PointData>" << std::endl;
    outFile << "      <DataArray type=\"Int64\" Name=\"Id\"  NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
    outFile << buffers.ids.rdbuf() << std::endl;
    outFile << "      </DataArray>" << std::endl;
    outFile << "      <DataArray type=\"Int32\" Name=\"Type\"  NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
    outFile << buffers.types.rdbuf() << std::endl;
    outFile << "      </DataArray>" << std::endl;
    outFile << "    </PointData>" << std::endl;
    outFile << "    <Points>" << std::endl;
    outFile << "      <DataArray type=\"Float64\" Name=\"\"  NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
    if (n_obb != 0)
      outFile << buffers.corners.rdbuf() << std::endl;
    outFile << "      </DataArray>" << std::endl;
    outFile << "    </Points>" << std::endl;
    outFile << "    <Lines>" << std::endl;
    outFile << "      <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << std::endl;
    if (n_obb != 0)
      outFile << buffers.lines.rdbuf() << std::endl;
    outFile << "      </DataArray>" << std::endl;
    outFile << "      <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << std::endl;
    if (n_obb != 0)
      outFile << buffers.offsets.rdbuf() << std::endl;
    outFile << "      </DataArray>" << std::endl;
    outFile << "    </Lines>" << std::endl;
    outFile << "    </Piece>" << std::endl;
    outFile << "  </PolyData>" << std::endl;
    outFile << "</VTKFile>" << std::endl;
  }

  inline void write_pvtp_polyhedron(std::string filename, size_t number_of_files, par_poly_helper& buffers)
  {
    std::string name = filename + ".pvtp";
    std::ofstream outFile(name);
    if (!outFile)
    {
      std::cerr << "[ERROR] Impossible to open the file: " << name << std::endl;
      return;
    }
    outFile << "<?xml version=\"1.0\"?>" << std::endl;
    outFile << "<VTKFile type=\"PPolyData\"> " << std::endl;
    outFile << "  <PPolyData GhostLevel=\"0\">" << std::endl;
    outFile << "    <PPointData>" << std::endl;
    /// PARTICLE FIELDS
    outFile << "      <PDataArray type=\"Int64\" Name=\"Id\"  NumberOfComponents=\"1\"/>" << std::endl;
    outFile << "      <PDataArray type=\"Int32\" Name=\"Type\"  NumberOfComponents=\"1\"/>" << std::endl;
    outFile << "      <PDataArray type=\"Float64\" Name=\"Velocity\"  NumberOfComponents=\"3\"/>" << std::endl;
    if( buffers.mpi_rank) 
    {
      outFile << "      <PDataArray type=\"Int32\" Name=\"MPI rank\"  NumberOfComponents=\"1\"/>" << std::endl;
    }
    outFile << "    </PPointData>" << std::endl;
    outFile << "    <PPoints>" << std::endl;
    outFile << "      <PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>" << std::endl;
    outFile << "    </PPoints> " << std::endl;
    outFile << "    <PLines>" << std::endl;
    outFile << "      <PDataArray type=\"Int32\" Name=\"connectivity\"  NumberOfComponents=\"1\"/>" << std::endl;
    outFile << "      <PDataArray type=\"Float64\" Name=\"TubeSize\"  NumberOfComponents=\"1\"/>" << std::endl;
    outFile << "    </PLines> " << std::endl;
    std::filesystem::path full_path(filename);
    std::string directory = full_path.filename().string();
    std::string subfile = directory + "/%06d.vtp";
    for (size_t i = 0; i < number_of_files; i++)
    {
      std::string file = onika::format_string(subfile,  i);
      outFile << "    <Piece Source=\"" << file << "\"/>" << std::endl;
    }
    outFile << "  </PPolyData>" << std::endl;
    outFile << "</VTKFile>" << std::endl;
  }

  inline void write_pvtp_obb(std::string basedir, std::string basename, size_t number_of_files)
  {

    std::string name = basedir + "/" + basename + ".pvtp";
    std::ofstream outFile(name);
    if (!outFile)
    {
      std::cerr << "[ERROR] Impossible to open the file: " << name << std::endl;
      return;
    }

    outFile << "<?xml version=\"1.0\"?>" << std::endl;
    outFile << "<VTKFile type=\"PPolyData\"> " << std::endl;
    outFile << "  <PPolyData GhostLevel=\"0\">" << std::endl;
    outFile << "    <PPointData>" << std::endl;
    outFile << "      <PDataArray type=\"Int64\" Name=\"Id\"  NumberOfComponents=\"1\"/>" << std::endl;
    outFile << "      <PDataArray type=\"Int32\" Name=\"Type\"  NumberOfComponents=\"1\"/>" << std::endl;
    outFile << "    </PPointData>" << std::endl;
    outFile << "    <PPoints>" << std::endl;
    outFile << "      <PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>" << std::endl;
    outFile << "    </PPoints> " << std::endl;
    outFile << "    <PLines>" << std::endl;
    outFile << "      <PDataArray type=\"Int32\" Name=\"connectivity\"  NumberOfComponents=\"1\"/>" << std::endl;
    outFile << "      <PDataArray type=\"Int32\" Name=\"offsets\"  NumberOfComponents=\"1\"/>" << std::endl;
    outFile << "    </PLines> " << std::endl;
    std::string subfile = basename + "/%06d.vtp";
    for (size_t i = 0; i < number_of_files; i++)
    {
      std::string file = onika::format_string(subfile,  i);
      outFile << "    <Piece Source=\"" << file << "\"/>" << std::endl;
    }
    outFile << "  </PPolyData>" << std::endl;
    outFile << "</VTKFile>" << std::endl;
  }
} // namespace exaDEM
