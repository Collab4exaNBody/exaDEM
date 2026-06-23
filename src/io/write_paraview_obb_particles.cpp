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

#include <mpi.h>

// onika
#include <onika/math/basic_types.h>
#include <onika/math/basic_types_operators.h>
#include <onika/math/basic_types_stream.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>
#include <onika/string_utils.h>

#include <cstdlib>

// exanb

#include <exanb/compute/compute_pair_optional_args.h>
#include <exanb/core/domain.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>

// exaDEM
#include <exaDEM/shapes.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace exaDEM {

struct par_obb_helper {
  int count = 0;  // use to count the number of corners
  int off = 0;    // use for lines
  std::stringstream corners;
  std::stringstream lines;
  std::stringstream offsets;
  std::stringstream ids;
  std::stringstream types;
};

inline void build_buffer_obb(const exanb::Vec3d& pos, const uint64_t id, const uint16_t type, const shape* shp,
                             const exanb::Quaternion& orient, par_obb_helper& buffers) {
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

  auto add_corner = [&buffers](vec3r& c) { buffers.corners << " " << c[0] << " " << c[1] << " " << c[2]; };
  auto add_offset = [&buffers](int incr) {
    buffers.offsets << " " << buffers.off;
    buffers.off += incr;
  };

  // [1] Add Corners

  vec3r e0 = obbi.e1;  //
  vec3r e1 = obbi.e2;  //
  vec3r e2 = obbi.e3;  //

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
  add_offset(5);                 // 1
  add_offset(5);                 // 2
  add_offset(2);                 // 3
  add_offset(2);                 // 4
  add_offset(2);                 // 5
  add_offset(2);                 // 6
  buffers.lines << " " << count  // 1
                << " " << count + 1 << " " << count + 2 << " " << count + 3 << " " << count;
  buffers.lines << " " << count + 4  // 2
                << " " << count + 5 << " " << count + 6 << " " << count + 7 << " " << count + 4;
  buffers.lines << " " << count << " " << count + 4;      // 3
  buffers.lines << " " << count + 1 << " " << count + 5;  // 4
  buffers.lines << " " << count + 2 << " " << count + 6;  // 5
  buffers.lines << " " << count + 3 << " " << count + 7;  // 6
  buffers.count += 8;                                     // number of corners
                                                          // [3] add id
  for (int i = 0; i < 8; i++) {
    buffers.ids << " " << id;
  }

  // [4] add type
  for (int i = 0; i < 8; i++) {
    buffers.types << " " << int(type);
  }
}

inline void write_vtp_obb(std::string name, par_obb_helper& buffers) {
  std::ofstream outFile(name);
  if (!outFile) {
    color_log::error("write_vtp_obb", "Impossible to open the file: " + name, false);
    return;
  }

  int n_obb = buffers.count / 8;  // count = number of corners

  outFile << "<?xml version=\"1.0\"?>" << std::endl;
  outFile << "<VTKFile type=\"PolyData\" version=\"1.0\" header_type=\"UInt64\">" << std::endl;
  outFile << "  <PolyData>" << std::endl;
  outFile << "    <Piece NumberOfPoints=\"" << n_obb * 8 << "\" NumberOfLines=\"" << n_obb * 6 << "\" NumberOfPolys=\""
          << 0 << "\">" << std::endl;
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
  if (n_obb != 0) {
    outFile << buffers.corners.rdbuf() << std::endl;
  }
  outFile << "      </DataArray>" << std::endl;
  outFile << "    </Points>" << std::endl;
  outFile << "    <Lines>" << std::endl;
  outFile << "      <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << std::endl;
  if (n_obb != 0) {
    outFile << buffers.lines.rdbuf() << std::endl;
  }
  outFile << "      </DataArray>" << std::endl;
  outFile << "      <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << std::endl;
  if (n_obb != 0) {
    outFile << buffers.offsets.rdbuf() << std::endl;
  }
  outFile << "      </DataArray>" << std::endl;
  outFile << "    </Lines>" << std::endl;
  outFile << "    </Piece>" << std::endl;
  outFile << "  </PolyData>" << std::endl;
  outFile << "</VTKFile>" << std::endl;
}

inline void write_pvtp_obb(std::string basedir, std::string basename, size_t number_of_files) {
  std::string name = basedir + "/" + basename + ".pvtp";
  std::ofstream outFile(name);
  if (!outFile) {
    color_log::error("write_pvtp_obb", "Impossible to open the file: " + name, false);
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
  for (size_t i = 0; i < number_of_files; i++) {
    std::string file = onika::format_string(subfile, i);
    outFile << "    <Piece Source=\"" << file << "\"/>" << std::endl;
  }
  outFile << "  </PPolyData>" << std::endl;
  outFile << "</VTKFile>" << std::endl;
}

template <class GridT, class = AssertGridHasFields<GridT>>
class WriteParaviewOBBParticlesOperator : public OperatorNode {
  using ComputeFields = FieldSet<field::_rx, field::_ry, field::_rz, field::_type, field::_orient>;
  static constexpr ComputeFields compute_field_set{};
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(Domain, domain, INPUT, REQUIRED);
  ADD_SLOT(std::string, basename, INPUT, "obb", DocString{"Output filename"});
  ADD_SLOT(std::string, dir_name, INPUT, REQUIRED, DocString{"Main output directory."});
  ADD_SLOT(long, timestep, INPUT, REQUIRED, DocString{"Iteration number"});
  ADD_SLOT(shapes, shapes_collection, INPUT, REQUIRED, DocString{"Collection of shapes"});

 public:
  inline std::string documentation() const final {
    return R"EOF( 
      This operator dumps obb into a paraview output file.

      YAML exmaple:
 
        - write_paraview_obb_particles
    	    			)EOF";
  }

  inline void execute() final {
    // mpi stuff
    int rank, size;
    MPI_Comm_rank(*mpi, &rank);
    MPI_Comm_size(*mpi, &size);

    std::string ts = "%010d";
    std::string rk = "%06d";

    std::string directory = (*dir_name) + "/ParaviewOutputFiles/" + (*basename) + "_" + ts;
    directory = onika::format_string(directory, *timestep);
    std::string filename = directory + "/" + rk + ".vtp";
    filename = onika::format_string(filename, rank);

    // prepro
    if (rank == 0) {
      namespace fs = std::filesystem;
      fs::create_directories(directory);
    }

    MPI_Barrier(*mpi);

    auto& shps = *shapes_collection;
    const auto cells = grid->cells();
    const size_t n_cells = grid->number_of_cells();

    par_obb_helper buffers;

    // fill string buffers
    for (size_t cell_a = 0; cell_a < n_cells; cell_a++) {
      if (grid->is_ghost_cell(cell_a)) {
        continue;
      }
      const int n_particles = cells[cell_a].size();
      auto* __restrict__ rx = cells[cell_a][field::rx];
      auto* __restrict__ ry = cells[cell_a][field::ry];
      auto* __restrict__ rz = cells[cell_a][field::rz];
      auto* __restrict__ id = cells[cell_a][field::id];
      auto* __restrict__ type = cells[cell_a][field::type];
      auto* __restrict__ orient = cells[cell_a][field::orient];
      for (int j = 0; j < n_particles; j++) {
        exanb::Vec3d pos{rx[j], ry[j], rz[j]};
        const shape* shp = shps[type[j]];
        build_buffer_obb(pos, id[j], type[j], shp, orient[j], buffers);
      }
    };

    if (rank == 0) {
      std::string dir = *dir_name + "/ParaviewOutputFiles/";
      std::string name = *basename + "_" + ts;
      name = onika::format_string(name, *timestep);
      exaDEM::write_pvtp_obb(dir, name, size);
    }
    exaDEM::write_vtp_obb(filename, buffers);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(write_paraview_obb_particles) {
  OperatorNodeFactory::instance()->register_factory("write_paraview_obb_particles",
                                                    make_grid_variant_operator<WriteParaviewOBBParticlesOperator>);
}
}  // namespace exaDEM
