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
#include <onika/math/basic_types.h>
#include <onika/math/basic_types_operators.h>
#include <onika/math/basic_types_stream.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>
#include <onika/string_utils.h>

// exaNBody
#include <exanb/core/domain.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/xform.h>

#include <exaDEM/shapes.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace exaDEM {
using namespace exanb;

/**
 * @brief Helper for assembling parallel polygonal data (MPI / VTK).
 */
struct par_poly_helper {
  bool mpi_rank;     ///< MPI rank flag
  bool has_cluster;  ///< Cluster data available

  int n_vertices = 0;  ///< Number of vertices
  int n_lines = 0;     ///< Number of lines
  int n_faces = 0;     ///< Number of faces
  int n_polygons = 0;  ///< Number of polygons

  uint64_t incr_offset = 0;       ///< Polygon connectivity offset
  uint64_t incr_line_offset = 0;  ///< Line connectivity offset

  std::stringstream vertices;      ///< Vertex coordinates
  std::stringstream lines;         ///< Line connectivity
  std::stringstream faces;         ///< Face connectivity
  std::stringstream line_offsets;  ///< Line offsets
  std::stringstream tube_size;     ///< Tube size (visualization)
  std::stringstream offsets;       ///< Polygon offsets
  std::stringstream ids;           ///< IDs
  std::stringstream types;         ///< Cell types
  std::stringstream velocities;    ///< Velocity field
  std::stringstream clusters;      ///< Cluster IDs
  std::stringstream ranks;         ///< MPI ranks
};

/**
 * @brief Build polyhedron data into stream buffers.
 *
 * @tparam has_field_cluster Enable cluster field output.
 * @param pos      Position of the polyhedron.
 * @param shp      Pointer to shape definition.
 * @param orient   Orientation (quaternion).
 * @param id       Element ID.
 * @param type     Element type.
 * @param vx,vy,vz Velocity components.
 * @param h        Homoethety.
 * @param cluster  Cluster ID (used if has_field_cluster).
 * @param buffers  Output buffer helper.
 */
template <bool has_field_cluster>
inline void build_buffer_polyhedron(const exanb::Vec3d& pos, const shape* shp, const exanb::Quaternion& orient,
                                    uint64_t id, uint16_t type, double vx, double vy, double vz, double h,
                                    uint32_t cluster, par_poly_helper& buffers) {
  auto writer_v = [&buffers](const exanb::Vec3d& v, const exanb::Vec3d& p, const double& h,
                             const exanb::Quaternion& q) {
    exanb::Vec3d new_pos = p + q * (h * v);
    buffers.vertices << " " << new_pos.x << " " << new_pos.y << " " << new_pos.z;
  };

  auto writer_components = [&buffers](const exanb::Vec3d& v, uint64_t i, uint16_t t, double v_x, double v_y, double v_z,
                                      uint32_t c) {
    buffers.ids << " " << i;
    buffers.types << " " << t;
    buffers.velocities << " " << v_x << " " << v_y << " " << v_z;
    if constexpr (has_field_cluster) {
      buffers.clusters << " " << c;
    }
  };

  buffers.has_cluster = has_field_cluster;

  shp->for_all_vertices(writer_v, pos, h, orient);
  shp->for_all_vertices(writer_components, id, type, vx, vy, vz, cluster);

  size_t n_faces = shp->get_number_of_faces();
  buffers.n_polygons += n_faces;

  if (n_faces == 0) {
    auto writer_e = [&buffers, &shp](const int first, const int second) {
      buffers.n_lines++;
      buffers.lines << " " << buffers.n_vertices + first << " " << buffers.n_vertices + second;
      buffers.tube_size << " " << shp->minkowski();
      buffers.line_offsets << " " << buffers.incr_line_offset;
      buffers.incr_line_offset += 2;
    };
    shp->for_all_edges(writer_e);
  }

  // faces
  auto writer_f = [&buffers](const size_t size, const int* data) {
    buffers.offsets << buffers.incr_offset + size << " ";
    buffers.incr_offset += size;
    for (size_t it = 0; it < size; it++) {
      buffers.faces << " " << data[it] + buffers.n_vertices;
    }
  };
  shp->for_all_faces(writer_f);
  buffers.n_vertices += shp->get_number_of_vertices();  // warning, increment n_vertices after all_faces
}

inline void write_vtp_polyhedron(std::string name, par_poly_helper& buffers) {
  std::ofstream outFile(name);
  if (!outFile) {
    color_log::error("write_vtp_polyhedron", "Impossible to open the file: " + name, false);
    return;
  }

  outFile << "<?xml version=\"1.0\"?>" << std::endl;
  outFile << "<VTKFile type=\"PolyData\">" << std::endl;
  outFile << "  <PolyData>" << std::endl;
  outFile << "    <Piece NumberOfPoints=\"" << buffers.n_vertices << "\" NumberOfLines=\"" << buffers.n_lines
          << "\" NumberOfPolys=\"" << buffers.n_polygons << "\">" << std::endl;
  outFile << "    <PointData>" << std::endl;

  /// PARTICLE FIELDS
  outFile << "      <DataArray type=\"Int64\" Name=\"Id\"  NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
  outFile << buffers.ids.rdbuf() << std::endl;
  outFile << "      </DataArray>" << std::endl;
  outFile << "      <DataArray type=\"Int32\" Name=\"Type\"  NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
  outFile << buffers.types.rdbuf() << std::endl;
  outFile << "      </DataArray>" << std::endl;
  outFile << "      <DataArray type=\"Float64\" Name=\"Velocity\"  NumberOfComponents=\"3\" format=\"ascii\">"
          << std::endl;
  outFile << buffers.velocities.rdbuf() << std::endl;
  outFile << "      </DataArray>" << std::endl;

  if (buffers.has_cluster) {  // only for fragmentation
    outFile << "      <DataArray type=\"Int32\" Name=\"Cluster\"  NumberOfComponents=\"1\" format=\"ascii\">"
            << std::endl;
    if (buffers.n_vertices != 0) {
      outFile << buffers.clusters.rdbuf() << std::endl;
    }
    outFile << "      </DataArray>" << std::endl;
  }

  if (buffers.mpi_rank) {  // MPI  rank - optional
    outFile << "      <DataArray type=\"Int32\" Name=\"MPI rank\"  NumberOfComponents=\"1\" format=\"ascii\">"
            << std::endl;
    if (buffers.n_vertices != 0) {
      outFile << buffers.ranks.rdbuf() << std::endl;
    }
    outFile << "      </DataArray>" << std::endl;
  }
  outFile << "    </PointData>" << std::endl;
  outFile << "    <Points>" << std::endl;
  outFile << "      <DataArray type=\"Float64\" Name=\"\"  NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
  if (buffers.n_vertices != 0) {
    outFile << buffers.vertices.rdbuf() << std::endl;
  }
  outFile << "      </DataArray>" << std::endl;
  outFile << "    </Points>" << std::endl;
  if (buffers.n_lines != 0) {
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
  if (buffers.n_polygons != 0) {
    outFile << "    <Polys>" << std::endl;
    outFile << "      <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << std::endl;
    outFile << buffers.faces.rdbuf() << std::endl;
    outFile << "      </DataArray>" << std::endl;
    outFile << "      <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << std::endl;
    if (buffers.n_polygons != 0) {
      outFile << buffers.offsets.rdbuf() << std::endl;
    }
    outFile << "      </DataArray>" << std::endl;
    outFile << "    </Polys>" << std::endl;
  }
  outFile << "    </Piece>" << std::endl;
  outFile << "  </PolyData>" << std::endl;
  outFile << "</VTKFile>" << std::endl;
}

inline void write_pvtp_polyhedron(std::string filename, size_t number_of_files, par_poly_helper& buffers) {
  std::string name = filename + ".pvtp";
  std::ofstream outFile(name);
  if (!outFile) {
    color_log::error("write_pvtp_polyhedron", "Impossible to open the file: " + name, false);
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
  if (buffers.has_cluster) {
    outFile << "      <PDataArray type=\"Int32\" Name=\"Cluster\"  NumberOfComponents=\"1\"/>" << std::endl;
  }
  if (buffers.mpi_rank) {
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
  for (size_t i = 0; i < number_of_files; i++) {
    std::string file = onika::format_string(subfile, i);
    outFile << "    <Piece Source=\"" << file << "\"/>" << std::endl;
  }
  outFile << "  </PPolyData>" << std::endl;
  outFile << "</VTKFile>" << std::endl;
}

template <class GridT, class = AssertGridHasFields<GridT>>
class WriteParaviewPolyhedraOperator : public OperatorNode {
  using ComputeFields = FieldSet<field::_rx, field::_ry, field::_rz, field::_type, field::_orient>;
  static constexpr ComputeFields compute_field_set{};
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(Domain, domain, INPUT, REQUIRED);
  ADD_SLOT(std::string, filename, INPUT, "output");
  ADD_SLOT(long, timestep, INPUT, REQUIRED, DocString{"Iteration number"});
  ADD_SLOT(shapes, shapes_collection, INPUT, REQUIRED, DocString{"Collection of shapes"});

  // optionnal
  ADD_SLOT(bool, mpi_rank, INPUT, false, DocString{"Add a field containing the mpi rank."});

 public:
  inline std::string documentation() const final {
    return R"EOF( 
      This operator initialize shapes data structure from a shape input file.

      YAML example:

        - write_paraview_polyhedra:
           filename: "OptionalFilename_%10d"
           mpi_rank: true
                )EOF";
  }

  inline void execute() final {
    using ParticleTuple = decltype(grid->cells()[0][0]);
    static constexpr bool has_field_cluster = ParticleTuple::has_field(field::cluster);
    // mpi stuff
    int rank, size;
    MPI_Comm_rank(*mpi, &rank);
    MPI_Comm_size(*mpi, &size);

    if (rank == 0) {
      std::filesystem::create_directories(*filename);
    }

    MPI_Barrier(*mpi);

    auto& shps = *shapes_collection;
    const auto cells = grid->cells();
    const size_t n_cells = grid->number_of_cells();
    par_poly_helper buffers = {*mpi_rank};  // it conatins streams

    bool defbox = !domain->xform_is_identity();
    LinearXForm xform;
    if (defbox) {
      xform.m_matrix = domain->xform();
    }

    uint32_t* cluster = nullptr;

    uint32_t cj = 0;  // default value, not used

    // fill string buffers
    for (size_t cell_a = 0; cell_a < n_cells; cell_a++) {
      if (grid->is_ghost_cell(cell_a)) {
        continue;
      }
      auto& cell = cells[cell_a];
      const int n_particles = cell.size();
      auto* __restrict__ rx = cell[field::rx];
      auto* __restrict__ ry = cell[field::ry];
      auto* __restrict__ rz = cell[field::rz];
      auto* __restrict__ vx = cell[field::vx];
      auto* __restrict__ vy = cell[field::vy];
      auto* __restrict__ vz = cell[field::vz];
      auto* __restrict__ type = cell[field::type];
      auto* __restrict__ id = cell[field::id];
      auto* __restrict__ h = cell[field::homothety];
      auto* __restrict__ orient = cell[field::orient];
      if constexpr (has_field_cluster) {
        cluster = cell[field::cluster];
      }
      for (int j = 0; j < n_particles; j++) {
        exanb::Vec3d pos{rx[j], ry[j], rz[j]};
        if (defbox) pos = xform.transformCoord(pos);
        const shape* shp = shps[type[j]];
        if constexpr (has_field_cluster) {
          cj = cluster[j];
        }
        build_buffer_polyhedron<has_field_cluster>(pos, shp, orient[j], id[j], type[j], vx[j], vy[j], vz[j], h[j], cj,
                                                   buffers);
      }
    };

    if (rank == 0) {
      exaDEM::write_pvtp_polyhedron(*filename, size, buffers);
    }

    if (buffers.mpi_rank) {  // add ranks
      for (int i = 0; i < buffers.n_vertices; i++) {
        buffers.ranks << rank << " ";
      }
    }

    std::string file = *filename + "/%06d.vtp";
    file = onika::format_string(file, rank);
    exaDEM::write_vtp_polyhedron(file, buffers);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(write_paraview_polyhedra) {
  OperatorNodeFactory::instance()->register_factory("write_paraview_polyhedra",
                                                    make_grid_variant_operator<WriteParaviewPolyhedraOperator>);
}
}  // namespace exaDEM
