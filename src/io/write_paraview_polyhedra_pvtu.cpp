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
#include <exanb/core/domain.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/xform.h>
#include <mpi.h>
#include <onika/math/basic_types.h>
#include <onika/math/basic_types_operators.h>
#include <onika/math/basic_types_stream.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>
#include <onika/string_utils.h>

#include <cstdlib>
#include <exaDEM/shape_printer.hpp>
#include <exaDEM/shapes.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace exaDEM {
using namespace exanb;

struct par_vtu_poly_helper {
  bool mpi_rank = false;
  bool has_cluster = false;
  int n_vertices = 0;
  int n_cells = 0;
  int64_t connectivity_offset = 0;
  int64_t face_offset = 0;
  std::stringstream vertices;
  std::stringstream connectivity;
  std::stringstream offsets;
  std::stringstream faces;
  std::stringstream faceoffsets;
  std::stringstream ids;
  std::stringstream types;
  std::stringstream velocities;
  std::stringstream stress;
  std::stringstream clusters;
  std::stringstream ranks;
};

template <bool has_field_cluster>
inline void build_buffer_vtu_polyhedron(const exanb::Vec3d& pos, const shape* shp, const exanb::Quaternion& orient,
                                        uint64_t id, uint16_t type, double vx, double vy, double vz, double h,
                                        exanb::Mat3d& stress, uint32_t cluster, int mpi_rank,
                                        par_vtu_poly_helper& buffers) {
  buffers.has_cluster = has_field_cluster;

  const int vertex_base = buffers.n_vertices;
  const int n_vertices = shp->get_number_of_vertices();

  auto writer_vertex = [&buffers](const exanb::Vec3d& v, const exanb::Vec3d& p, const double& h,
                                  const exanb::Quaternion& q) {
    exanb::Vec3d new_pos = p + q * (h * v);
    buffers.vertices << " " << new_pos.x << " " << new_pos.y << " " << new_pos.z;
  };

  shp->for_all_vertices(writer_vertex, pos, h, orient);

  buffers.n_vertices += n_vertices;

  const int n_faces = shp->get_number_of_faces();
  for (int vid = 0; vid < n_vertices; vid++) {
    buffers.connectivity << " " << vertex_base + vid;
  }
  buffers.connectivity_offset += n_vertices;
  buffers.offsets << " " << buffers.connectivity_offset;

  buffers.faces << " " << n_faces;
  buffers.face_offset += 1;

  for (int face_idx = 0; face_idx < n_faces; face_idx++) {
    auto [face_vertices, face_size] = shp->get_face(face_idx);
    buffers.face_offset += face_size + 1;

    buffers.faces << " " << face_size;
    for (int it = 0; it < face_size; it++) {
      buffers.faces << " " << vertex_base + face_vertices[it];
    }
  }
  buffers.faceoffsets << " " << buffers.face_offset;

  // Write cell data (one entry per polyhedron, not per vertex)
  buffers.ids << " " << id;
  buffers.types << " " << type;
  buffers.velocities << " " << vx << " " << vy << " " << vz;
  buffers.stress << " " << stress.m11 << " " << stress.m12 << " " << stress.m13 << " " << stress.m21 << " "
                 << stress.m22 << " " << stress.m23 << " " << stress.m31 << " " << stress.m32 << " " << stress.m33;
  if constexpr (has_field_cluster) {
    buffers.clusters << " " << cluster;
  }
  if (buffers.mpi_rank) {
    buffers.ranks << " " << mpi_rank;
  }

  buffers.n_cells += 1;
}

inline void write_vtu_polyhedron(const std::string& name, par_vtu_poly_helper& buffers) {
  std::ofstream outFile(name);
  if (!outFile) {
    color_log::error("write_vtu_polyhedron", "Impossible to open the file: " + name, false);
    return;
  }

  outFile << "<?xml version=\"1.0\"?>" << std::endl;
  outFile << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
  outFile << "  <UnstructuredGrid>" << std::endl;
  outFile << "    <Piece NumberOfPoints=\"" << buffers.n_vertices << "\" NumberOfCells=\"" << buffers.n_cells << "\">"
          << std::endl;
  outFile << "      <PointData>" << std::endl;
  outFile << "      </PointData>" << std::endl;
  outFile << "      <Points>" << std::endl;
  outFile << "        <DataArray type=\"Float64\" Name=\"\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
  outFile << buffers.vertices.rdbuf() << std::endl;
  outFile << "        </DataArray>" << std::endl;
  outFile << "      </Points>" << std::endl;
  outFile << "      <CellData>" << std::endl;
  outFile << "        <DataArray type=\"Int64\" Name=\"Id\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
  outFile << buffers.ids.rdbuf() << std::endl;
  outFile << "        </DataArray>" << std::endl;
  outFile << "        <DataArray type=\"Int32\" Name=\"Type\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
  outFile << buffers.types.rdbuf() << std::endl;
  outFile << "        </DataArray>" << std::endl;
  outFile << "        <DataArray type=\"Float64\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"ascii\">"
          << std::endl;
  outFile << buffers.velocities.rdbuf() << std::endl;
  outFile << "        </DataArray>" << std::endl;
  outFile << "        <DataArray type=\"Float64\" Name=\"Stress\" NumberOfComponents=\"9\" format=\"ascii\">"
          << std::endl;
  outFile << buffers.stress.rdbuf() << std::endl;
  outFile << "        </DataArray>" << std::endl;
  if (buffers.has_cluster) {
    outFile << "        <DataArray type=\"Int32\" Name=\"Cluster\" NumberOfComponents=\"1\" format=\"ascii\">"
            << std::endl;
    outFile << buffers.clusters.rdbuf() << std::endl;
    outFile << "        </DataArray>" << std::endl;
  }
  if (buffers.mpi_rank) {
    outFile << "        <DataArray type=\"Int32\" Name=\"MPI rank\" NumberOfComponents=\"1\" format=\"ascii\">"
            << std::endl;
    outFile << buffers.ranks.rdbuf() << std::endl;
    outFile << "        </DataArray>" << std::endl;
  }
  outFile << "      </CellData>" << std::endl;
  outFile << "      <Cells>" << std::endl;
  outFile << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << std::endl;
  outFile << buffers.connectivity.rdbuf() << std::endl;
  outFile << "        </DataArray>" << std::endl;
  outFile << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << std::endl;
  outFile << buffers.offsets.rdbuf() << std::endl;
  outFile << "        </DataArray>" << std::endl;
  outFile << "        <DataArray type=\"Int32\" Name=\"faces\" format=\"ascii\">" << std::endl;
  outFile << buffers.faces.rdbuf() << std::endl;
  outFile << "        </DataArray>" << std::endl;
  outFile << "        <DataArray type=\"Int32\" Name=\"faceoffsets\" format=\"ascii\">" << std::endl;
  outFile << buffers.faceoffsets.rdbuf() << std::endl;
  outFile << "        </DataArray>" << std::endl;
  outFile << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">" << std::endl;
  for (int i = 0; i < buffers.n_cells; ++i) {
    outFile << " 42";
  }
  outFile << std::endl;
  outFile << "        </DataArray>" << std::endl;
  outFile << "      </Cells>" << std::endl;
  outFile << "    </Piece>" << std::endl;
  outFile << "  </UnstructuredGrid>" << std::endl;
  outFile << "</VTKFile>" << std::endl;
}

inline void write_pvtu_polyhedron(const std::string& filename, size_t number_of_files, bool has_cluster,
                                  bool mpi_rank) {
  const std::string name = filename + ".pvtu";
  std::ofstream outFile(name);
  if (!outFile) {
    color_log::error("write_pvtu_polyhedron", "Impossible to open the file: " + name, false);
    return;
  }

  outFile << "<?xml version=\"1.0\"?>" << std::endl;
  outFile << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
  outFile << "  <PUnstructuredGrid>" << std::endl;
  outFile << "    <PPointData>" << std::endl;
  outFile << "    </PPointData>" << std::endl;
  outFile << "    <PPoints>" << std::endl;
  outFile << "      <PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>" << std::endl;
  outFile << "    </PPoints>" << std::endl;
  outFile << "    <PCellData>" << std::endl;
  outFile << "      <PDataArray type=\"Int64\" Name=\"Id\" NumberOfComponents=\"1\"/>" << std::endl;
  outFile << "      <PDataArray type=\"Int32\" Name=\"Type\" NumberOfComponents=\"1\"/>" << std::endl;
  outFile << "      <PDataArray type=\"Float64\" Name=\"Velocity\" NumberOfComponents=\"3\"/>" << std::endl;
  outFile << "      <PDataArray type=\"Float64\" Name=\"Stress\" NumberOfComponents=\"9\"/>" << std::endl;
  if (has_cluster) {
    outFile << "      <PDataArray type=\"Int32\" Name=\"Cluster\" NumberOfComponents=\"1\"/>" << std::endl;
  }
  if (mpi_rank) {
    outFile << "      <PDataArray type=\"Int32\" Name=\"MPI rank\" NumberOfComponents=\"1\"/>" << std::endl;
  }
  outFile << "    </PCellData>" << std::endl;
  outFile << "    <PCells>" << std::endl;
  outFile << "      <PDataArray type=\"Int32\" Name=\"connectivity\" NumberOfComponents=\"1\"/>" << std::endl;
  outFile << "      <PDataArray type=\"Int32\" Name=\"offsets\" NumberOfComponents=\"1\"/>" << std::endl;
  outFile << "      <PDataArray type=\"Int32\" Name=\"faces\" NumberOfComponents=\"1\"/>" << std::endl;
  outFile << "      <PDataArray type=\"Int32\" Name=\"faceoffsets\" NumberOfComponents=\"1\"/>" << std::endl;
  outFile << "      <PDataArray type=\"UInt8\" Name=\"types\" NumberOfComponents=\"1\"/>" << std::endl;
  outFile << "    </PCells>" << std::endl;

  std::filesystem::path full_path(filename);
  std::string directory = full_path.filename().string();
  std::string subfile = directory + "/%06d.vtu";
  for (size_t i = 0; i < number_of_files; i++) {
    std::string file = onika::format_string(subfile, i);
    outFile << "    <Piece Source=\"" << file << "\"/>" << std::endl;
  }
  outFile << "  </PUnstructuredGrid>" << std::endl;
  outFile << "</VTKFile>" << std::endl;
}

template <class GridT, class = AssertGridHasFields<GridT>>
class WriteParaviewPolyhedraPVTU2Operator : public OperatorNode {
  using ComputeFields = FieldSet<field::_rx, field::_ry, field::_rz, field::_type, field::_orient>;
  static constexpr ComputeFields compute_field_set{};
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(Domain, domain, INPUT, REQUIRED);
  ADD_SLOT(std::string, filename, INPUT, "output");
  ADD_SLOT(long, timestep, INPUT, REQUIRED, DocString{"Iteration number"});
  ADD_SLOT(shapes, shapes_collection, INPUT, REQUIRED, DocString{"Collection of shapes"});
  ADD_SLOT(bool, mpi_rank, INPUT, false, DocString{"Add a field containing the mpi rank."});

 public:
  inline std::string documentation() const final {
    return R"EOF( 
      This operator writes parallel VTK unstructured grid files (.pvtu/.vtu) for polyhedral shapes.

      YAML example:

        - write_paraview_polyhedra_pvtu2:
           filename: "output_directory"
           mpi_rank: true
                )EOF";
  }

  inline void execute() final {
    using ParticleTuple = decltype(grid->cells()[0][0]);
    static constexpr bool has_field_cluster = ParticleTuple::has_field(field::cluster);

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
    par_vtu_poly_helper buffers;
    buffers.mpi_rank = *mpi_rank;

    bool defbox = !domain->xform_is_identity();
    LinearXForm xform;
    if (defbox) {
      xform.m_matrix = domain->xform();
    }

    uint32_t* cluster = nullptr;
    uint32_t cj = 0;

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
      auto* __restrict__ stress = cell[field::stress];
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
        build_buffer_vtu_polyhedron<has_field_cluster>(pos, shp, orient[j], id[j], type[j], vx[j], vy[j], vz[j], h[j],
                                                       stress[j], cj, rank, buffers);
      }
    }

    if (rank == 0) {
      write_pvtu_polyhedron(*filename, size, buffers.has_cluster, buffers.mpi_rank);
    }

    std::string file = *filename + "/%06d.vtu";
    file = onika::format_string(file, rank);
    write_vtu_polyhedron(file, buffers);
  }
};

ONIKA_AUTORUN_INIT(write_paraview_polyhedra_pvtu2) {
  OperatorNodeFactory::instance()->register_factory("write_paraview_polyhedra_pvtu2",
                                                    make_grid_variant_operator<WriteParaviewPolyhedraPVTU2Operator>);
}
}  // namespace exaDEM
