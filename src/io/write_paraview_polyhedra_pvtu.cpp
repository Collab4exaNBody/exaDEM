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
#include <onika/math/quaternion_operators.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>
#include <onika/string_utils.h>

// exaNBody
#include <exanb/core/domain.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/xform.h>

// exaDEM
#include <exaDEM/color_log.hpp>
#include <exaDEM/shapes.hpp>
#include <exaDEM/vtk_data_array.hpp>

// C++ standard library
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

namespace exaDEM {
using namespace exanb;

struct par_vtu_poly_helper {
  int n_vertices = 0;
  int n_cells = 0;
  int64_t connectivity_offset = 0;
  int64_t face_offset = 0;
  std::vector<double> vertices;
  std::vector<int64_t> connectivity;
  std::vector<int64_t> offsets;
  std::vector<int64_t> faces;
  std::vector<int64_t> faceoffsets;

  CellDataField<int64_t> id{"id"};
  CellDataField<int32_t> type{"type"};
  CellDataField<Vec3d> velocity{"velocity"};
  CellDataField<Vec3d> vrot{"vrot"};
  CellDataField<Vec3d> arot{"arot"};
  CellDataField<double> mass{"mass"};
  CellDataField<double> radius{"radius"};
  CellDataField<Vec3d> inertia{"inertia"};
  CellDataField<Mat3d> stress{"stress"};
  CellDataField<int32_t> cluster{"cluster"};
  CellDataField<int32_t> group{"group"};
  CellDataField<int32_t> mpi_rank{"mpi_rank"};
};

inline void build_buffer_vtu_polyhedron(const exanb::Vec3d& pos, const shape* shp, const exanb::Quaternion& orient,
                                        uint64_t id, uint16_t type, double vx, double vy, double vz, double h,
                                        const exanb::Vec3d& vrot, const exanb::Vec3d& arot, double mass, double radius,
                                        const exanb::Vec3d& inertia, const exanb::Mat3d& stress, uint32_t cluster,
                                        uint32_t group, int mpi_rank, par_vtu_poly_helper& buffers) {
  const int vertex_base = buffers.n_vertices;
  const int n_vertices = shp->get_number_of_vertices();

  auto writer_vertex = [&buffers](const exanb::Vec3d& v, const exanb::Vec3d& p, const double& h,
                                  const exanb::Quaternion& q) {
    exanb::Vec3d new_pos = p + q * (h * v);
    append_value(buffers.vertices, new_pos);
  };

  shp->for_all_vertices(writer_vertex, pos, h, orient);

  buffers.n_vertices += n_vertices;

  const int n_faces = shp->get_number_of_faces();
  for (int vid = 0; vid < n_vertices; vid++) {
    buffers.connectivity.push_back(vertex_base + vid);
  }
  buffers.connectivity_offset += n_vertices;
  buffers.offsets.push_back(buffers.connectivity_offset);

  buffers.faces.push_back(n_faces);
  buffers.face_offset += 1;

  for (int face_idx = 0; face_idx < n_faces; face_idx++) {
    auto [face_vertices, face_size] = shp->get_face(face_idx);
    buffers.face_offset += face_size + 1;

    buffers.faces.push_back(face_size);
    for (int it = 0; it < face_size; it++) {
      buffers.faces.push_back(vertex_base + face_vertices[it]);
    }
  }
  buffers.faceoffsets.push_back(buffers.face_offset);

  // Cell data (one entry per polyhedron, not per vertex). Each append() is a no-op
  // unless the corresponding field was selected via the "fields" slot.
  buffers.id.append(static_cast<int64_t>(id));
  buffers.type.append(static_cast<int32_t>(type));
  buffers.velocity.append(exanb::Vec3d{vx, vy, vz});
  buffers.vrot.append(vrot);
  buffers.arot.append(arot);
  buffers.mass.append(mass);
  buffers.radius.append(radius);
  buffers.inertia.append(inertia);
  buffers.stress.append(stress);
  buffers.cluster.append(static_cast<int32_t>(cluster));
  buffers.group.append(static_cast<int32_t>(group));
  buffers.mpi_rank.append(static_cast<int32_t>(mpi_rank));

  buffers.n_cells += 1;
}

inline void write_vtu_polyhedron(const std::string& name, par_vtu_poly_helper& buffers, const vtk_output_mode& mode) {
  std::ofstream outFile(name);
  if (!outFile) {
    color_log::error("write_vtu_polyhedron", "Impossible to open the file: " + name, false);
    return;
  }

  outFile << "<?xml version=\"1.0\"?>" << std::endl;
  outFile << "<VTKFile type=\"UnstructuredGrid\" " << vtk_file_attributes(mode) << ">" << std::endl;
  outFile << "  <UnstructuredGrid>" << std::endl;
  outFile << "    <Piece NumberOfPoints=\"" << buffers.n_vertices << "\" NumberOfCells=\"" << buffers.n_cells << "\">"
          << std::endl;
  outFile << "      <PointData>" << std::endl;
  outFile << "      </PointData>" << std::endl;
  outFile << "      <Points>" << std::endl;
  write_vtk_data_array(outFile, "Float64", "", 3, buffers.vertices, mode);
  outFile << "      </Points>" << std::endl;
  outFile << "      <CellData>" << std::endl;
  write_cell_data_array(outFile, buffers.id, mode);
  write_cell_data_array(outFile, buffers.type, mode);
  write_cell_data_array(outFile, buffers.velocity, mode);
  write_cell_data_array(outFile, buffers.vrot, mode);
  write_cell_data_array(outFile, buffers.arot, mode);
  write_cell_data_array(outFile, buffers.mass, mode);
  write_cell_data_array(outFile, buffers.radius, mode);
  write_cell_data_array(outFile, buffers.inertia, mode);
  write_cell_data_array(outFile, buffers.stress, mode);
  write_cell_data_array(outFile, buffers.cluster, mode);
  write_cell_data_array(outFile, buffers.group, mode);
  write_cell_data_array(outFile, buffers.mpi_rank, mode);
  outFile << "      </CellData>" << std::endl;
  outFile << "      <Cells>" << std::endl;
  write_vtk_data_array(outFile, "Int64", "connectivity", 0, buffers.connectivity, mode);
  write_vtk_data_array(outFile, "Int64", "offsets", 0, buffers.offsets, mode);
  write_vtk_data_array(outFile, "Int64", "faces", 0, buffers.faces, mode);
  write_vtk_data_array(outFile, "Int64", "faceoffsets", 0, buffers.faceoffsets, mode);
  const std::vector<uint8_t> types(buffers.n_cells, 42);  // VTK_POLYHEDRON
  write_vtk_data_array(outFile, "UInt8", "types", 0, types, mode);
  outFile << "      </Cells>" << std::endl;
  outFile << "    </Piece>" << std::endl;
  outFile << "  </UnstructuredGrid>" << std::endl;
  outFile << "</VTKFile>" << std::endl;
}

inline void write_pvtu_polyhedron(const std::string& filename, size_t number_of_files,
                                  const par_vtu_poly_helper& buffers) {
  const std::string name = filename + ".pvtu";
  std::ofstream outFile(name);
  if (!outFile) {
    color_log::error("write_pvtu_polyhedron", "Impossible to open the file: " + name, false);
    return;
  }

  outFile << "<?xml version=\"1.0\"?>" << std::endl;
  outFile << "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">" << std::endl;
  outFile << "  <PUnstructuredGrid>" << std::endl;
  outFile << "    <PPointData>" << std::endl;
  outFile << "    </PPointData>" << std::endl;
  outFile << "    <PPoints>" << std::endl;
  outFile << "      <PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>" << std::endl;
  outFile << "    </PPoints>" << std::endl;
  outFile << "    <PCellData>" << std::endl;
  write_pcell_data_array(outFile, buffers.id);
  write_pcell_data_array(outFile, buffers.type);
  write_pcell_data_array(outFile, buffers.velocity);
  write_pcell_data_array(outFile, buffers.vrot);
  write_pcell_data_array(outFile, buffers.arot);
  write_pcell_data_array(outFile, buffers.mass);
  write_pcell_data_array(outFile, buffers.radius);
  write_pcell_data_array(outFile, buffers.inertia);
  write_pcell_data_array(outFile, buffers.stress);
  write_pcell_data_array(outFile, buffers.cluster);
  write_pcell_data_array(outFile, buffers.group);
  write_pcell_data_array(outFile, buffers.mpi_rank);
  outFile << "    </PCellData>" << std::endl;
  outFile << "    <PCells>" << std::endl;
  outFile << "      <PDataArray type=\"Int64\" Name=\"connectivity\" NumberOfComponents=\"1\"/>" << std::endl;
  outFile << "      <PDataArray type=\"Int64\" Name=\"offsets\" NumberOfComponents=\"1\"/>" << std::endl;
  outFile << "      <PDataArray type=\"Int64\" Name=\"faces\" NumberOfComponents=\"1\"/>" << std::endl;
  outFile << "      <PDataArray type=\"Int64\" Name=\"faceoffsets\" NumberOfComponents=\"1\"/>" << std::endl;
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
  using StringList = std::vector<std::string>;
  static constexpr ComputeFields compute_field_set{};
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(StringList, fields, INPUT, StringList({".*"}),
           DocString{"List of regular expressions to select fields to project"});
  ADD_SLOT(Domain, domain, INPUT, REQUIRED);
  ADD_SLOT(std::string, filename, INPUT, "output");
  ADD_SLOT(long, timestep, INPUT, REQUIRED, DocString{"Iteration number"});
  ADD_SLOT(shapes, shapes_collection, INPUT, REQUIRED, DocString{"Collection of shapes"});
  ADD_SLOT(bool, mpi_rank, INPUT, false, DocString{"Add a field containing the mpi rank."});
  ADD_SLOT(bool, binary_mode, INPUT, true,
           DocString{"Write data arrays as zlib-compressed binary data (true) or as ascii (false)."});
  ADD_SLOT(std::string, compression, INPUT, "default",
           DocString{"Compression level used in binary mode: default (medium, zlib level 6), min, max or [0-9]."});

 public:
  inline std::string documentation() const final {
    return R"EOF(
      This operator writes parallel VTK unstructured grid files (.pvtu/.vtu) for polyhedral shapes.

      The "fields" slot is a list of regular expressions selecting which cell data arrays
      are projected. Available fields are: "id", "type", "velocity", "vrot",
      "arot", "mass", "radius", "inertia", "stress", "group", "cluster"
      (if the cluster field is present) and "mpi_rank" (if mpi_rank is enabled).
      By default (".*"), every available field is written, and the .pvtu file
      only declares the fields that are actually selected.

      By default, data arrays are written in binary mode: they are zlib-compressed and
      base64-encoded with a medium compression level ("default", i.e. zlib level 6).
      The "compression" slot changes this level: "min" (0, no compression), "max" (9)
      or any digit in [0-9]. To write human-readable ascii files instead, set
      "binary_mode: false" (the "compression" slot is then ignored).

      YAML example (binary, default compression):

        - write_paraview_polyhedra_pvtu:
           filename: "output_directory"
           mpi_rank: true
           fields: ["id", "type", "velocity"]

      YAML example (ascii):

        - write_paraview_polyhedra_pvtu:
           filename: "output_directory"
           binary_mode: false
                )EOF";
  }

  inline void execute() final {
    using ParticleTuple = decltype(grid->cells()[0][0]);
    static constexpr bool has_field_cluster = ParticleTuple::has_field(field::cluster);

    int rank, size;
    MPI_Comm_rank(*mpi, &rank);
    MPI_Comm_size(*mpi, &size);

    // field selector function
    const auto& flist = *fields;
    auto field_selector = [&flist](const std::string& name) -> bool {
      for (const auto& f : flist)
        if (std::regex_match(name, std::regex(f))) return true;
      return false;
    };

    if (rank == 0) {
      std::filesystem::create_directories(*filename);
    }
    MPI_Barrier(*mpi);

    auto& shps = *shapes_collection;
    const auto cells = grid->cells();
    const size_t n_cells = grid->number_of_cells();

    par_vtu_poly_helper buffers;
    buffers.id.write = field_selector("id");
    buffers.type.write = field_selector("type");
    buffers.velocity.write = field_selector("velocity");
    buffers.vrot.write = field_selector("vrot");
    buffers.arot.write = field_selector("arot");
    buffers.mass.write = field_selector("mass");
    buffers.radius.write = field_selector("radius");
    buffers.inertia.write = field_selector("inertia");
    buffers.stress.write = field_selector("stress");
    buffers.cluster.write = has_field_cluster && field_selector("cluster");
    buffers.group.write = field_selector("group");
    buffers.mpi_rank.write = *mpi_rank && field_selector("mpi_rank");

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
      auto* __restrict__ vrot = cell[field::vrot];
      auto* __restrict__ arot = cell[field::arot];
      auto* __restrict__ mass = cell[field::mass];
      auto* __restrict__ radius = cell[field::radius];
      auto* __restrict__ inertia = cell[field::inertia];
      auto* __restrict__ stress = cell[field::stress];
      auto* __restrict__ group = cell[field::group];
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
        build_buffer_vtu_polyhedron(pos, shp, orient[j], id[j], type[j], vx[j], vy[j], vz[j], h[j], vrot[j], arot[j],
                                    mass[j], radius[j], inertia[j], stress[j], cj, group[j], rank, buffers);
      }
    }

    if (rank == 0) {
      write_pvtu_polyhedron(*filename, size, buffers);
    }

    std::string file = *filename + "/%06d.vtu";
    file = onika::format_string(file, rank);
    vtk_output_mode mode;
    mode.binary = *binary_mode;
    mode.compression_level = vtk_compression_level(*compression);
    write_vtu_polyhedron(file, buffers, mode);
  }
};

ONIKA_AUTORUN_INIT(write_paraview_polyhedra_pvtu2) {
  OperatorNodeFactory::instance()->register_factory("write_paraview_polyhedra_pvtu",
                                                    make_grid_variant_operator<WriteParaviewPolyhedraPVTU2Operator>);
  // alias
  OperatorNodeFactory::instance()->register_factory("write_paraview_polyhedra",
                                                    make_grid_variant_operator<WriteParaviewPolyhedraPVTU2Operator>);
}
}  // namespace exaDEM
