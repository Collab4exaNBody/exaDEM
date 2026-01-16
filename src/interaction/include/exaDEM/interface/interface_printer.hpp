#pragma once

#include <filesystem>

namespace exaDEM {
struct paraview_interface_helper {
  bool mpi_rank;

  int n_vertices = 0;
  int n_polygons = 0;
  std::stringstream vertices;
  std::stringstream offsets;
  std::stringstream ranks;
  std::stringstream ids;
  //  std::stringstream sub_id;
  std::stringstream connectivities;
  std::stringstream fracturation;
  std::stringstream en;
  std::stringstream et;
};

inline void write_vtp_interface(std::string name, paraview_interface_helper& buffers) {
  std::ofstream outFile(name);
  if (!outFile) {
    color_log::error("write_vtp_polyhedron", "Impossible to open the file: " + name, false);
    return;
  }

  outFile << "<?xml version=\"1.0\"?>" << std::endl;
  outFile << "<VTKFile type=\"PolyData\">" << std::endl;
  outFile << "  <PolyData>" << std::endl;
  outFile << "    <Piece NumberOfPoints=\"" << buffers.n_vertices << "\" NumberOfLines=\"" << 0 << "\" NumberOfPolys=\""
          << buffers.n_polygons << "\">" << std::endl;
  outFile << "    <PointData>" << std::endl;

  outFile << "      <DataArray type=\"Int64\" Name=\"Id\"  NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
  outFile << buffers.ids.rdbuf() << std::endl;
  outFile << "      </DataArray>" << std::endl;

  if (buffers.mpi_rank) {  // MPI  rank - optional
    outFile << "      <DataArray type=\"Int32\" Name=\"MPI rank\"  NumberOfComponents=\"1\" format=\"ascii\">"
            << std::endl;
    outFile << buffers.ranks.rdbuf() << std::endl;
    outFile << "      </DataArray>" << std::endl;
  }

  outFile << "      <DataArray type=\"Float64\" Name=\"Fracturation rate\"  NumberOfComponents=\"1\" format=\"ascii\">"
          << std::endl;
  outFile << buffers.fracturation.rdbuf() << std::endl;
  outFile << "      </DataArray>" << std::endl;

  outFile << "      <DataArray type=\"Float64\" Name=\"En\"  NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
  outFile << buffers.en.rdbuf() << std::endl;
  outFile << "      </DataArray>" << std::endl;

  outFile << "      <DataArray type=\"Float64\" Name=\"Et\"  NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
  outFile << buffers.et.rdbuf() << std::endl;
  outFile << "      </DataArray>" << std::endl;

  outFile << "    </PointData>" << std::endl;
  outFile << "    <Points>" << std::endl;
  outFile << "      <DataArray type=\"Float64\" Name=\"\"  NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
  outFile << buffers.vertices.rdbuf() << std::endl;
  outFile << "      </DataArray>" << std::endl;
  outFile << "    </Points>" << std::endl;
  outFile << "    <Polys>" << std::endl;
  outFile << "      <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << std::endl;
  outFile << buffers.connectivities.rdbuf() << std::endl;
  outFile << "      </DataArray>" << std::endl;
  outFile << "      <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << std::endl;
  outFile << buffers.offsets.rdbuf() << std::endl;
  outFile << "      </DataArray>" << std::endl;
  outFile << "    </Polys>" << std::endl;
  outFile << "    </Piece>" << std::endl;
  outFile << "  </PolyData>" << std::endl;
  outFile << "</VTKFile>" << std::endl;
}

inline void write_pvtp_interface(std::string filename, size_t number_of_files, paraview_interface_helper& buffers) {
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
  if (buffers.mpi_rank) {
    outFile << "      <PDataArray type=\"Int32\" Name=\"MPI rank\"  NumberOfComponents=\"1\"/>" << std::endl;
  }
  outFile << "      <PDataArray type=\"Float64\" Name=\"Fracturation rate\"  NumberOfComponents=\"1\"/>" << std::endl;
  outFile << "      <PDataArray type=\"Float64\" Name=\"En\"  NumberOfComponents=\"1\"/>" << std::endl;
  outFile << "      <PDataArray type=\"Float64\" Name=\"Et\"  NumberOfComponents=\"1\"/>" << std::endl;
  outFile << "    </PPointData>" << std::endl;
  outFile << "    <PPoints>" << std::endl;
  outFile << "      <PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>" << std::endl;
  outFile << "    </PPoints> " << std::endl;
  outFile << "    <PLines>" << std::endl;
  outFile << "      <PDataArray type=\"Int32\" Name=\"connectivity\"  NumberOfComponents=\"1\"/>" << std::endl;
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
}  // namespace exaDEM
