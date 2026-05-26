#pragma once

#include <filesystem>

namespace exaDEM {
/** @brief Helper struct for creating Paraview-compatible interface files */
struct paraview_interface_helper {
  bool mpi_rank;                     // Store MPI rank if needed
  int n_vertices = 0;                // Number of vertices "in all interfaces"
  int n_polygons = 0;                // Number of polygons "in all interfaces"
  std::stringstream vertices;        // Stream to store vertex coordinates
  std::stringstream offsets;         // Stream to store polygon offsets (number of vertices per polygon)
  std::stringstream ranks;           // Stream to store MPI ranks (optional)
  std::stringstream ids;             // Stream to store unique IDs for each vertex (optional)
  std::stringstream connectivities;  // Stream to store polygon / face connectivity (vertex indices for each polygon)
  std::stringstream fracturation;    // Stream to store fracturation rate of each vertex (optional)
  std::stringstream en;              // Stream to store normal energy of each vertex (optional)
  std::stringstream tds;             // Stream to store tangential displacement of each vertex (optional)
  std::stringstream et;              // Stream to store tangential energy of each vertex (optional)
};

/** @brief Write a VTP file for interfaces
 * @param name The name of the output file (should end with .vtp)
 * @param buffers The helper struct containing the data to write
 */
inline void write_vtp_interface(std::string name, paraview_interface_helper& buffers) {
  std::ofstream outFile(name);
  if (!outFile) {
    color_log::error("write_vtp_interface", "Impossible to open the file: " + name, false);
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

  outFile
      << "      <DataArray type=\"Float64\" Name=\"TangentialDisplacement\"  NumberOfComponents=\"3\" format=\"ascii\">"
      << std::endl;
  outFile << buffers.tds.rdbuf() << std::endl;
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

/** @brief Write a PVTP file for the interface
 * @param filename The name of the output file (should end with .pvtp)
 * @param number_of_files The number of VTP files to include
 * @param buffers The helper struct containing the data to write
 */
inline void write_pvtp_interface(std::string filename, size_t number_of_files, paraview_interface_helper& buffers) {
  std::string name = filename + ".pvtp";
  std::ofstream outFile(name);
  if (!outFile) {
    color_log::error("write_pvtp_interface", "Impossible to open the file: " + name, false);
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
  outFile << "      <PDataArray type=\"Float64\" Name=\"TangentialDisplacement\"  NumberOfComponents=\"3\"/>"
          << std::endl;
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

/** @brief Orders vertices to form a non-self-intersecting face.
 * @param vertices The vector of vertices to order.
 * Note: This assumes the vertices are roughly coplanar.
 * This function is used to order the vertices of the faces of the interfaces before writing them in VTP files, to
 * ensure that the faces are correctly displayed in Paraview.
 */
void order_face_vertices(std::vector<Vec3d>& vertices) {
  if (vertices.size() < 3) {
    return;
  }

  //  Calculate the Centroid
  Vec3d centroid = {0, 0, 0};
  for (const auto& v : vertices) {
    centroid.x += v.x;
    centroid.y += v.y;
    centroid.z += v.z;
  }
  centroid.x /= vertices.size();
  centroid.y /= vertices.size();
  centroid.z /= vertices.size();

  Vec3d a = vertices[0] - centroid;
  Vec3d normal = {0, 0, 1};  // Default value to prevent faillure

  Vec3d b = vertices[1] - centroid;
  normal = exanb::cross(a, b);
  normal = normal / exanb::norm(normal);

  Vec3d u = (vertices[0] - centroid);
  u = u / exanb::norm(u);  // Normalize u
  Vec3d v = exanb::cross(normal, u);

  // Sort by polar angle around the centroid
  std::sort(vertices.begin(), vertices.end(), [centroid, u, v](const Vec3d& a, const Vec3d& b) {
    Vec3d da = a - centroid;
    Vec3d db = b - centroid;

    // Project on local plan to get 2D coordonates
    double xA = exanb::dot(da, u);
    double yA = exanb::dot(da, v);

    double xB = exanb::dot(db, u);
    double yB = exanb::dot(db, v);

    double angleA = std::atan2(yA, xA);
    double angleB = std::atan2(yB, xB);

    // Handle numerical error
    if (std::abs(angleA - angleB) > 1e-9) {
      return angleA < angleB;
    }

    // Tie-breaker
    return exanb::dot(da, da) < exanb::dot(db, db);
  });
}
}  // namespace exaDEM
