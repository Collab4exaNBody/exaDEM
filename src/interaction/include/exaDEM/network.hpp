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

#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/shape/shapes.hpp>
#include <exaDEM/shape/shape_detection.hpp>
#include <exaDEM/contact_force_parameters.h>
#include <exaDEM/compute_contact_force.h>
#include <vector>
#include <tuple>
#include <algorithm>

namespace exaDEM
{
  /**
   * @brief Functor for contact network operations on a grid.
   *
   * It can be used to perform various network-related operations on the grid,
   * including displaying the contact network between polyhedra.
   *
   * @tparam GridT The type of the grid on which network operations will be performed.
   */
  template <typename GridT> struct NetworkFunctor
  {
    using IdType = std::pair<size_t, size_t>;     ///< Type for identifying particles.
    using ForceType = double;                     ///< Type for representing forces (could be a vec3d).
    using CoupleType = std::pair<IdType, IdType>; ///< Type for identifying couples of particles.
    using StorageType = ForceType;                ///< Type for storing forces.
    using KeyType = CoupleType;                   ///< Type used as keys in data storage.

    // Members
    typename GridT::CellParticles *cells; ///< Pointer to the cells of the grid.
    shapes &shps;                         ///< Reference to the list of shapes.
    ContactParams params;                 ///< Parameters for Contact operations.
    double time;                          ///< Incrementation time value

    using signature = std::tuple<bool, double, exanb::Vec3d, exanb::Vec3d> (*)(const onika::oarray_t<exanb::Vec3d, EXADEM_MAX_VERTICES> &, int, const exaDEM::shape *, const onika::oarray_t<exanb::Vec3d, EXADEM_MAX_VERTICES> &, int, const exaDEM::shape *);

    /**
     * @brief Array of function pointers for precomputed detection signatures.
     *
     * This static constexpr array contains function pointers for precomputed detection.
     * Each element of the array corresponds to a specific type of interaction detection:
     * - detection[0]: Vertex-Vertex interaction detection.
     * - detection[1]: Vertex-Edge interaction detection.
     * - detection[2]: Vertex-Face interaction detection.
     * - detection[3]: Edge-Edge interaction detection.
     */
    static constexpr signature detection[4] = {exaDEM::detection_vertex_vertex_precompute, exaDEM::detection_vertex_edge_precompute, exaDEM::detection_vertex_face_precompute, exaDEM::detection_edge_edge_precompute};

    // TODO optimize if later with another data storage
    std::map<KeyType, StorageType> network; ///< Stores network data, with keys defined by KeyType and values by StorageType.

    /**
     * @brief Constructor for NetworkFunctor.
     *
     * Initializes a NetworkFunctor object with the provided grid, shapes, configuration parameters, and time.
     *
     * @param g Reference to the grid.
     * @param s Reference to the shapes.
     * @param config Reference to the configuration parameters.
     * @param t Time parameter.
     */
    NetworkFunctor(GridT &g, shapes &s, ContactParams &config, double t) : cells(g.cells()), shps(s), params(config), time(t) {}

    /**
     * @brief Gets the position vector of a particle in a specified cell.
     *
     * Retrieves the position vector of the particle with the position in the specified cell.
     *
     * @param cell_id The ID of the cell containing the particle.
     * @param p_id The position of the particle in the cell_id.
     * @return The position vector of the particle.
     */
    const Vec3d get_position(const int cell_id, const int p_id)
    {
      const Vec3d res = {cells[cell_id][field::rx][p_id], cells[cell_id][field::ry][p_id], cells[cell_id][field::rz][p_id]};
      return res;
    };

    /**
     * @brief Gets the velocity vector of a particle in a specified cell.
     *
     * Retrieves the velocity vector of the particle with the position in the specified cell.
     *
     * @param cell_id The ID of the cell containing the particle.
     * @param p_id The position of the particle in the cell_id.
     * @return The velocity vector of the particle.
     */
    const Vec3d get_velocity(const int cell_id, const int p_id)
    {
      const Vec3d res = {cells[cell_id][field::vx][p_id], cells[cell_id][field::vy][p_id], cells[cell_id][field::vz][p_id]};
      return res;
    };

    /**
     * @brief Computes the normal force value according to contact law and stores it based on the pair of polyhedra considered.
     * @param I The Interaction object representing the pair of polyhedra and associated interaction data.
     * @return A key-value pair representing the processed interaction data (pair of polyhedra and normal force value).
     */
    std::pair<KeyType, StorageType> operator()(exaDEM::Interaction &I)
    {
      // === build contact network key
      IdType i = {I.cell_i, I.p_i};
      IdType j = {I.cell_j, I.p_j};
      KeyType key = {i, j};

      // === only interactions between polyhedron are considered
      if (I.type > 3)
      {
        return {key, ForceType(0)};
      }

      // === positions
      const Vec3d ri = get_position(I.cell_i, I.p_i);
      const Vec3d rj = get_position(I.cell_j, I.p_j);

      // === get cells
      auto &cell_i = cells[I.cell_i];
      auto &cell_j = cells[I.cell_j];

      // === get angular velocities
      const Vec3d &vrot_i = cell_i[field::vrot][I.p_i];
      const Vec3d &vrot_j = cell_j[field::vrot][I.p_j];

      // === get polyhedron types
      const auto &type_i = cell_i[field::type][I.p_i];
      const auto &type_j = cell_j[field::type][I.p_j];

      // === get vertex positions
      const auto &vertices_i = cell_i[field::vertices][I.p_i];
      const auto &vertices_j = cell_j[field::vertices][I.p_j];

      // === get shape relative to polyhedron types
      const shape *shp_i = this->shps[type_i];
      const shape *shp_j = this->shps[type_j];

      // === make detection
      auto [contact, dn, n, contact_position] = detection[I.type](vertices_i, I.sub_i, shp_i, vertices_j, I.sub_j, shp_j);

      // if contact detection, compute contact law
      if (contact)
      {
        const Vec3d vi = get_velocity(I.cell_i, I.p_i);
        const Vec3d vj = get_velocity(I.cell_j, I.p_j);
        const auto &m_i = cell_i[field::mass][I.p_i];
        const auto &m_j = cell_j[field::mass][I.p_j];

        // === Utilize temporary values to avoid updating friction and moment in contact_force_core.
        Vec3d f = {0, 0, 0};
        Vec3d fr = I.friction;
        Vec3d mom = I.moment;
        const double meff = compute_effective_mass(m_i, m_j);

        contact_force_core(dn, n, time, params.m_kn, params.m_kt, params.m_kr, params.m_mu, params.m_damp_rate, meff, fr, contact_position, ri, vi, f, mom, vrot_i, // particle 1
                           rj, vj, vrot_j                                                                                                                           // particle nbh
        );
        // === compute normal force vector (f = ft + fn)
        Vec3d fn = f - fr;
        // === compute normal force
        ForceType res = exanb::norm(fn);
        return {key, res};
      }
      else
      {
        return {key, ForceType(0)};
      }
    }

    /**
     * @brief Functor operator for processing multiple contact interactions (network). Every Interactions concerns the same particle.
     * @param I Pointer to the array of Interaction objects.
     * @param offset Offset indicating the starting index of the interactions to process.
     * @param size Size of the range of interactions to process.
     */
    void operator()(exaDEM::Interaction *I, const size_t offset, const size_t size)
    {
      // interaction are sorted by construction
      // sort is used because interaction i to j and j to i have the same key.
      auto sort = [](CoupleType &in) -> void
      {
        if (in.first > in.second)
          std::swap(in.first, in.second);
      };

      for (size_t i = offset; i < offset + size; i++)
      {
        auto [couple, value] = this->operator()(I[i]);

        // value = 0 means that the contact detection has failled
        if (value == 0)
          continue;
        if (value == 0)
          continue;

        // update key
        sort(couple);

        // check if this contact exists, increment the value if true, otherwise it creates it
        auto it = network.find(couple);
        if (it != network.end())
        {
          it->second += value;
        }
        else
        {
          network[couple] = value;
        }
      }
    }

    /**
     * @brief Creates an indirection array for particle indexes (cell_id, pos_id).
     *
     * This function creates and returns an indirection array as a vector of IdType. Not that id are unique.
     *
     * @return A vector of IdType representing the created indirection array.
     */
    std::vector<IdType> create_indirection_array()
    {
      const size_t size = network.size();
      std::vector<IdType> ids(size * 2); // 2 Id per elem
      size_t offset = 0;

      // Get all particles
      for (auto it : network)
      {
        KeyType c = it.first;
        ids[offset++] = c.first;
        ids[offset++] = c.second;
      }

      // Remove doublons
      auto it = unique(ids.begin(), ids.end());
      ids.resize(distance(ids.begin(), it));

      // Sort them to speed up the future lookup processes.
      std::sort(ids.begin(), ids.end());
      return ids;
    }

    /**
     * @brief Fills a stringstream with position data.
     * @param buff_position The stringstream to fill with position data.
     * @param ids The vector of particle IDs for which position data is to be retrieved.
     */
    void fill_position(std::stringstream &buff_position, std::vector<IdType> &ids)
    {
      for (size_t i = 0; i < ids.size(); i++)
      {
        auto [cell, id] = ids[i];
        buff_position << " " << cells[cell][field::rx][id] << " " << cells[cell][field::ry][id] << " " << cells[cell][field::rz][id];
      }
    }

    /**
     * @brief Fills stringstreams with connectivity and value data.
     * @param buff_connect The stringstream to fill with connectivity data.
     * @param buff_value The stringstream to fill with value data.
     * @param ids The vector of particle IDs for which connectivity and value data is to be retrieved.
     */
    void fill_connect_and_value(std::stringstream &buff_connect, std::stringstream &buff_value, std::vector<IdType> &ids)
    {
      for (auto it : network)
      {
        auto [i, j] = it.first;
        auto iti = std::lower_bound(ids.begin(), ids.end(), i);
        auto itj = std::lower_bound(ids.begin(), ids.end(), j);
        assert(iti != ids.end());
        assert(itj != ids.end());
        int ii = std::distance(ids.begin(), iti);
        int jj = std::distance(ids.begin(), itj);
        buff_connect << " " << ii << " " << jj;
        buff_value << " " << it.second;
      }
    }

    /**
     * @brief Writes VTP (VTK PolyData) files.
     *
     * This function writes VTP (VTK PolyData) files using the provided data streams for position, connectivity, and value.
     * It writes the data corresponding to the specified number of particles into the VTP file with the given name.
     *
     * @param name The name of the VTP file to write.
     * @param n_particles The number of particles to write data for.
     * @param buff_position The stringstream containing position data.
     * @param buff_connect The stringstream containing connectivity data.
     * @param buff_value The stringstream containing value data.
     */
    void write_vtp(std::string name, size_t n_particles, std::stringstream &buff_position, std::stringstream &buff_connect, std::stringstream &buff_value)
    {
      size_t n_interactions = network.size();
      name = name + ".vtp";
      std::ofstream outFile(name);
      if (!outFile)
      {
        std::cerr << "Erreur : impossible de créer le fichier de sortie suivant: " << name << std::endl;
        return;
      }

      outFile << "<VTKFile type=\"PolyData\">" << std::endl;
      outFile << " <PolyData>" << std::endl;
      outFile << "   <Piece NumberOfPoints=\"" << n_particles << "\" NumberOfLines=\"" << n_interactions << "\">" << std::endl;
      outFile << "   <Points>" << std::endl;
      outFile << "     <DataArray type=\"Float64\" Name=\"\"  NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
      if (n_particles != 0)
        outFile << buff_position.rdbuf() << std::endl;
      outFile << "     </DataArray>" << std::endl;
      outFile << "   </Points>" << std::endl;
      outFile << "   <Lines>" << std::endl;
      outFile << "     <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << std::endl;
      if (n_interactions != 0)
        outFile << buff_connect.rdbuf() << std::endl;
      outFile << "     </DataArray>" << std::endl;
      outFile << "     <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << std::endl;
      for (size_t i = 0; i < n_interactions; i++)
        outFile << " " << 2 * i;
      outFile << std::endl;
      outFile << "     </DataArray>" << std::endl;
      outFile << "   </Lines>" << std::endl;
      outFile << "   <CellData>" << std::endl;
      outFile << "     <DataArray type=\"Float64\" Name=\"fn\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
      if (n_interactions != 0)
        outFile << buff_value.rdbuf() << std::endl;
      outFile << "     </DataArray>" << std::endl;
      outFile << "   </CellData>" << std::endl;
      outFile << "  </Piece>" << std::endl;
      outFile << " </PolyData>" << std::endl;
      outFile << "</VTKFile>" << std::endl;
    }

    /**
     * @brief Writes PVTP (Parallel VTK PolyData) files.
     *
     * This function writes PVTP (Parallel VTK PolyData) files with the specified base directory, base name, and number of files.
     * It creates a series of PVTP files with the given base name and number of files, each containing metadata for parallel visualization.
     *
     * @param basedir The base directory where the PVTP files will be written.
     * @param basename The base name for the PVTP files.
     * @param number_of_files The number of PVTP files to create.
     */
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
      outFile << "     <PCellData Scalar=\"fn\">" << std::endl;
      outFile << "       <PDataArray type=\"Float64\" Name=\"fn\" NumberOfComponents=\"1\"/>" << std::endl;
      outFile << "     </PCellData> " << std::endl;
      for (size_t i = 0; i < number_of_files; i++)
      {
        std::string subfile = basename + "/" + basename + "_" + std::to_string(i) + ".vtp";
        outFile << "     <Piece Source=\"" << subfile << "\"/>" << std::endl;
      }
      outFile << "   </PPolyData>" << std::endl;
      outFile << " </VTKFile>" << std::endl;
    }
  };
} // namespace exaDEM
