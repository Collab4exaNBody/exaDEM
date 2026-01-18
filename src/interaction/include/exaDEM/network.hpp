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
#include <exaDEM/shapes.hpp>
#include <exaDEM/shape_detection.hpp>
#include <exaDEM/type/contact.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

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

    GridT& grid; ///< Pointer to the cells of the grid.
    std::stringstream pos; // store particle positions
    std::stringstream connect; // store interaction connections
    std::stringstream val; // store force values

    // TODO optimize if later with another data storage
    std::map<KeyType, StorageType> network; ///< Stores network data, with keys defined by KeyType and values by StorageType.

    NetworkFunctor(GridT &g) : grid(g) {}

    template<typename InteractionT>
    void add(InteractionT& I, double value)
    {
      auto& pi = I.i();
      auto& pj = I.j();
      // === build contact network key
      IdType i = {pi.cell, pi.p};
      IdType j = {pj.cell, pj.p};
      KeyType key = {i, j};
      auto it = network.find(key);
      if (it != network.end())
      {
        it->second += value;
      }
      else
      {
        network[key] = value;
      }
    }

    template<typename Is, typename Data> 
      void operator()(const size_t size, Is& interactions, Data& data)
      {
        Vec3d* fn = onika::cuda::vector_data(data.fn); 
        Vec3d* ft = onika::cuda::vector_data(data.ft);
        for(size_t i = 0; i < size ; i++)
        {
          const double f = exanb::norm(fn[i] + ft[i]);
          if( f != 0)
          {
            auto I = interactions[i];
            if (filter_duplicates(I)) add(I, f);
          }
        } 
      }

    /**
     * @brief Creates an indirection array for particle indexes (cell_id, pos_id).
     *
     */
    void fill_fn_at_point_data()
    {
      auto * cells = grid.cells();
      for (auto it : network)
      {
        auto& [i, j] = it.first;
        auto& [cell_i, p_i] = i;
        auto& [cell_j, p_j] = j;
        pos << " " << cells[cell_i][field::rx][p_i] << " " << cells[cell_i][field::ry][p_i] << " " << cells[cell_i][field::rz][p_i];
        pos << " " << cells[cell_j][field::rx][p_j] << " " << cells[cell_j][field::ry][p_j] << " " << cells[cell_j][field::rz][p_j];
        val << " " << it.second << " " << it.second;
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
     */
    void write_vtp(std::string name)
    {
      size_t n_interactions = network.size();
      std::ofstream outFile(name);
      if (!outFile)
      {
        color_log::error("dump_network", "Impossible to create the output file: " + name, false);
        return;
      }

      outFile << "<?xml version=\"1.0\"?>" << std::endl;
      outFile << "<VTKFile type=\"PolyData\">" << std::endl;
      outFile << "  <PolyData>" << std::endl;
      outFile << "    <Piece NumberOfPoints=\"" << n_interactions*2 << "\" NumberOfLines=\"" << n_interactions << "\">" << std::endl;
      outFile << "    <PointData>" << std::endl;
      outFile << "      <DataArray type=\"Float64\" Name=\"fn\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
      if (n_interactions != 0)
        outFile << val.rdbuf() << std::endl;
      outFile << "      </DataArray>" << std::endl;
      outFile << "    </PointData>" << std::endl;
      outFile << "    <Points>" << std::endl;
      outFile << "      <DataArray type=\"Float64\" Name=\"\"  NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
      if (n_interactions != 0)
        outFile << pos.rdbuf() << std::endl;
      outFile << "      </DataArray>" << std::endl;
      outFile << "    </Points>" << std::endl;
      outFile << "    <Lines>" << std::endl;
      outFile << "      <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << std::endl;
      for (size_t i = 0; i < 2*n_interactions; i++)
        outFile << " " << i;
      outFile << "      </DataArray>" << std::endl;
      outFile << "      <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << std::endl;
      for (size_t i = 1; i <= n_interactions; i++)
        outFile << " " << 2 * i;
      outFile << std::endl;
      outFile << "      </DataArray>" << std::endl;
      outFile << "    </Lines>" << std::endl;
      outFile << "    </Piece>" << std::endl;
      outFile << "  </PolyData>" << std::endl;
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
    void write_pvtp(std::string basename, size_t number_of_files)
    {
      std::string name = basename + ".pvtp";
      std::ofstream outFile(name);
      if (!outFile)
      {
        color_log::error("dump_network", "Impossible to create the output file: " + name, false);
        return;
      }
      outFile << "<?xml version=\"1.0\"?>" << std::endl;
      outFile << "<VTKFile type=\"PPolyData\"> " << std::endl;
      outFile << "   <PPolyData GhostLevel=\"0\">" << std::endl;
      outFile << "     <PPoints Scalar=\"fn\">" << std::endl;
      outFile << "       <PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>" << std::endl;
      outFile << "     </PPoints> " << std::endl;
      outFile << "     <PPointData>" << std::endl;
      outFile << "       <PDataArray type=\"Float64\" Name=\"fn\" NumberOfComponents=\"1\"/>" << std::endl;
      outFile << "     </PPointData>" << std::endl;
      std::filesystem::path full_path(basename);
      std::string directory = full_path.filename().string();
      std::string subfile = directory + "/%06d.vtp";
      for (size_t i = 0; i < number_of_files; i++)
      {
        std::string file = onika::format_string(subfile,  i);
        outFile << "     <Piece Source=\"" << file << "\"/>" << std::endl;
      }
      outFile << "   </PPolyData>" << std::endl;
      outFile << "</VTKFile>" << std::endl;
    }
  };
} // namespace exaDEM
