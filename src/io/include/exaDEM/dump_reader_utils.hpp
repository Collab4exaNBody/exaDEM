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

#include <mpi.h>
#include <onika/file_utils.h>
#include <onika/log.h>

// ExaNBody
#include <exanb/core/domain.h>
#include <exanb/extra_storage/dump_filter_dynamic_data_storage.h>
#include <exanb/io/grid_memory_compact.h>
#include <exanb/io/mpi_file_io.h>
#include <exanb/io/sim_dump_io.h>
#include <exanb/io/sim_dump_reader.h>

// ExaDEM
#include <exaDEM/dump_field_sets.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/placeholder_interaction.hpp>

namespace exaDEM {
using namespace exanb;

/**
 * @brief Reads a .dump's header and returns its field names, without reading any particle data.
 * @param mpi MPI communicator used to open the file.
 * @param filename Path to the .dump file.
 * @return The dump's field names, in header order.
 */
inline std::vector<std::string> read_dump_field_names(MPI_Comm mpi, const std::string& filename) {
  std::string file_name = onika::data_file_path(filename);
  MpiIO file;
  file.open(mpi, file_name, "r");
  SimDumpHeader header = {};
  file.read(&header);
  header.post_process();
  file.close();
  return std::vector<std::string>(header.m_fields, header.m_fields + header.m_nb_fields);
}

/**
 * @brief True if this dump was written with fragmentation enabled (has a "cluster" field).
 * @param field_names Field names from read_dump_field_names().
 */
inline bool dump_field_names_have_fragmentation(const std::vector<std::string>& field_names) {
  return std::find(field_names.begin(), field_names.end(), "cluster") != field_names.end();
}

/**
 * @brief True if this dump has a "group" field (false for pre-group, legacy122 dumps).
 * @param field_names Field names from read_dump_field_names().
 */
inline bool dump_field_names_have_group(const std::vector<std::string>& field_names) {
  return std::find(field_names.begin(), field_names.end(), "group") != field_names.end();
}

/**
 * @brief Reads `filename` into `grid`/`domain`/`ges`, picking whichever of the 4 known field-set
 * combinations matches `fragmentation`/`has_group`. Bootstraps grid's cell allocator first if
 * needed, so GridT must be able to hold any of the 4 (see DumpReaderGridT).
 * @param mpi MPI communicator used for the collective read.
 * @param grid Grid to read particles into.
 * @param domain Domain, updated from the dump's header.
 * @param ges Interaction storage, updated from the dump's extra data.
 * @param physical_time Updated from the dump's header.
 * @param timestep Updated from the dump's header.
 * @param filename Path to the .dump file.
 * @param fragmentation Whether the dump has a "cluster" field (see dump_field_names_have_fragmentation).
 * @param has_group Whether the dump has a "group" field (see dump_field_names_have_group).
 */
template <typename GridT>
inline void read_dump_particles(MPI_Comm mpi, GridT& grid, Domain& domain, GridCellParticleInteraction& ges,
                                double& physical_time, long& timestep, const std::string& filename, bool fragmentation,
                                bool has_group) {
  if (grid.number_of_cells() == 0) {
    grid.set_cell_allocator_for_fields(FragmentationDEMFieldSet{});
    grid.rebuild_particle_offsets();
  }
  if (fragmentation && has_group) {
    ParticleDumpFilterWithExtraDataStorage<GridT, PlaceholderInteraction, DumpFragmentationFieldSet> dump_filter = {
        ges, grid};
    exanb::read_dump(mpi, ldbg, grid, domain, physical_time, timestep, filename, DumpFragmentationFieldSet{},
                     dump_filter);
  } else if (fragmentation && !has_group) {
    ParticleDumpFilterWithExtraDataStorage<GridT, PlaceholderInteraction, DumpFragmentationFieldSetLegacy122>
        dump_filter = {ges, grid};
    exanb::read_dump(mpi, ldbg, grid, domain, physical_time, timestep, filename, DumpFragmentationFieldSetLegacy122{},
                     dump_filter);
  } else if (!fragmentation && has_group) {
    ParticleDumpFilterWithExtraDataStorage<GridT, PlaceholderInteraction, DumpFieldSet> dump_filter = {ges, grid};
    exanb::read_dump(mpi, ldbg, grid, domain, physical_time, timestep, filename, DumpFieldSet{}, dump_filter);
  } else {
    ParticleDumpFilterWithExtraDataStorage<GridT, PlaceholderInteraction, DumpFieldSetLegacy122> dump_filter = {ges,
                                                                                                                grid};
    exanb::read_dump(mpi, ldbg, grid, domain, physical_time, timestep, filename, DumpFieldSetLegacy122{}, dump_filter);
  }
  exanb::grid_memory_compact(grid);
}

/**
 * @brief Like read_dump_particles, for a pipeline-embedded GridT known to support fragmentation
 * (has a "cluster" field): no allocator bootstrap, only group/legacy122 is picked at runtime.
 * @param mpi MPI communicator used for the collective read.
 * @param grid Grid to read particles into; must support field::_cluster.
 * @param domain Domain, updated from the dump's header.
 * @param ges Interaction storage, updated from the dump's extra data.
 * @param physical_time Updated from the dump's header.
 * @param timestep Updated from the dump's header.
 * @param filename Path to the .dump file.
 * @param has_group Whether the dump has a "group" field (see dump_field_names_have_group).
 */
template <typename GridT>
inline void read_dump_particles_fragmentation(MPI_Comm mpi, GridT& grid, Domain& domain,
                                              GridCellParticleInteraction& ges, double& physical_time, long& timestep,
                                              const std::string& filename, bool has_group) {
  if (has_group) {
    ParticleDumpFilterWithExtraDataStorage<GridT, PlaceholderInteraction, DumpFragmentationFieldSet> dump_filter = {
        ges, grid};
    exanb::read_dump(mpi, ldbg, grid, domain, physical_time, timestep, filename, DumpFragmentationFieldSet{},
                     dump_filter);
  } else {
    ParticleDumpFilterWithExtraDataStorage<GridT, PlaceholderInteraction, DumpFragmentationFieldSetLegacy122>
        dump_filter = {ges, grid};
    exanb::read_dump(mpi, ldbg, grid, domain, physical_time, timestep, filename, DumpFragmentationFieldSetLegacy122{},
                     dump_filter);
  }
  exanb::grid_memory_compact(grid);
}

/**
 * @brief Like read_dump_particles, for a pipeline-embedded GridT with no fragmentation support
 * (no "cluster" field): no allocator bootstrap, only group/legacy122 is picked at runtime.
 * @param mpi MPI communicator used for the collective read.
 * @param grid Grid to read particles into.
 * @param domain Domain, updated from the dump's header.
 * @param ges Interaction storage, updated from the dump's extra data.
 * @param physical_time Updated from the dump's header.
 * @param timestep Updated from the dump's header.
 * @param filename Path to the .dump file.
 * @param has_group Whether the dump has a "group" field (see dump_field_names_have_group).
 */
template <typename GridT>
inline void read_dump_particles_interaction(MPI_Comm mpi, GridT& grid, Domain& domain, GridCellParticleInteraction& ges,
                                            double& physical_time, long& timestep, const std::string& filename,
                                            bool has_group) {
  if (has_group) {
    ParticleDumpFilterWithExtraDataStorage<GridT, PlaceholderInteraction, DumpFieldSet> dump_filter = {ges, grid};
    exanb::read_dump(mpi, ldbg, grid, domain, physical_time, timestep, filename, DumpFieldSet{}, dump_filter);
  } else {
    ParticleDumpFilterWithExtraDataStorage<GridT, PlaceholderInteraction, DumpFieldSetLegacy122> dump_filter = {ges,
                                                                                                                grid};
    exanb::read_dump(mpi, ldbg, grid, domain, physical_time, timestep, filename, DumpFieldSetLegacy122{}, dump_filter);
  }
  exanb::grid_memory_compact(grid);
}

}  // namespace exaDEM
