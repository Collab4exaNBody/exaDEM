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
#include <onika/log.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

// ExaNBody
#include <exanb/core/domain.h>
#include <exanb/core/grid.h>
#include <exanb/core/grid_fields.h>
#include <exanb/extra_storage/dump_filter_dynamic_data_storage.h>
#include <exanb/io/grid_memory_compact.h>
#include <exanb/io/mpi_file_io.h>
#include <exanb/io/sim_dump_io.h>
#include <exanb/io/sim_dump_reader.h>

#include <algorithm>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/inner_bond_interaction.hpp>
#include <exaDEM/interaction/interaction.hpp>
#include <fstream>
#include <string>
#include <vector>

namespace exaDEM {
using namespace exanb;

using DumpFieldSet = FieldSet<field::_rx, field::_ry, field::_rz, field::_vx, field::_vy, field::_vz, field::_mass,
                              field::_homothety, field::_radius, field::_orient, field::_mom, field::_vrot,
                              field::_arot, field::_inertia, field::_id, field::_type, field::_group>;
using DumpFragmentationFieldSet =
    FieldSet<field::_rx, field::_ry, field::_rz, field::_vx, field::_vy, field::_vz, field::_cluster, field::_mass,
             field::_homothety, field::_radius, field::_orient, field::_mom, field::_vrot, field::_arot,
             field::_inertia, field::_id, field::_type, field::_group>;
using DumpFieldSetLegacy122 = FieldSet<field::_rx, field::_ry, field::_rz, field::_vx, field::_vy, field::_vz,
                                       field::_mass, field::_homothety, field::_radius, field::_orient, field::_mom,
                                       field::_vrot, field::_arot, field::_inertia, field::_id, field::_type>;
using DumpFragmentationFieldSetLegacy122 =
    FieldSet<field::_rx, field::_ry, field::_rz, field::_vx, field::_vy, field::_vz, field::_cluster, field::_mass,
             field::_homothety, field::_radius, field::_orient, field::_mom, field::_vrot, field::_arot,
             field::_inertia, field::_id, field::_type>;

using DumpToTxtGridT = GridFromFieldSet<FragmentationDEMFieldSet>;

class DumpToTxtNode : public OperatorNode {
  using GridT = DumpToTxtGridT;
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(std::string, filename, INPUT, REQUIRED, DocString{"The .dump file to export."});
  ADD_SLOT(GridT, grid, INPUT_OUTPUT);
  ADD_SLOT(Domain, domain, INPUT_OUTPUT);
  ADD_SLOT(GridCellParticleInteraction, ges, INPUT_OUTPUT, DocString{"Interaction list"});
  ADD_SLOT(long, timestep, INPUT, 0);
  ADD_SLOT(double, physical_time, INPUT, 0.0);
  ADD_SLOT(std::string, pattern_name, INPUT, REQUIRED,
           DocString{"Output prefix: writes <pattern_name>_particles.txt, _interactions.txt (if any) and "
                     "_summary.txt."});

 public:
  inline bool is_sink() const final { return true; }

  inline std::string documentation() const final {
    return R"EOF(
        Reads a .dump checkpoint file and exports it to plain-text column files: particles (one
        row per particle, columns taken from the dump's own header), interactions (one row per
        interaction, only written if any exist), and a summary. Standalone: picks the reader
        matching the dump (interaction or fragmentation) itself, from its header.

        YAML example:

          - dump_to_txt:
             filename: exadem_0000012345.dump
             pattern_name: exadem_0000012345
      )EOF";
  }

  inline std::vector<std::string> read_dump_field_names() {
    std::string file_name = onika::data_file_path(*filename);
    MpiIO file;
    file.open(*mpi, file_name, "r");
    SimDumpHeader header = {};
    file.read(&header);
    header.post_process();
    file.close();
    return std::vector<std::string>(header.m_fields, header.m_fields + header.m_nb_fields);
  }

  // Picks whichever of the 4 known field-set combinations (see DumpFieldSet & friends above)
  // exactly matches this dump's header, and reads with it.
  inline void read_particles(bool fragmentation, bool has_group) {
    if (grid->number_of_cells() == 0) {
      grid->set_cell_allocator_for_fields(FragmentationDEMFieldSet{});
      grid->rebuild_particle_offsets();
    }
    if (fragmentation && has_group) {
      ParticleDumpFilterWithExtraDataStorage<GridT, PlaceholderInteraction, DumpFragmentationFieldSet> dump_filter = {
          *ges, *grid};
      exanb::read_dump(*mpi, ldbg, *grid, *domain, *physical_time, *timestep, *filename, DumpFragmentationFieldSet{},
                       dump_filter);
    } else if (fragmentation && !has_group) {
      ParticleDumpFilterWithExtraDataStorage<GridT, PlaceholderInteraction, DumpFragmentationFieldSetLegacy122>
          dump_filter = {*ges, *grid};
      exanb::read_dump(*mpi, ldbg, *grid, *domain, *physical_time, *timestep, *filename,
                       DumpFragmentationFieldSetLegacy122{}, dump_filter);
    } else if (!fragmentation && has_group) {
      ParticleDumpFilterWithExtraDataStorage<GridT, PlaceholderInteraction, DumpFieldSet> dump_filter = {*ges, *grid};
      exanb::read_dump(*mpi, ldbg, *grid, *domain, *physical_time, *timestep, *filename, DumpFieldSet{}, dump_filter);
    } else {
      ParticleDumpFilterWithExtraDataStorage<GridT, PlaceholderInteraction, DumpFieldSetLegacy122> dump_filter = {
          *ges, *grid};
      exanb::read_dump(*mpi, ldbg, *grid, *domain, *physical_time, *timestep, *filename, DumpFieldSetLegacy122{},
                       dump_filter);
    }
    exanb::grid_memory_compact(*grid);
  }

  inline bool write_particle_header_field(std::ostream& out, const std::string& name) const {
    if (name == "rx" || name == "ry" || name == "rz" || name == "vx" || name == "vy" || name == "vz" ||
        name == "mass" || name == "homothety" || name == "radius" || name == "id" || name == "type" ||
        name == "group" || name == "cluster") {
      out << name << " ";
    } else if (name == "orient") {
      out << "orient_w orient_x orient_y orient_z ";
    } else if (name == "mom" || name == "vrot" || name == "arot" || name == "inertia") {
      out << name << "_x " << name << "_y " << name << "_z ";
    } else {
      return false;
    }
    return true;
  }

  // Appends one field's value(s) for particle p of the given cell to the row.
  template <typename CellT>
  inline void write_particle_value_field(std::ostream& out, const std::string& name, CellT& cell, size_t p) const {
    if (name == "rx") {
      out << cell[field::rx][p] << " ";
    } else if (name == "ry") {
      out << cell[field::ry][p] << " ";
    } else if (name == "rz") {
      out << cell[field::rz][p] << " ";
    } else if (name == "vx") {
      out << cell[field::vx][p] << " ";
    } else if (name == "vy") {
      out << cell[field::vy][p] << " ";
    } else if (name == "vz") {
      out << cell[field::vz][p] << " ";
    } else if (name == "mass") {
      out << cell[field::mass][p] << " ";
    } else if (name == "homothety") {
      out << cell[field::homothety][p] << " ";
    } else if (name == "radius") {
      out << cell[field::radius][p] << " ";
    } else if (name == "id") {
      out << cell[field::id][p] << " ";
    } else if (name == "type") {
      out << cell[field::type][p] << " ";
    } else if (name == "group") {
      out << cell[field::group][p] << " ";
    } else if (name == "cluster") {
      out << cell[field::cluster][p] << " ";
    } else if (name == "orient") {
      const auto& q = cell[field::orient][p];
      out << q.w << " " << q.x << " " << q.y << " " << q.z << " ";
    } else if (name == "mom") {
      const auto& v = cell[field::mom][p];
      out << v.x << " " << v.y << " " << v.z << " ";
    } else if (name == "vrot") {
      const auto& v = cell[field::vrot][p];
      out << v.x << " " << v.y << " " << v.z << " ";
    } else if (name == "arot") {
      const auto& v = cell[field::arot][p];
      out << v.x << " " << v.y << " " << v.z << " ";
    } else if (name == "inertia") {
      const auto& v = cell[field::inertia][p];
      out << v.x << " " << v.y << " " << v.z << " ";
    }
  }

  inline size_t write_particles(const std::vector<std::string>& field_names) {
    std::ofstream out(*pattern_name + "_particles.txt");
    for (const auto& name : field_names) {
      write_particle_header_field(out, name);
    }
    out << std::endl;

    auto cells = grid->cells();
    size_t n_written = 0;
    for (size_t c = 0; c < grid->number_of_cells(); c++) {
      if (grid->is_ghost_cell(c)) continue;
      const size_t n = cells[c].size();
      for (size_t p = 0; p < n; p++) {
        for (const auto& name : field_names) {
          write_particle_value_field(out, name, cells[c], p);
        }
        out << std::endl;
        n_written++;
      }
    }
    return n_written;
  }

  inline void write_interaction_row(std::ostream& out, const PlaceholderInteraction& item) const {
    const auto& p = item.pair_;
    out << p.type_ << " " << p.pi_.id_ << " " << p.pj_.id_ << " " << p.pi_.cell_ << " " << p.pj_.cell_ << " "
        << p.pi_.p_ << " " << p.pj_.p_ << " " << p.pi_.sub_ << " " << p.pj_.sub_ << " " << int(p.swap_) << " "
        << int(p.ghost_) << " ";
    if (p.type_ < InteractionTypeId::NTypesParticleParticle) {
      const auto& I = item.as<Interaction>();
      const Vec3d& friction = I[attr::friction];
      const Vec3d& moment = I[attr::moment];
      out << friction.x << " " << friction.y << " " << friction.z << " " << moment.x << " " << moment.y << " "
          << moment.z << " 0 0 0 0 0";
    } else {  // InnerBond
      const auto& I = item.as<InnerBondInteraction>();
      const Vec3d& friction = I[attr::friction];
      out << friction.x << " " << friction.y << " " << friction.z << " 0 0 0 " << I[attr::en] << " " << I[attr::et]
          << " " << I[attr::dn0] << " " << I[attr::weight] << " " << int(I[attr::unbroken]);
    }
    out << std::endl;
  }

  inline size_t write_interactions() {
    size_t n_written = 0;
    for (size_t c = 0; c < ges->m_data.size(); c++) {
      for (auto& item : ges->m_data[c].m_data) {
        if (item.pair_.ghost_ != InteractionPair::PartnerGhost) n_written++;
      }
    }
    if (n_written == 0) return 0;

    std::ofstream out(*pattern_name + "_interactions.txt");
    out << "type id_i id_j cell_i cell_j p_i p_j sub_i sub_j swap ghost friction_x friction_y friction_z moment_x "
           "moment_y moment_z en et dn0 weight unbroken"
        << std::endl;
    for (size_t c = 0; c < ges->m_data.size(); c++) {
      for (auto& item : ges->m_data[c].m_data) {
        if (item.pair_.ghost_ != InteractionPair::PartnerGhost) {
          write_interaction_row(out, item);
        }
      }
    }
    return n_written;
  }

  inline void write_summary(size_t n_particles, size_t n_interactions) {
    std::ofstream out(*pattern_name + "_summary.txt");
    out << "file           = " << *filename << std::endl;
    out << "time step      = " << *timestep << std::endl;
    out << "time           = " << *physical_time << std::endl;
    out << "particles      = " << n_particles << std::endl;
    out << "interactions   = " << n_interactions << std::endl;
    out << "domain bounds  = " << domain->bounds() << " , size=" << domain->bounds_size() << std::endl;
    out << "domain         = " << *domain << std::endl;
  }

  inline void execute() final {
    int rank = 0;
    MPI_Comm_rank(*mpi, &rank);
    if (rank != 0) {
      color_log::warning("dump_to_txt",
                         "This operator only writes files from rank 0; run with a single MPI rank "
                         "(mpirun -n 1) for a complete export.");
      return;
    }

    const std::vector<std::string> field_names = read_dump_field_names();
    const bool fragmentation = std::find(field_names.begin(), field_names.end(), "cluster") != field_names.end();
    const bool has_group = std::find(field_names.begin(), field_names.end(), "group") != field_names.end();

    read_particles(fragmentation, has_group);

    const size_t n_particles = write_particles(field_names);
    const size_t n_interactions = write_interactions();
    write_summary(n_particles, n_interactions);

    lout << "wrote " << n_particles << " particles to " << *pattern_name << "_particles.txt" << std::endl;
    if (n_interactions > 0) {
      lout << "wrote " << n_interactions << " interactions to " << *pattern_name << "_interactions.txt" << std::endl;
    } else {
      lout << "no interactions in this dump, " << *pattern_name << "_interactions.txt not written" << std::endl;
    }
    lout << "wrote summary to " << *pattern_name << "_summary.txt" << std::endl;
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(dump_to_txt) {
  OperatorNodeFactory::instance()->register_factory("dump_to_txt", make_compatible_operator<DumpToTxtNode>);
}
}  // namespace exaDEM
