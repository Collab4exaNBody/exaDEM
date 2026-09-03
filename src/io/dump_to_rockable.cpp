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

// ExaDEM
#include <algorithm>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/placeholder_interaction.hpp>
#include <exaDEM/shape_reader.hpp>
#include <exaDEM/shapes.hpp>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace exaDEM {
using namespace exanb;

// TODO add a common .hpp for the 4 FieldSet definitions below.
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

using DumpToRockableGridT = GridFromFieldSet<FragmentationDEMFieldSet>;

class DumpToRockableNode : public OperatorNode {
  using GridT = DumpToRockableGridT;
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(std::string, filename, INPUT, REQUIRED, DocString{"The .dump file to convert."});
  ADD_SLOT(std::string, conf_filename, INPUT, REQUIRED, DocString{"Output Rockable .conf file path."});
  ADD_SLOT(std::string, shape_filename, INPUT, OPTIONAL,
           DocString{"Optional .shp file mapping each particle type index to its shape name; copied next to "
                     "conf_filename as shape.shp. Without it, the numeric type index is used as the name."});
  ADD_SLOT(GridT, grid, INPUT_OUTPUT);
  ADD_SLOT(Domain, domain, INPUT_OUTPUT);
  ADD_SLOT(GridCellParticleInteraction, ges, INPUT_OUTPUT,
           DocString{"Interaction list -- read (dumps always carry this section) but not exported yet."});
  ADD_SLOT(long, timestep, INPUT, 0);
  ADD_SLOT(double, physical_time, INPUT, 0.0);
  ADD_SLOT(double, dt, INPUT, 0.0,
           DocString{"Timestep size to write in the .conf's \"dt\" line; not stored in a .dump, so it defaults to "
                     "0 unless given explicitly."});

 public:
  inline bool is_sink() const final { return true; }

  inline std::string documentation() const final {
    return R"EOF(
        Reads a .dump checkpoint file and exports its particles to a Rockable .conf file.

        YAML example:

          - dump_to_rockable:
             filename: ExaDEMOutputDir/CheckpointFiles/exadem_0000012345.dump
             conf_filename: conf0.conf
             shape_filename: ExaDEMOutputDir/CheckpointFiles/RestartShapeFile.shp
             dt: 0.0001
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

  inline void write_conf(bool has_group, bool has_cluster) {
    std::vector<std::string> type_names;
    shapes shps;  // kept alive for get_volume() (density), not just the name lookup below
    const bool have_shapes = shape_filename.has_value();
    if (have_shapes) {
      exaDEM::read_shp(shps, *shape_filename);
      type_names.resize(shps.size());
      for (size_t i = 0; i < shps.size(); i++) {
        type_names[i] = shps[i]->name_;
      }
    }

    namespace fs = std::filesystem;
    fs::path conf_path(*conf_filename);
    if (conf_path.has_parent_path()) {
      fs::create_directories(conf_path.parent_path());
    }

    const int prec = 13;  // could an input slot
    std::ostringstream particles_buffer;
    particles_buffer.precision(prec);
    size_t n_particles = 0;
    std::map<uint32_t, double> group_density;

    auto cells = grid->cells();
    for (size_t c = 0; c < grid->number_of_cells(); c++) {
      if (grid->is_ghost_cell(c)) continue;
      const auto& cell = cells[c];
      const size_t n = cell.size();
      const auto* __restrict__ type = cell[field::type];
      const double* __restrict__ h = cell[field::homothety];
      const double* __restrict__ mass = cell[field::mass];
      const double* __restrict__ rx = cell[field::rx];
      const double* __restrict__ ry = cell[field::ry];
      const double* __restrict__ rz = cell[field::rz];
      const double* __restrict__ vx = cell[field::vx];
      const double* __restrict__ vy = cell[field::vy];
      const double* __restrict__ vz = cell[field::vz];
      const Quaternion* __restrict__ quat = cell[field::orient];
      const Vec3d* __restrict__ vrot = cell[field::vrot];
      const Vec3d* __restrict__ arot = cell[field::arot];
      const uint32_t* __restrict__ group = has_group ? cell[field::group] : nullptr;
      const uint32_t* __restrict__ cluster = has_cluster ? cell[field::cluster] : nullptr;
      for (size_t p = 0; p < n; p++) {
        const uint32_t t = type[p];
        const uint32_t g = group ? group[p] : 0;
        const std::string name = (have_shapes && t < type_names.size()) ? type_names[t] : std::to_string(t);
        if (have_shapes && t < shps.size() && group_density.find(g) == group_density.end()) {
          group_density[g] = mass[p] / shps[t]->get_volume(h[p]);
        }
        particles_buffer << name << " " << g << " " << (cluster ? cluster[p] : 0) << " " << h[p] << " " << rx[p] << " "
                         << ry[p] << " " << rz[p] << " " << vx[p] << " " << vy[p] << " " << vz[p] << " " << 0 << " "
                         << 0 << " " << 0 << " "  // acc: force is not stored in a .dump
                         << quat[p].w << " " << quat[p].x << " " << quat[p].y << " " << quat[p].z << " " << vrot[p].x
                         << " " << vrot[p].y << " " << vrot[p].z << " " << arot[p].x << " " << arot[p].y << " "
                         << arot[p].z << std::endl;
        n_particles++;
      }
    }

    std::ofstream file(*conf_filename);
    file << "Rockable 29-11-2018" << std::endl;
    file << "t " << *physical_time << std::endl;
    file << "dt " << *dt << std::endl;  // not stored in a .dump; 0 unless given explicitly
    file << "iconf 0" << std::endl;
    file << "periodicity " << domain->periodic_boundary_x() << " " << domain->periodic_boundary_y() << " "
         << domain->periodic_boundary_z() << std::endl;
    file << "nDriven 0" << std::endl;  // drivers are not exported yet
    file << "shapeFile shape.shp" << std::endl;
    file.precision(prec);
    for (const auto& [g, density] : group_density) {
      file << "density " << g << " " << density << std::endl;
    }
    file << "precision " << prec << std::endl;
    file << "Particles " << n_particles << std::endl;
    file << particles_buffer.str();

    if (have_shapes) {
      fs::path out_shp = conf_path.has_parent_path() ? (conf_path.parent_path() / "shape.shp") : fs::path("shape.shp");
      std::error_code ec;
      fs::copy_file(*shape_filename, out_shp, fs::copy_options::overwrite_existing, ec);
      if (ec) {
        color_log::warning("dump_to_rockable", "Failed to copy shape file next to the .conf: " + ec.message());
      }
    } else {
      lout << "no shape_filename given: the particle 'name' column is the numeric type index, and shape.shp was "
              "not written."
           << std::endl;
    }

    lout << "wrote " << n_particles << " particles to " << *conf_filename << std::endl;
  }

  inline void execute() final {
    int rank = 0;
    MPI_Comm_rank(*mpi, &rank);
    if (rank != 0) {
      color_log::warning("dump_to_rockable",
                         "This operator only writes files from rank 0; run with a single MPI rank "
                         "(mpirun -n 1) for a complete export.");
      return;
    }

    const std::vector<std::string> field_names = read_dump_field_names();
    const bool fragmentation = std::find(field_names.begin(), field_names.end(), "cluster") != field_names.end();
    const bool has_group = std::find(field_names.begin(), field_names.end(), "group") != field_names.end();

    read_particles(fragmentation, has_group);
    write_conf(has_group, fragmentation);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(dump_to_rockable) {
  OperatorNodeFactory::instance()->register_factory("dump_to_rockable", make_compatible_operator<DumpToRockableNode>);
}
}  // namespace exaDEM
