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
#include <exanb/core/make_grid_variant_operator.h>

// ExaDEM
#include <exaDEM/dump_reader_utils.hpp>
#include <exaDEM/shape_reader.hpp>
#include <exaDEM/shapes.hpp>

namespace exaDEM {
using namespace exanb;

template <class GridT>
class RestartNode : public OperatorNode {
  static constexpr bool fragmentation_grid = exanb::grid_has_field_v<GridT, field::_cluster>;

  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(std::string, dir_name, INPUT, REQUIRED,
           DocString{"Main output directory, usually set by io_config. Checkpoints are looked up under "
                     "<dir_name>/CheckpointFiles/."});
  ADD_SLOT(long, restart_iteration, INPUT, OPTIONAL,
           DocString{"Checkpoint iteration to restart from. Defaults to the latest exadem_*.dump found under "
                     "<dir_name>/CheckpointFiles/."});
  ADD_SLOT(std::string, shape_filename, INPUT, OPTIONAL,
           DocString{"Overrides the default <dir_name>/CheckpointFiles/RestartShapeFile.shp shape file."});
  ADD_SLOT(GridT, grid, INPUT_OUTPUT);
  ADD_SLOT(Domain, domain, INPUT_OUTPUT);
  ADD_SLOT(GridCellParticleInteraction, ges, INPUT_OUTPUT, DocString{"Interaction list"});
  ADD_SLOT(long, timestep, INPUT, DocString{"Iteration number"});
  ADD_SLOT(double, physical_time, INPUT, DocString{"Physical time"});
  ADD_SLOT(shapes, shapes_collection, INPUT_OUTPUT, DocString{"Collection of shapes"});
  ADD_SLOT(ParticleTypeMap, particle_type_map, INPUT_OUTPUT);

 public:
  inline std::string documentation() const final {
    return R"EOF(
        Finds and reads the files needed to restart a simulation: the latest (or a given)
        exadem_*.dump checkpoint under <dir_name>/CheckpointFiles/, and, if the simulation uses
        polyhedra, the matching shape file (RestartShapeFile.shp by default). For spheres there
        is no shape file at all: if none is given and none is found at the default path, shape
        loading is simply skipped and particles restart with the radii stored in the dump itself.

        YAML example:

          input_data:
            - restart

          input_data:
            - restart:
               restart_iteration: 12345
               shape_filename: OtherCheckpointDir/shapes.shp
      )EOF";
  }

  inline long find_latest_checkpoint_iteration(const std::filesystem::path& checkpoint_dir) const {
    long latest = -1;
    for (const auto& entry : std::filesystem::directory_iterator(checkpoint_dir)) {
      const std::string name = entry.path().filename().string();
      if (name.rfind("exadem_", 0) != 0 || entry.path().extension() != ".dump") continue;
      const std::string num_str = name.substr(7, name.size() - 7 - 5);  // strip "exadem_" and ".dump"
      try {
        latest = std::max(latest, std::stol(num_str, nullptr, 10));
      } catch (const std::exception&) {
        continue;
      }
    }
    return latest;
  }

  inline void execute() final {
    namespace fs = std::filesystem;

    const fs::path checkpoint_dir = *dir_name + "/CheckpointFiles";
    if (!fs::is_directory(checkpoint_dir)) {
      color_log::error("restart", "No checkpoint directory found at " + checkpoint_dir.string() +
                                      " (set dir_name, usually via io_config, to override).");
    }

    long iteration = 0;
    if (restart_iteration.has_value()) {
      iteration = *restart_iteration;
    } else {
      iteration = find_latest_checkpoint_iteration(checkpoint_dir);
      if (iteration < 0) {
        color_log::error("restart", "No exadem_*.dump file found in " + checkpoint_dir.string());
      }
    }

    std::ostringstream dump_name;
    dump_name << "exadem_" << std::setw(10) << std::setfill('0') << iteration << ".dump";
    const fs::path dump_file = checkpoint_dir / dump_name.str();
    if (!fs::is_regular_file(dump_file)) {
      color_log::error("restart", "Checkpoint file not found: " + dump_file.string());
    }

    // Spheres have no shape file at all: shp_file is left empty (not an error) when none is
    // given and none exists at the default path, and shape loading below is simply skipped.
    fs::path shp_file;
    if (shape_filename.has_value()) {
      shp_file = *shape_filename;
      if (!fs::is_regular_file(shp_file)) {
        color_log::error("restart", "Shape file not found: " + shp_file.string());
      }
    } else {
      fs::path default_shp = checkpoint_dir / "RestartShapeFile.shp";
      if (fs::is_regular_file(default_shp)) shp_file = default_shp;
    }

    lout << "==================== Restart =================" << std::endl;
    lout << "Checkpoint directory: " << checkpoint_dir.string() << std::endl;
    lout << "Checkpoint file:      " << dump_file.string() << std::endl;
    lout << "Particle mode:        " << (shp_file.empty() ? "Spheres" : "Polyhedra") << std::endl;
    if (!shp_file.empty()) {
      lout << "Shape file:           " << shp_file.string() << std::endl;
    }
    lout << "=================================================" << std::endl;

    if (!shp_file.empty()) {
      std::vector<shape> list_of_shapes = exaDEM::read_shps(shp_file.string(), false, false);
      exaDEM::register_shapes(*particle_type_map, *shapes_collection, list_of_shapes);
      for (const auto& [name, type] : *particle_type_map) {
        lout << "Shape[" << type << "] is " << name << std::endl;
      }
    }

    const std::vector<std::string> field_names = read_dump_field_names(*mpi, dump_file.string());
    const bool file_has_fragmentation = dump_field_names_have_fragmentation(field_names);
    const bool has_group = dump_field_names_have_group(field_names);

    if (file_has_fragmentation != fragmentation_grid) {
      color_log::error("restart", "Checkpoint " + dump_file.string() + (file_has_fragmentation ? " has" : " has no") +
                                      " fragmentation (cluster) data, but this simulation's grid flavor " +
                                      (fragmentation_grid ? "supports" : "does not support") +
                                      " it -- check your grid_flavor / includes.");
    }

    if constexpr (fragmentation_grid) {
      read_dump_particles_fragmentation(*mpi, *grid, *domain, *ges, *physical_time, *timestep, dump_file.string(),
                                        has_group);
    } else {
      read_dump_particles_interaction(*mpi, *grid, *domain, *ges, *physical_time, *timestep, dump_file.string(),
                                      has_group);
    }

    lout << "Restarted from iteration " << iteration << " (t=" << *physical_time << ")" << std::endl;
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(restart) {
  OperatorNodeFactory::instance()->register_factory("restart", make_grid_variant_operator<RestartNode>);
}
}  // namespace exaDEM
