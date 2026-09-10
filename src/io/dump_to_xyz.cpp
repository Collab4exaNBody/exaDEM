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

// ExaDEM
#include <exaDEM/dump_reader_utils.hpp>
#include <exaDEM/shape_reader.hpp>
#include <exaDEM/shapes.hpp>

namespace exaDEM {
using namespace exanb;

class DumpToXYZNode : public OperatorNode {
  using GridT = DumpReaderGridT;
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(std::string, filename, INPUT, REQUIRED, DocString{"The .dump file to convert."});
  ADD_SLOT(std::string, xyz_filename, INPUT, REQUIRED, DocString{"Output .xyz file path."});
  ADD_SLOT(std::string, shape_filename, INPUT, OPTIONAL,
           DocString{"Optional .shp file mapping each particle type index to its shape name, used as the "
                     "\"type\" column. Without it, the numeric type index is used."});
  ADD_SLOT(std::vector<std::string>, type_map, INPUT, OPTIONAL,
           DocString{"Optional list of names indexed by particle type, e.g. [S1,S2,S3], used as the \"type\" "
                     "column instead of the numeric index. Takes priority over shape_filename's names."});
  ADD_SLOT(GridT, grid, INPUT_OUTPUT);
  ADD_SLOT(Domain, domain, INPUT_OUTPUT);
  ADD_SLOT(GridCellParticleInteraction, ges, INPUT_OUTPUT,
           DocString{"Interaction list -- read (dumps always carry this section) but the XYZ format has no place "
                     "for it."});
  ADD_SLOT(long, timestep, INPUT, 0);
  ADD_SLOT(double, physical_time, INPUT, 0.0);

 public:
  inline bool is_sink() const final { return true; }

  inline std::string documentation() const final {
    return R"EOF(
        Reads a .dump checkpoint file and exports its particles to a plain XYZ file (particle
        count, domain bounds upper corner, then one "type x y z" row per particle).

        YAML example:

          - dump_to_xyz:
             filename: ExaDEMOutputDir/CheckpointFiles/exadem_0000012345.dump
             xyz_filename: out.xyz
             shape_filename: ExaDEMOutputDir/CheckpointFiles/RestartShapeFile.shp
             type_map: [S1, S2, S3]
      )EOF";
  }

  inline void write_xyz() {
    std::vector<std::string> type_names;
    const bool have_type_map = type_map.has_value();
    const bool have_shapes = !have_type_map && shape_filename.has_value();
    if (have_type_map) {
      type_names = *type_map;
    } else if (have_shapes) {
      shapes shps;
      exaDEM::read_shp(shps, *shape_filename);
      type_names.resize(shps.size());
      for (size_t i = 0; i < shps.size(); i++) {
        type_names[i] = shps[i]->name_;
      }
    }
    const bool have_names = have_type_map || have_shapes;

    std::ostringstream particles_buffer;
    particles_buffer.precision(13);
    size_t n_particles = 0;

    auto cells = grid->cells();
    for (size_t c = 0; c < grid->number_of_cells(); c++) {
      if (grid->is_ghost_cell(c)) continue;
      const auto& cell = cells[c];
      const size_t n = cell.size();
      const auto* __restrict__ type = cell[field::type];
      const double* __restrict__ rx = cell[field::rx];
      const double* __restrict__ ry = cell[field::ry];
      const double* __restrict__ rz = cell[field::rz];
      for (size_t p = 0; p < n; p++) {
        const uint32_t t = type[p];
        const std::string name = (have_names && t < type_names.size()) ? type_names[t] : std::to_string(t);
        particles_buffer << name << " " << rx[p] << " " << ry[p] << " " << rz[p] << std::endl;
        n_particles++;
      }
    }

    std::ofstream file(*xyz_filename);
    file.precision(13);
    file << n_particles << std::endl;
    const Vec3d sup = domain->bounds().bmax;
    file << sup.x << " " << sup.y << " " << sup.z << std::endl;
    file << particles_buffer.str();

    lout << "wrote " << n_particles << " particles to " << *xyz_filename << std::endl;
  }

  inline void execute() final {
    int rank = 0;
    MPI_Comm_rank(*mpi, &rank);
    if (rank != 0) {
      color_log::warning("dump_to_xyz",
                         "This operator only writes files from rank 0; run with a single MPI rank "
                         "(mpirun -n 1) for a complete export.");
      return;
    }

    const std::vector<std::string> field_names = read_dump_field_names(*mpi, *filename);
    const bool fragmentation = dump_field_names_have_fragmentation(field_names);
    const bool has_group = dump_field_names_have_group(field_names);

    read_dump_particles(*mpi, *grid, *domain, *ges, *physical_time, *timestep, *filename, fragmentation, has_group);
    write_xyz();
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(dump_to_xyz) {
  OperatorNodeFactory::instance()->register_factory("dump_to_xyz", make_compatible_operator<DumpToXYZNode>);
}
}  // namespace exaDEM
