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
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

// exaNBody
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>

// reduction
#include <exanb/compute/reduce_cell_particles.h>

// exaDEM
#include <exaDEM/analysis_manager.hpp>
#include <exaDEM/sphere_compacity.hpp>
#include <exaDEM/traversal.hpp>

namespace exaDEM {

using namespace exanb;
template <typename GridT, class = AssertGridHasFields<GridT>>
class SphereCompacityAnalysis : public OperatorNode {
  static constexpr FieldSet<field::_radius> reduce_field_set{};
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(Traversal, traversal_real, INPUT, REQUIRED, DocString{"list of non empty cells within the current grid"});
  ADD_SLOT(double, volume, INPUT, REQUIRED, DocString{"Domain volume, e.g. from domain_volume."});
  ADD_SLOT(double, dt, INPUT, REQUIRED);
  ADD_SLOT(long, timestep, INPUT, REQUIRED, DocString{"Time iteration number"});
  ADD_SLOT(std::string, dir_name, INPUT, REQUIRED, DocString{"Output directory, usually defined into io_config."});
  ADD_SLOT(std::string, name, INPUT, "SphereCompacity.txt", DocString{"Filename. Default is: SphereCompacity.txt"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        Computes the compacity (volume fraction) of a sphere assembly.

        YAML example:

          - sphere_compacity
      )EOF";
  }

  inline void execute() final {
    analysis::AnalysisFileManager manager = {};
    manager.set_path((*dir_name) + "/ExaDEMAnalyses");
    manager.set_filename(*name);
    double t = (*dt) * (*timestep);
    manager.add_element("Time", t, "%3f");

    const ReduceCellParticlesOptions rcpo = traversal_real->get_reduce_cell_particles_options();

    ReduceSphereVolumeFunctor func;
    SphereVolumeValue value = {0, 0.0, 0.0};
    reduce_cell_particles(*grid, false, func, value, reduce_field_set, parallel_execution_context(), {}, rcpo);

    double local[3] = {static_cast<double>(value.count_), value.sum_radius_, value.sum_volume_};
    double global[3] = {0.0, 0.0, 0.0};
    MPI_Allreduce(local, global, 3, MPI_DOUBLE, MPI_SUM, *mpi);

    double mean_radius = (global[0] > 0.0) ? (global[1] / global[0]) : 0.0;
    double compacity = global[2] / (*volume);

    manager.add_element("NbParticles", global[0], "%f");
    manager.add_element("MeanRadius", mean_radius, "%f");
    manager.add_element("Compacity", compacity, "%f");
    manager.endl();
    manager.write();
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(sphere_compacity) {
  OperatorNodeFactory::instance()->register_factory("sphere_compacity",
                                                    make_grid_variant_operator<SphereCompacityAnalysis>);
}
}  // namespace exaDEM
