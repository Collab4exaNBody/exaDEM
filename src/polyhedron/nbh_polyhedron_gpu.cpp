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
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>

#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/particle_neighbors/chunk_neighbors_apply.h>

#include <cassert>

#include <exaDEM/traversal.hpp>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/interaction_manager.hpp>
#include <exaDEM/interaction/migration_test.hpp>
#include <exaDEM/shapes.hpp>
#include <exaDEM/polyhedron/vertices.hpp>
#include <exaDEM/polyhedron/nbh_polyhedron_driver.hpp>
#include <exaDEM/polyhedron/nbh_dem.hpp>
#include <exaDEM/polyhedron/nbh_runner.hpp>

namespace exaDEM {
template <typename GridT, class = AssertGridHasFields<GridT>>
class UpdateClassifierPolyhedronGPU : public OperatorNode {

  ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
  ADD_SLOT(CellVertexField, cvf, INPUT, REQUIRED, DocString{"Store vertex positions for every polyhedron"});
  ADD_SLOT(Domain, domain, INPUT, REQUIRED);
  ADD_SLOT(exanb::GridChunkNeighbors, chunk_neighbors, INPUT, OPTIONAL, DocString{"Neighbor list"});
  ADD_SLOT(GridCellParticleInteraction, ges, INPUT_OUTPUT, DocString{"Interaction list"});
  ADD_SLOT(shapes, shapes_collection, INPUT, REQUIRED, DocString{"Collection of shapes"});
  ADD_SLOT(double, rcut_inc, INPUT, REQUIRED,
           DocString{"value added to the search distance to update neighbor list less frequently. in physical space"});
  ADD_SLOT(Drivers, drivers, INPUT, REQUIRED, DocString{"List of Drivers"});
  ADD_SLOT(Traversal, traversal_real, INPUT, REQUIRED, DocString{"list of non empty cells within the current grid"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This function builds the list of interactions per particle (polyhedron). Interactions are between two particles or a particle and a driver. In this function, frictions and moments are updated if the interactions are still actived. Note that, a list of non-empty cells is built during this function.

        YAML example [no option]:

          - nbh_polyhedron
       )EOF";
  }

  inline void check_slots() {
  }

  inline void execute() final {
    lout << "Operator in progress" << std::endl;
    auto& g = *grid;
    const auto cells = g.cells();
    const IJK dims = g.dimension();
    const size_t n_cells = g.number_of_cells();
    shapes& shps = *shapes_collection;
    auto [cell_ptr, cell_size] = traversal_real->info();

    brute_force_storage storage;
    storage.m_data.resize(n_cells);

    ApplyNbhBruteForceFunc func_brute_force;
    NeighborRunner runner(cell_ptr, dims, func_brute_force,  // members
                          cells, storage.m_data.data(), *rcut_inc, shps.data());  // params

    ParallelForOptions opts;
    opts.omp_scheduling = OMP_SCHED_GUIDED;   
    // 26 = number of neighbor cells per cell
    ParallelExecutionSpace<3> parallel_range = {{0,0,0} , {static_cast<long int>(cell_size),26,1}};
    parallel_for(parallel_range, runner, parallel_execution_context(), opts);

    for (size_t i = 0 ; i < n_cells ; i++) {
      lout << "Neighbors cell["<<i<<"]: "<< storage.m_data[i] << std::endl;
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(nbh_polyhedron_gpu) {
  OperatorNodeFactory::instance()->register_factory("nbh_polyhedron_gpu",
                                                    make_grid_variant_operator<UpdateClassifierPolyhedronGPU>);
}
}  // namespace exaDEM
