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
#include <onika/math/basic_types_yaml.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/grid.h>
#include <exanb/core/grid_fields.h>

#include <mpi.h>

namespace exaDEM {
template <class GridT, class FieldSubSetT = typename GridT::field_set_t>
struct InitGridFlavorNode : public OperatorNode {
  ADD_SLOT(GridT, grid, INPUT_OUTPUT);

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator initializes the DEM grid.
        )EOF";
  }

  inline InitGridFlavorNode() { set_profiling(false); }

  inline void execute() final {
    if (grid->number_of_cells() == 0) {
      grid->set_cell_allocator_for_fields(FieldSubSetT{});
      grid->rebuild_particle_offsets();
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(grid_flavor) {
  OperatorNodeFactory::instance()->register_factory(
      "grid_flavor_dem", make_compatible_operator<InitGridFlavorNode<GridFromFieldSet<DEMFieldSet>>>);
}

ONIKA_AUTORUN_INIT(grid_flavor_fragmentation) {
  OperatorNodeFactory::instance()->register_factory(
      "grid_flavor_dem_fragmentation",
      make_compatible_operator<InitGridFlavorNode<GridFromFieldSet<FragmentationDEMFieldSet>>>);
}

}  // namespace exaDEM
