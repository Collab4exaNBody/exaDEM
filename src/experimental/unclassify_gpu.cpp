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

#define DEBUG_NBH_GPU 1

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>

#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>

#include <cassert>

#include <exaDEM/traversal.hpp>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/interaction_manager.hpp>
#include <exaDEM/interaction/migration_test.hpp>
#include <exaDEM/classifier/classifier_transfert.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_utils.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_storage.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_cell_data.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_interaction_history.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_manager.hpp>

namespace exaDEM {
template <typename GridT, class = AssertGridHasFields<GridT>>
class UnclassifyGPU : public OperatorNode {

  ADD_SLOT(Domain, domain, INPUT, REQUIRED);
  ADD_SLOT(GridCellParticleInteraction, ges, INPUT_OUTPUT, DocString{"Interaction list"});
  ADD_SLOT(Traversal, traversal_real, INPUT, REQUIRED, DocString{"list of non empty cells within the current grid"});
  ADD_SLOT(Classifier, ic, INPUT, DocString{"Interaction lists classified according to their types"});
  ADD_SLOT(NBHManager, nbh_manager, INPUT, DocString{"Data about packed interactions within classifier."});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This function

        YAML example [no option]:

          - unclassify_gpu
       )EOF";
  }

  inline void execute() final {
    //lout << "unclassify active interaction on GridCellParticleInteraction" << std::endl;
    classify_interaction_grid(*ic, *traversal_real, *nbh_manager, *ges);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(unclassify_gpu) {
  OperatorNodeFactory::instance()->register_factory("unclassify_gpu",
                                                    make_grid_variant_operator<UnclassifyGPU>);
}
}  // namespace exaDEM
