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

#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/classifier/classifier_transfert.hpp>

namespace exaDEM {
class UnclassifyInteractions : public OperatorNode {
  // attributes processed during computation
  using ComputeFields = FieldSet<field::_vrot, field::_arot>;
  static constexpr ComputeFields compute_field_set{};

  ADD_SLOT(GridCellParticleInteraction, ges, INPUT_OUTPUT, REQUIRED, DocString{"Interaction list"});
  ADD_SLOT(Classifier, ic, INPUT, DocString{"Interaction lists classified according to their types"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator copies data from the Interaction Classifier to the GridCellParticleInteraction.

        YAML example [no option]:

          - unclassify_interactions
        )EOF";
  }

  inline void execute() final {
    if (!ic.has_value()) {
      return;
    }
    unclassify(*ic, *ges);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(unclassify_interactions) {
  OperatorNodeFactory::instance()->register_factory("unclassify_interactions",
                                                    make_simple_operator<UnclassifyInteractions>);
}
}  // namespace exaDEM
