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

#include <exanb/core/domain.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>
#include <onika/parallel/parallel_for.h>

#include <cassert>
#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/interaction_list_layout.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_cell_data.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_interaction_history.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_storage.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_utils.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/interaction_manager.hpp>
#include <exaDEM/interaction/migration_test.hpp>
#include <exaDEM/traversal.hpp>

namespace exaDEM {
using namespace exanb;

/** @brief Copies one PP type of the classifier tail into a contiguous buffer. */
struct CopyPPTailFunc {
  ClassifierContainer<InteractionType::ParticleParticle>::View view;
  size_t base;
  PlaceholderInteraction* out;
  ONIKA_HOST_DEVICE_FUNC inline void operator()(size_t i) const { out[i] = view(base + i); }
};
}  // namespace exaDEM

namespace onika {
namespace parallel {
template <>
struct ParallelForFunctorTraits<exaDEM::CopyPPTailFunc> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika

namespace exaDEM {
template <typename GridT, class = AssertGridHasFields<GridT>>
class UnclassifyGPU : public OperatorNode {
  ADD_SLOT(Domain, domain, INPUT, REQUIRED);
  ADD_SLOT(GridCellParticleInteraction, ges, INPUT_OUTPUT, DocString{"Interaction list"});
  ADD_SLOT(Traversal, traversal_real, INPUT, REQUIRED, DocString{"list of non empty cells within the current grid"});
  ADD_SLOT(Classifier, ic, INPUT, DocString{"Interaction lists classified according to their types"});
  ADD_SLOT(InteractionListBuildLayout, interaction_list_layout, INPUT,
           DocString{"Data about packed interactions within classifier."});
  ADD_SLOT(onika::memory::CudaMMVector<PlaceholderInteraction>, tail_buffer, PRIVATE);
  ADD_SLOT(PPTailByCell, pp_tail, PRIVATE);

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator transfers the active interactions held by the GPU interaction Classifier (ic)
        back into the GridCellParticleInteraction storage (ges), PP tail included (copied on the GPU).

        YAML example [no option]:

          - unclassify_gpu
       )EOF";
  }

  inline void execute() final {
    gather_pp_tail();
    classify_interaction_grid(*ic, *traversal_real, *interaction_list_layout, *ges, &(*pp_tail));
  }

 private:
  /** Active PP tail interactions grouped by owner cell: one contiguous GPU copy, grouping on the host. */
  void gather_pp_tail() {
    PPTailByCell& tail = *pp_tail;
    tail.items.clear();
    tail.cells.clear();
    tail.begin.clear();
    if (!ic->has_pp_tail_) {
      return;
    }

    size_t total = 0;
    for (int typeID = InteractionTypeId::FirstIdParticle; typeID <= InteractionTypeId::LastIdParticle; typeID++) {
      total += ic->get_size(typeID) - ic->base_size(typeID);
    }
    auto& buffer = *tail_buffer;
    buffer.resize(total);
    size_t shift = 0;
    for (int typeID = InteractionTypeId::FirstIdParticle; typeID <= InteractionTypeId::LastIdParticle; typeID++) {
      const size_t base = ic->base_size(typeID);
      const size_t n = ic->get_size(typeID) - base;
      if (n > 0) {
        CopyPPTailFunc func = {ic->get_data<InteractionType::ParticleParticle>(typeID).view(), base,
                               buffer.data() + shift};
        parallel_for(n, func, parallel_execution_context("copy_pp_tail"));
        shift += n;
      }
    }
    ONIKA_CU_DEVICE_SYNCHRONIZE();

    std::vector<std::pair<uint32_t, uint32_t>> order;  // (owner cell, index in buffer)
    for (size_t i = 0; i < buffer.size(); i++) {
      if (buffer[i].active()) {
        order.push_back({buffer[i].owner().cell_, static_cast<uint32_t>(i)});
      }
    }
    std::stable_sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    tail.items.resize(order.size());
    for (size_t k = 0; k < order.size(); k++) {
      tail.items[k] = buffer[order[k].second];
      if (k == 0 || order[k].first != order[k - 1].first) {
        tail.cells.push_back(order[k].first);
        tail.begin.push_back(k);
      }
    }
    tail.begin.push_back(order.size());
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(unclassify_gpu) {
  OperatorNodeFactory::instance()->register_factory("unclassify_gpu", make_grid_variant_operator<UnclassifyGPU>);
}
}  // namespace exaDEM
