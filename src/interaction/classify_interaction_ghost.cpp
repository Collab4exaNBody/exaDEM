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
#include <mpi.h>

// exaDEM
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/traversal.hpp>
#include <exaDEM/classifier/classifier.hpp>

namespace exaDEM {
template <typename GridT, class = AssertGridHasFields<GridT>>
class ClassificationInteractionGhost : public OperatorNode {
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(GridCellParticleInteraction, ges, INPUT_OUTPUT, DocString{"Interaction list"});
  ADD_SLOT(Classifier, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
       This operator ...
       )EOF";
  }

  inline void execute() final {
    constexpr int ntypes = InteractionTypeId::NTypes;
    using InteractionBuffer  = std::vector<PlaceholderInteraction>;
    using InteractionBuffers = std::array<InteractionBuffer, ntypes>;

    // Get slots
    auto& interaction_cells = ges->m_data;
    auto& classifier = *ic;
    auto& g = *grid;

    // Used by the classifier dispatcher
    ClassifierContainerResizerFunc resize;
    ClassifierContainerCopierFunc interaction_copier;

    size_t n_threads;
#  pragma omp parallel
    {
      n_threads = omp_get_num_threads();
    }

    InteractionBuffers buffer;
    std::vector<std::array<std::pair<size_t, size_t>, ntypes> > bounds;
    bounds.resize(n_threads);


    // Reset Interaction within the grid ghost layer
#pragma omp parallel
    {
      size_t threads = omp_get_thread_num();
      InteractionBuffers TmpBuff;
#     pragma omp for nowait
      for (size_t i = 0; i < g.number_of_cells(); i++) {
        if (!g.is_ghost_cell(i)) {
          continue;
        }
        CellExtraDynamicDataStorageT<PlaceholderInteraction>& storage = interaction_cells[i];
        auto& data = storage.m_data;
        for (size_t i = 0 ; i < data.size() ; i++) {
          TmpBuff[data[i].type()].push_back(data[i]);
        }
      }

      for (int typeID = 0 ; typeID < ntypes; typeID++) {
        bounds[threads][typeID].second = TmpBuff[typeID].size();
      }

#   pragma omp barrier

      auto& bound = bounds[threads];
      for (int typeID = 0; typeID < ntypes; typeID++) {
        size_t start = 0;
        for (size_t i = 0; i < threads; i++) {
          start += bounds[i][typeID].second;
        }
        bound[typeID].first = start;
      }

#   pragma omp barrier
      // Partial
#   pragma omp for
      for (int typeID = 0; typeID < ntypes; typeID++) {
	size_t size = bounds[n_threads-1][typeID].first + bounds[n_threads-1][typeID].second;
	CDispatcher::dispatch(typeID, classifier, resize, size);
      }

#   pragma omp barrier
      // All
      for (int typeID = 0; typeID < ntypes; typeID++) {
lout << "typeID " << typeID << std::endl;
	size_t start = bound[typeID].first;
	size_t size = bound[typeID].second;
	CDispatcher::dispatch(typeID, classifier, interaction_copier,
	    TmpBuff[typeID], start, size, typeID);
      }
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(classify_interaction_ghost) {
  OperatorNodeFactory::instance()->register_factory("classify_interaction_ghost",
      make_grid_variant_operator<ClassificationInteractionGhost>);
}
}  // namespace exaDEM
