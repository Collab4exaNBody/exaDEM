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
#include <exanb/core/grid.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/traversal.hpp>

namespace exaDEM {
class ClassifyInteractions : public OperatorNode {
  // attributes processed during computation
  using ComputeFields = FieldSet<field::_vrot, field::_arot>;
  static constexpr ComputeFields compute_field_set{};

  ADD_SLOT(GridCellParticleInteraction, ges, INPUT, REQUIRED, DocString{"Interaction list"});
  ADD_SLOT(Classifier, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
  ADD_SLOT(Traversal, traversal_all, INPUT, REQUIRED, DocString{"list of non empty cells within the current grid"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator copies interactions from GridCellParticleInteraction to the Interaction Classifier.
        This operator categorizes interactions into different \"waves\" based on their types,
        utilizing the `waves` vector in the `Classifier` struct. It resets existing waves,
        calculates the number of interactions per wave, and then stores interactions accordingly.

        YAML example [no option]:

          - classify_interactions
      )EOF";
  }

  inline void execute() final {
    // Note: Interactions are copied in a temporary array before being copied to the classifier. This is because the
    // number of interactions in each wave is not known beforehand, and copying interactions one by one to the
    // classifier would be inefficient. This operator could be optimized by directly copying interactions to the
    // classifier without using a temporary array.

    using namespace onika::cuda;
    constexpr int ntypes = InteractionTypeId::NTypes;

    auto [cell_ptr, cell_size] = traversal_all->info();
    if (!ic.has_value()) {
      ic->initialize();
    }

    Classifier& classifier = *ic;
    auto& ces = ges->m_data;  // Reference to cells containing interactions

    classifier.reset_containers();  // Clear existing waves

    size_t n_threads;
#pragma omp parallel
    {
      n_threads = omp_get_num_threads();
    }

    // Storage for the starting index and size of each wave for each thread.
    std::vector<std::array<std::pair<size_t, size_t>, ntypes> > bounds;
    // Initialize bounds for each thread and interaction type.
    bounds.resize(n_threads);

#pragma omp parallel
    {
      size_t threads = omp_get_thread_num();
      ///< Storage for interactions categorized by type.
      std::array<std::vector<exaDEM::PlaceholderInteraction>, ntypes> tmp;

#pragma omp for schedule(static) nowait
      // Iterate through non-empty cells and categorize interactions into waves based on their types.
      for (size_t c = 0; c < cell_size; c++) {
        auto& interactions = ces[cell_ptr[c]];
        const unsigned int n_interactions_in_cell = interactions.m_data.size();
        auto* const __restrict__ data_ptr = interactions.m_data.data();
        // Place interactions into their respective waves
        for (size_t it = 0; it < n_interactions_in_cell; it++) {
          auto& item = data_ptr[it];
          const int typeID = item.type();
          tmp[typeID].push_back(item);
          item.reset();
        }
      }

      // Get the size of each wave for the current thread.
      for (int typeID = 0; typeID < ntypes; typeID++) {
        bounds[threads][typeID].second = tmp[typeID].size();
      }

#pragma omp barrier

      auto& bound = bounds[threads];
      // calculate the starting index for each wave in the classifier based on the sizes of interaction types.
      for (int typeID = 0; typeID < ntypes; typeID++) {
        size_t start = 0;
        for (size_t i = 0; i < threads; i++) {
          start += bounds[i][typeID].second;
        }
        bound[typeID].first = start;
      }

#pragma omp barrier

#pragma omp for
      // resize classifiers to fit interactions
      for (int typeID = 0; typeID < ntypes; typeID++) {
        size_t size = bounds[n_threads - 1][typeID].first + bounds[n_threads - 1][typeID].second;
        classifier.resize(typeID, size);
      }
#pragma omp barrier

      // Copy interactions from the grid of interactions to the classifier.
      for (int typeID = 0; typeID < ntypes; typeID++) {
        classifier.copy(typeID, bound[typeID].first, bound[typeID].second, tmp[typeID]);
      }
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(classify_interactions) {
  OperatorNodeFactory::instance()->register_factory("classify_interactions",
                                                    make_simple_operator<ClassifyInteractions>);
}
}  // namespace exaDEM
