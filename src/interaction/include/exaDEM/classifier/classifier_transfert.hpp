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

#pragma once

#include <exaDEM/classifier/classifier.hpp>

namespace exaDEM {
/**
 * @brief Classifies interactions into categorized waves based on their types.
 *
 * This function categorizes interactions into different waves based on their types,
 * utilizing the `waves` vector in the `Classifier` struct. It resets existing waves,
 * calculates the number of interactions per wave, and then stores interactions
 * accordingly.
 *
 * @param ges Reference to the GridCellParticleInteraction object containing interactions to classify.
 */
inline void classify(Classifier& classifier, GridCellParticleInteraction& ges, size_t* idxs, size_t size) {
  using namespace onika::cuda;
  constexpr int ntypes = InteractionTypeId::NTypes;

  classifier.reset_containers();  // Clear existing waves
  auto& ces = ges.m_data;    // Reference to cells containing interactions

  size_t n_threads;
#  pragma omp parallel
  {
    n_threads = omp_get_num_threads();
  }

  std::vector<std::array<std::pair<size_t, size_t>, ntypes> > bounds;
  bounds.resize(n_threads);

# pragma omp parallel
  {
    size_t threads = omp_get_thread_num();
    ///< Storage for interactions categorized by type.
    std::array<std::vector<exaDEM::PlaceholderInteraction>, ntypes> tmp;

#   pragma omp for schedule(static) nowait
    for (size_t c = 0; c < size; c++) {
      auto& interactions = ces[idxs[c]];
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

    for (int typeID = 0; typeID < ntypes; typeID++) {
      bounds[threads][typeID].second = tmp[typeID].size();
    }

#   pragma omp barrier

    // All
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
      size_t size = bounds[n_threads - 1][typeID].first + bounds[n_threads - 1][typeID].second;
      classifier.resize(typeID, size);
    }
#   pragma omp barrier

    // All
    for (int typeID = 0; typeID < ntypes; typeID++) {
      classifier.copy(typeID,
                  bound[typeID].first,
                  bound[typeID].second,
                  tmp[typeID]);
    }
  }
}

struct UnclassifyFunc {
  template <InteractionType IT>
  void operator()(ClassifierContainer<IT>& container, GridCellParticleInteraction& ges) {
    using namespace onika::cuda;
    auto& ces = ges.m_data;
# pragma omp for schedule(guided) nowait
    for (size_t it = 0; it < container.size(); it++) {
      auto item1 = container[it];
      if (item1.active())
      {
        auto& celli = ces[item1.pair.owner().cell];
        const unsigned int ni = vector_size(celli.m_data);
        PlaceholderInteraction* __restrict__ data_i_ptr = vector_data(celli.m_data);

        // Binary search 
        // belonging to the same owner particle
        uint16_t owner_p = item1.pair.owner().p;

        // lower_bound
        size_t lo = 0, hi = ni;
        while (lo < hi) {
          size_t mid = lo + (hi - lo) / 2;
          if (data_i_ptr[mid].owner().p < owner_p) lo = mid + 1;
          else hi = mid;
        }
        size_t start = lo;

        // upper_bound
        hi = ni;
        while (lo < hi) {
          size_t mid = lo + (hi - lo) / 2;
          if (data_i_ptr[mid].owner().p <= owner_p) lo = mid + 1;
          else hi = mid;
        }
        size_t end = lo;

        // Linear search 
        bool find = false;
        for (size_t it2 = start; it2 < end; it2++) {
          auto& item2 = (data_i_ptr[it2]).convert<IT>();
          if (item1 == item2) {
            item2.update(item1);
            find = true;
            break;
          }
        }

        if (!find) {
          item1.print();
          color_log::error("unclassify", "One active interaction has not been updated");
        }
      }
    }
  }
};

/**
 * @brief Restores friction and moment data for interactions from categorized waves to cell interactions.
 *
 * This function restores friction and moment data from categorized waves back to their corresponding
 * interactions in cell data (`ges.m_data`). It iterates through each wave, retrieves interactions
 * with non-zero friction and moment from the wave, and updates the corresponding interaction in the
 * cell data.
 *
 * @param ges Reference to the GridCellParticleInteraction object containing cell interactions.
 */
inline void unclassify(Classifier& classifier, GridCellParticleInteraction& ges) {
  using namespace onika::cuda;

  UnclassifyFunc func;
  for (int typeID = 0; typeID < InteractionTypeId::NTypes; typeID++) {
    CDispatcher::dispatch(typeID, classifier, func, ges);
  }
}
}  // namespace exaDEM
