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
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <algorithm>
#include <array>
#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/classifier/classifier_pp_tail.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <limits>

namespace exaDEM {
using namespace exanb;

/** @brief Function to unclassify interactions */
struct UnclassifyFunc {
  size_t size_limit_ = std::numeric_limits<size_t>::max();

  template <InteractionType IT>
  void operator()(ClassifierContainer<IT>& container, GridCellParticleInteraction& ges) {
    using namespace onika::cuda;
    auto& ces = ges.m_data;
    constexpr uint32_t InvalidCell = std::numeric_limits<uint32_t>::max();
    constexpr uint16_t InvalidP = std::numeric_limits<uint16_t>::max();
    uint32_t previous_cell = InvalidCell;
    uint16_t previous_owner_p = InvalidP;
    size_t previous_start = 0, previous_end = 0;
    PlaceholderInteraction* previous_data_i_ptr = nullptr;
    const size_t n_interactions = std::min(container.size(), size_limit_);
#pragma omp for schedule(guided) nowait
    for (size_t it = 0; it < n_interactions; it++) {
      auto item1 = container[it];
      assert(item1.pair_.consistent());  // Check if the interaction is correctly formed before updating
      // Only active interactions should be updated.
      // If an interaction is not active, it means it has been removed / recomputed (nbh operators).
      if (item1.active()) {
        const uint32_t cell_i = item1.pair_.owner().cell_;
        // belonging to the same owner particle
        const uint16_t owner_p = item1.pair_.owner().p_;

        size_t start, end;
        PlaceholderInteraction* __restrict__ data_i_ptr;
        if (cell_i == previous_cell && owner_p == previous_owner_p) {
          start = previous_start;
          end = previous_end;
          data_i_ptr = previous_data_i_ptr;
        } else {
          auto& celli = ces[cell_i];
          const unsigned int ni = vector_size(celli.m_data);
          data_i_ptr = vector_data(celli.m_data);

          // Binary search
          // lower_bound
          size_t lo = 0, hi = ni;
          while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (data_i_ptr[mid].owner().p_ < owner_p) {
              lo = mid + 1;
            } else
              hi = mid;
          }
          start = lo;

          // upper_bound
          hi = ni;
          while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (data_i_ptr[mid].owner().p_ <= owner_p) {
              lo = mid + 1;
            } else {
              hi = mid;
            }
          }
          end = lo;

          previous_cell = cell_i;
          previous_owner_p = owner_p;
          previous_start = start;
          previous_end = end;
          previous_data_i_ptr = data_i_ptr;
        }

        // Linear search
        bool found = false;
        for (size_t it2 = start; it2 < end; it2++) {
          auto& item2 = (data_i_ptr[it2]).convert<IT>();
          assert(item2.pair_.consistent());  // Check if the interaction is correctly formed before updating
          // Only one interaction should match.
          if (item1 == item2) {
            item2.update(item1);
            // DIAGNOSTIC: item2's type_ must still be a valid interaction type right after this write.
            if (item2.type() >= InteractionTypeId::NTypesParticleParticle &&
                item2.type() != InteractionTypeId::InnerBondId) {
              item2.print();
              color_log::error(
                  "unclassify",
                  "Interaction type became invalid right after unclassify write: " + std::to_string(item2.type()) +
                      " (cell: " + std::to_string(item1.pair_.owner().cell_) + ", it2: " + std::to_string(it2) + ")");
            }
            found = true;
            break;
          }
        }

        // If no matching interaction is found.
        // This should not happen.
        if (!found) {
          item1.print();
          color_log::error("unclassify", "One active interaction has not been updated");
        }
      }
    }
  }
};

template <>
inline void UnclassifyFunc::operator()<InteractionType::InnerBond>(
    ClassifierContainer<InteractionType::InnerBond>& container, GridCellParticleInteraction& ges) {
  using namespace onika::cuda;
  auto& ces = ges.m_data;  // Reference to cells containing interactions
                           // Parallel loop to process interactions within a wave
  // Cache of the (cell, owner_p) range located for the previous interaction (see the generic
  // operator() above for the rationale).
  constexpr uint32_t InvalidCell = std::numeric_limits<uint32_t>::max();
  constexpr uint16_t InvalidP = std::numeric_limits<uint16_t>::max();
  uint32_t previous_cell = InvalidCell;
  uint16_t previous_owner_p = InvalidP;
  size_t previous_start = 0, previous_end = 0;
  PlaceholderInteraction* previous_data_i_ptr = nullptr;
#pragma omp for schedule(guided) nowait
  for (size_t it = 0; it < container.size(); it++) {
    auto item1 = container[it];
    // Check if the interaction is correctly formed before updating
    if (!item1.pair_.consistent()) {
      item1.print();
      color_log::error("unclassify", "One active interaction is illformed in the classified wave");
    }
    const uint32_t cell_i = item1.pair_.owner().cell_;
    const uint16_t owner_p = item1.pair_.owner().p_;

    size_t start, end;
    PlaceholderInteraction* __restrict__ data_i_ptr;
    if (cell_i == previous_cell && owner_p == previous_owner_p) {
      start = previous_start;
      end = previous_end;
      data_i_ptr = previous_data_i_ptr;
    } else {
      auto& celli = ces[cell_i];
      const unsigned int ni = vector_size(celli.m_data);
      data_i_ptr = vector_data(celli.m_data);

      // Binary search: narrow down to the entries sharing the same owner particle before
      // scanning for the exact (owner, partner) pair match, instead of scanning the whole cell
      // (which mixes every interaction type, not just inner bonds).
      size_t lo = 0, hi = ni;
      while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (data_i_ptr[mid].owner().p_ < owner_p) {
          lo = mid + 1;
        } else
          hi = mid;
      }
      start = lo;

      hi = ni;
      while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (data_i_ptr[mid].owner().p_ <= owner_p) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      end = lo;

      previous_cell = cell_i;
      previous_owner_p = owner_p;
      previous_start = start;
      previous_end = end;
      previous_data_i_ptr = data_i_ptr;
    }

    // Iterate through interactions sharing the same owner particle to find the matching one.
    bool found = false;
    for (size_t it2 = start; it2 < end; it2++) {
      PlaceholderInteraction& item2 = data_i_ptr[it2];
      // Only one interaction should match.
      if (item1.pair_ == item2.pair_) {
        found = true;
        // Check if the interaction is correctly formed before updating
        if (!item2.consistent()) {
          item1.print();
          color_log::error("unclassify", "One active interaction is illformed");
        }
        // If the interaction is persistent, we can directly update it. Otherwise, we need to transform it back to a
        // particle-particle interaction.
        if (item1.persistent()) {
          auto& view = item2.convert<InteractionType::InnerBond>();
          view.update(item1);
        } else {  // transform an innerbond interaction to a vertex-vertex interaction
          auto& view = item2.convert<InteractionType::ParticleParticle>();
          // item1 is an inner bond, broke_interaction transforms it to a vertex-vertex interaction.
          view = broke_interaction(item1);
        }
        break;
      }
    }

    // If no matching interaction is found.
    // This should not happen, as all interactions in the wave should have a corresponding interaction in the cell data.
    if (!found) {
      item1.print();
      color_log::error("unclassify", "One active interaction has not been updated");
    }
  }
}

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

        This operator restores friction and moment data from categorized waves back to their corresponding
        interactions in cell data. It iterates through each wave, retrieves active interactions
        and updates the corresponding interaction in the cell data.

        YAML example [no option]:

          - unclassify_interactions
        )EOF";
  }

  inline void execute() final {
    using namespace onika::cuda;
    // If the classifier is not initialized, there are no interactions to unclassify, so we can return early.
    if (!ic.has_value()) {
      return;
    }

    Classifier& classifier = *ic;
    GridCellParticleInteraction& grid_interactions = *ges;  // Reference to cell interactions
    std::array<size_t, InteractionTypeId::NTypes> size_limits;
    for (int typeID = 0; typeID < InteractionTypeId::NTypes; typeID++) {
      size_limits[typeID] = classifier.base_size(typeID);
    }
#pragma omp parallel
    {
      UnclassifyFunc func;
      for (int typeID = 0; typeID < InteractionTypeId::NTypes; typeID++) {
        func.size_limit_ = size_limits[typeID];
        CDispatcher::dispatch(typeID, classifier, func, grid_interactions);
      }
    }

    // Contacts appended since the last rebuild (broken interfaces) are merged into the grid storage.
    merge_pp_tail_into_grid(classifier, grid_interactions);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(unclassify_interactions) {
  OperatorNodeFactory::instance()->register_factory("unclassify_interactions",
                                                    make_simple_operator<UnclassifyInteractions>);
}
}  // namespace exaDEM
