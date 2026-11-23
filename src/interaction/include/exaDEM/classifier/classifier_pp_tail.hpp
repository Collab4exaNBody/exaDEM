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

#include <onika/cuda/cuda_context.h>

#include <algorithm>
#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <map>
#include <vector>

namespace exaDEM {

/**
 * @brief Merges the particle-particle tail of the classifier (interactions appended since the last neighbor-list
 * rebuild, see Classifier::append_pp_tail) into the GridCellParticleInteraction storage.
 */
inline void merge_pp_tail_into_grid(Classifier& classifier, GridCellParticleInteraction& ges) {
  if (!classifier.has_pp_tail_) {
    return;
  }

  ONIKA_CU_DEVICE_SYNCHRONIZE();

  std::map<size_t, std::vector<PlaceholderInteraction>> per_cell;
  for (int typeID = InteractionTypeId::FirstIdParticle; typeID <= InteractionTypeId::LastIdParticle; typeID++) {
    auto& container = classifier.get_data<InteractionType::ParticleParticle>(typeID);
    const size_t begin = classifier.base_size(typeID);
    for (size_t idx = begin; idx < container.size(); idx++) {
      PlaceholderInteraction item;
      item.clear_placeholder();
      item.convert<InteractionType::ParticleParticle>() = container[idx];
      if (!item.active()) {
        continue;
      }
      assert(item.pair_.ghost_ == InteractionPair::NotGhost);
      per_cell[item.owner().cell_].push_back(item);
    }
  }

  auto& cells = ges.m_data;
  for (auto& [cell, items] : per_cell) {
    assert(cell < cells.size());
    auto& storage = cells[cell];
    auto& data = storage.m_data;
    auto& info = storage.m_info;

    std::vector<PlaceholderInteraction> to_insert;
    for (auto& item : items) {
      const size_t p = item.owner().p_;
      bool found = false;
      if (p < info.size()) {
        const size_t start = info[p].offset;
        const size_t end = start + info[p].size;
        for (size_t k = start; k < end && k < data.size(); k++) {
          if (data[k].pair_ == item.pair_) {
            data[k] = item;
            found = true;
            break;
          }
        }
      }
      if (!found) {
        to_insert.push_back(item);
      }
    }

    if (to_insert.empty()) {
      continue;
    }

    data.insert(data.end(), to_insert.begin(), to_insert.end());
    std::stable_sort(data.begin(), data.end(), [](const PlaceholderInteraction& a, const PlaceholderInteraction& b) {
      return a.sort_by_owner_p(b);
    });

    // reindex info
    size_t offset = 0;
    for (size_t p = 0; p < info.size(); p++) {
      info[p].offset = offset;
      size_t size = 0;
      while (offset + size < data.size() && data[offset + size].owner().p_ == p) {
        size++;
      }
      info[p].size = size;
      offset += size;
    }
    assert(offset == data.size());
  }
}
}  // namespace exaDEM
