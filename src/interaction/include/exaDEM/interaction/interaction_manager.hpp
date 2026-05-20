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

#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/interaction.hpp>

namespace exaDEM {
/** @brief A manager for handling interactions between particles.
 */
struct InteractionManager {
  std::vector<exaDEM::PlaceholderInteraction> hist =
      {};  // historical interactions, used to update interactions with history
  std::vector<std::vector<exaDEM::PlaceholderInteraction>> list =
      {};  // list of interactions per particle, used to update interactions without history
  std::vector<std::vector<uint64_t>> ignore = {};  // list of ignored interaction IDs per particle
  size_t current_cell_id;         // current cell id, used for consistency check when update persistent interactions
  size_t current_cell_particles;  // current number of particles in the cell, used for consistency check when update
                                  // persistent interactions

  /** @brief Reset the interaction manager.
   * This function clears all interactions and initializes the lists with the specified size.
   * [param size] The number of particles in the system.
   */
  void reset(const size_t size) {
    list.clear();
    ignore.clear();
    list.resize(size);
    ignore.resize(size);
    for (size_t p = 0; p < size; p++) {
      list[p].clear();
      ignore[p].clear();
    }
  }

  /** @brief Add an interaction to the manager.
   * This function adds the given interaction to the list corresponding to its owner particle index.
   * If the interaction is marked as ignored, it will not be added.
   * [param I] The PlaceholderInteraction to add.
   */
  void add_item(exaDEM::PlaceholderInteraction& I) {
    size_t p = I.owner().p;
    assert(p < list.size());
    if (!skip_ignored_interactions(p, I)) {
      list[p].push_back(I);
    }
  }

  /** @brief Add multiple interactions to the manager.
   * This function adds all interactions from the given vector to the list corresponding to their owner particle
   * indices. [param vec] The vector of PlaceholderInteractions to add.
   */
  void add(std::vector<exaDEM::PlaceholderInteraction>& vec) {
    for (auto& it : vec) {
      add_item(it);
    }
  }

  /** @brief Get the total number of interactions in the manager.
   * This function iterates over all lists and counts the total number of interactions.
   * [return] The total number of interactions.
   */
  size_t get_size() {
    size_t count(0);
    for (auto& it : list) {
      count += it.size();
    }
    return count;
  }

  /** @brief Update the extra storage with interactions from the manager.
   * This function updates the provided storage with interactions from the manager, optionally including historical
   * data. [param storage] The CellExtraDynamicDataStorageT to update.
   */
  template <bool use_history>
  void update_extra_storage(CellExtraDynamicDataStorageT<PlaceholderInteraction>& storage) {
    const size_t total_size = this->get_size();
    size_t offset = 0;
    auto& info = storage.m_info;
    auto& data = storage.m_data;

    data.resize(total_size);
    assert(data.size() == total_size);  // data correctly resized

    // Loop over each particle's interaction list and copy interactions to the storage data array.
    for (size_t p = 0; p < list.size(); p++) {
      info[p].offset = offset;
      // Optionally update interactions with history before copying to storage.
      if constexpr (use_history) {
        update(list[p], hist);
      }
      assert(offset + list[p].size() <= total_size);  // offset within bounds
      std::copy(list[p].begin(), list[p].end(), data.data() + offset);
      info[p].size = list[p].size();
      offset += list[p].size();
    }

    assert(offset == total_size);  // all data accounted for
  }

  /** @brief Update the ignore interactions list.
   * This function updates the ignore list for each particle based on the interactions in the manager.
   * Call it after update persistent interaction
   */
  void update_ignore_interaction() {
    size_t n = list.size();
    ignore.resize(n);
    // Loop over all particles (related to a cell).
    for (size_t p = 0; p < n; p++) {
      auto& interactions = list[p];
      // Loop over interactions of particle at position p to find those that should be ignored.
      for (size_t i = 0; i < interactions.size(); i++) {
        if (interactions[i].ignore_other_interactions()) {
          auto& partner = interactions[i].j();
          auto& ignore_ids = ignore[p];
          bool add_info = true;
          // Check if the partner id is already in the ignore list to avoid duplicates.
          for (size_t j = 0; j < ignore_ids.size(); j++) {
            if (partner.id == ignore_ids[j]) {
              add_info = false;
              break;
            }
          }
          if (add_info) {
            ignore_ids.push_back(partner.id);
          }
        }
      }
    }
  }

  /** @brief Check if an interaction should be skipped due to being ignored.
   * This function checks if the given interaction is marked as ignored based on the ignore list for its owner particle.
   * [param p] The particle index.
   * [param I] The PlaceholderInteraction to check.
   * [return] True if the interaction should be skipped, false otherwise.
   */
  bool skip_ignored_interactions(size_t p, exaDEM::PlaceholderInteraction& I) {
    auto& partner = I.pair.partner();
    auto& ignore_ids = ignore[p];
    for (size_t j = 0; j < ignore_ids.size(); j++) {
      if (partner.id == ignore_ids[j]) {
        return true;
      }
    }
    return false;
  }
};

/** @brief Update persistent interactions in the manager list from the storage data.
 * This function iterates over the interactions stored in the provided storage, identifies those that are marked as
 * persistent. [param manager] The InteractionManager instance to update with persistent interactions. [param storage]
 * The CellExtraDynamicDataStorageT containing PlaceholderInteraction data to extract persistent interactions from.
 */
inline void update_persistent_interactions(InteractionManager& manager,
                                           CellExtraDynamicDataStorageT<PlaceholderInteraction>& storage) {
  size_t n_interactions = storage.m_data.size();
  for (size_t i = 0; i < n_interactions; i++) {
    PlaceholderInteraction& I = storage.m_data[i];
    if (I.persistent()) {
      if (I.pair.owner().cell != manager.current_cell_id) {
        color_log::mpi_error(
            "update_persistent_interactions",
            "This interaction is illformed, owner.cell should be: " + std::to_string(manager.current_cell_id) +
                " cell: " + std::to_string(I.pair.owner().cell) + " p: " + std::to_string(I.pair.owner().p) +
                " id: " + std::to_string(I.pair.owner().id) + " type: " + std::to_string(I.pair.type));
      }
      if (I.pair.owner().p >= manager.current_cell_particles) {
        color_log::mpi_error("update_persistent_interactions",
                             "This interaction is illformed, owner.p should be inferior to: " +
                                 std::to_string(manager.current_cell_particles) +
                                 " cell: " + std::to_string(I.pair.owner().cell) + " p; " +
                                 std::to_string(I.pair.owner().p) + " id: " + std::to_string(I.pair.owner().id));
      }
      assert(I.pair.owner().p < manager.list.size());
      manager.list[I.pair.owner().p].push_back(I);
    }
  }

  // Keep interactions sorted for interfaces (contiguous interactions).
  for (size_t p = 0; p < manager.list.size(); p++) {
    std::stable_sort(manager.list[p].begin(), manager.list[p].end());
  }
}

/* @brief Extract historical interactions from the provided data.
 * This function iterates over the given array of interactions and copies the active ones to the local vector.
 * [param local] The vector to store the extracted interactions.
 * [param data] The array of PlaceholderInteractions to extract from.
 * [param size] The number of interactions in the data array.
 */
inline void extract_history(std::vector<PlaceholderInteraction>& local,
                            const PlaceholderInteraction* __restrict__ const data, const unsigned int size) {
  local.clear();
  for (size_t i = 0; i < size; i++) {
    const auto& item = data[i];
    if (item.persistent()) {
      continue;  // skip persistent interactions, they are already stored in the manager list latter;
    }
    if (item.active()) {
      local.push_back(item);  // copy active interactions in the history
    }
  }
}

/* @brief Check if the face defined by the interactions of a particle is well formed.
 * [param interactions] The vector of PlaceholderInteractions to check.
 * [param p] The index of the particle whose interactions are being checked.
 * [param pvv] The ParticleVertexView providing access to vertex positions.
 * [return] True if the face is well formed, false if it is sticked or illformed.
 */
template <typename ParticleVertexViewT>
bool check_stiked_face(std::vector<exaDEM::PlaceholderInteraction>& interactions,
                       size_t p,  // particle index
                       ParticleVertexViewT& pvv) {
  std::vector<int> vertex_id = {};
  std::vector<Vec3d> vertices;
  // identify vertices
  for (size_t i = 0; i < interactions.size(); i++) {
    if (interactions[i].type() == InteractionTypeId::InnerBond) {
      vertex_id.push_back(interactions[i].pair.pi.sub);
    }
  }

  if (vertex_id.size() == 0) {
    return true;  // true cause no stiked face
  }
  vertices.resize(vertex_id.size());

  for (size_t i = 0; i < vertex_id.size(); i++) {
    vertices[i] = pvv[vertex_id[i]];
  }

  if (vertices.size() <= 2) {
    color_log::warning("interaction_manager::check_stiked_face",
                       "sticked face is illformed (n_vertices should be >= 3) with only " +
                           std::to_string(vertices.size()) + " vertices");
    return false;
  }

  Vec3d va = vertices[1] - vertices[0];
  Vec3d vb = vertices[2] - vertices[0];

  bool res = true;

  Vec3d normal = exanb::cross(va, vb);
  constexpr double tol = 1.e-10;  // tolerance for coplanarity check.

  // Check that remaining vertices belong to the same plane
  for (size_t j = 3; j < vertices.size(); j++) {
    Vec3d vj = vertices[j] - vertices[0];
    double distance_to_plane = std::abs(exanb::dot(vj, normal));

    if (distance_to_plane > tol) {
      color_log::warning("interaction_manager::check_sticked_face",
                         "Sticked face is not coplanar: vertex " + std::to_string(j) +
                             " is out of the plane defined by the first three vertices. " +
                             "Distance to plane = " + std::to_string(distance_to_plane));
      return false;
    }
  }
  return res;
}
}  // namespace exaDEM
