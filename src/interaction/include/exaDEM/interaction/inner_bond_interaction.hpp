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

#include <onika/math/basic_types.h>
#include <onika/math/basic_types_stream.h>

#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/interaction_decoration.hpp>
#include <exaDEM/interaction/interaction_pair.hpp>
#include <exaDEM/interface/rupture_criterion.hpp>
#include <iostream>

// Single source of truth for InnerBondInteraction's per-interaction fields.
// 'friction' reuses the tag already declared (NEW_TAG) by Interaction in interaction.hpp,
// hence SHARED_TAG here; it must keep the same tuple index (0) as there.
#define EXADEM_INNER_BOND_FIELDS(X)         \
  X(exanb::Vec3d, friction, 0, SHARED_TAG)  \
  X(double, en, 1, NEW_TAG)                 \
  X(exanb::Vec3d, tds, 2, NEW_TAG)          \
  X(double, et, 3, NEW_TAG)                 \
  X(double, dn0, 4, NEW_TAG)                \
  X(double, weight, 5, NEW_TAG)             \
  X(RuptureCriteria, criterion, 6, NEW_TAG) \
  X(uint8_t, unbroken, 7, NEW_TAG)

EXADEM_INTERACTION_FIELD_TAGS(EXADEM_INNER_BOND_FIELDS)

namespace exaDEM {
/**
 * @brief Structure representing an interaction in a Discrete Element Method (DEM) simulation.
 */
struct InnerBondInteraction {
  InteractionPair pair_; /**< The InteractionPair structure containing information about the interacting particles and
                           the type of interaction. */

  // specialized members, accessed through operator[] with the tags declared above, e.g.
  // It[attr::friction], It[attr::en], It[attr::tds], It[attr::et], It[attr::dn0],
  // It[attr::weight], It[attr::criterion], It[attr::unbroken]
  EXADEM_INTERACTION_TUPLE_TYPE(EXADEM_INNER_BOND_FIELDS)
  fm_ = {
      {0, 0, 0},  // friction
      0.0,        // en
      {0, 0, 0},  // tds
      0.0,        // et
      0.0,        // dn0
      1.0,        // weight: used for scaling forces and moments.
      {},         // criterion: used for determining bond breakage.
      true        // unbroken: whether the bond is unbroken (true) or broken (false).
  };

  EXADEM_INTERACTION_FIELD_ACCESSOR(EXADEM_INNER_BOND_FIELDS)

  /** @brief Get the first particle location.
   * [return] Reference to the first particle location.
   */
  ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& i() { return pair_.pi_; }
  ONIKA_HOST_DEVICE_FUNC inline const ParticleSubLocation& i() const { return pair_.pi_; }

  /** @brief Get the second particle location.
   * [return] Reference to the second particle location.
   */
  ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& j() { return pair_.pj_; }
  ONIKA_HOST_DEVICE_FUNC inline const ParticleSubLocation& j() const { return pair_.pj_; }

  /** @brief Get the type of the interaction.
   * [return] The type of the interaction.
   */
  ONIKA_HOST_DEVICE_FUNC inline uint16_t type() { return pair_.type_; }
  ONIKA_HOST_DEVICE_FUNC inline uint16_t type() const { return pair_.type_; }

  /** @brief Get the cell index associated with the interaction.
   * [return] The cell index.
   */
  ONIKA_HOST_DEVICE_FUNC inline uint32_t cell() { return pair_.owner().cell_; }

  /** @brief Get the pair information associated with the interaction.
   * [return] Reference to the interaction pair.
   */
  ONIKA_HOST_DEVICE_FUNC inline InteractionPair& pair_info() { return pair_; }
  ONIKA_HOST_DEVICE_FUNC inline const InteractionPair& pair_info() const { return pair_; }

  /** @brief Resets the Interaction structure by setting friction and moment vectors to zero.
   */
  ONIKA_HOST_DEVICE_FUNC void reset() {
    constexpr exanb::Vec3d null = {0, 0, 0};
    (*this)[attr::friction] = null;
    (*this)[attr::dn0] = 0;
    (*this)[attr::en] = 0;
    (*this)[attr::tds] = null;
    (*this)[attr::et] = 0;
    (*this)[attr::unbroken] = true;
    // weight and criteria are not reset
  }

  /** @brief Checks if the interaction is active.
   * [return] True if the interaction is active (moment vector or friction vector is non-zero), false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool active() const {
    if (!pair_.active()) {
      return false;
    }
    return true;
  }

  /** @brief Defines whether an interaction will be reconstructed or not.
   * [return] True if the interaction is persistent, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool persistent() { return (*this)[attr::unbroken]; }

  /** @brief Defines whether an interaction will be reconstructed or not.
   * [return] True if the interaction is persistent, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool persistent() const { return (*this)[attr::unbroken]; }

  /** @brief Defines whether to skip other interactions.
   * [return] True.
   */
  ONIKA_HOST_DEVICE_FUNC bool ignore_other_interactions() { return true; }

  /**
   * @brief Displays the Interaction data.
   */
  void print() {
    pair_.print();
    std::cout << "Friction: " << (*this)[attr::friction] << ", en: " << (*this)[attr::en]
              << ", tds: " << (*this)[attr::tds] << ", et: " << (*this)[attr::et] << ", dn0: " << (*this)[attr::dn0]
              << ", weight: " << (*this)[attr::weight] << ", criterion: " << (*this)[attr::criterion] << ")"
              << std::endl;
  }

  /**
   * @brief Displays the Interaction data.
   */
  void print() const {
    pair_.print();
    std::cout << "Friction: " << (*this)[attr::friction] << ", en: " << (*this)[attr::en]
              << ", tds: " << (*this)[attr::tds] << ", et: " << (*this)[attr::et] << ", dn0: " << (*this)[attr::dn0]
              << ", weight: " << (*this)[attr::weight] << ", criterion: " << (*this)[attr::criterion] << ")"
              << std::endl;
  }

  /** @brief Updates the interaction with values from another interaction.
   * @param I The interaction to update from.
   * [return] None.
   */
  ONIKA_HOST_DEVICE_FUNC void update(InnerBondInteraction& I) { this->fm_ = I.fm_; }

  /** @brief Checks if two interactions are equal.
   * @param I The interaction to compare with.
   * [return] True if the interactions are equal, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool operator==(InnerBondInteraction& I) { return (pair_ == I.pair_); }

  /** @brief Checks if two interactions are equal.
   * @param I The interaction to compare with.
   * [return] True if the interactions are equal, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool operator==(const InnerBondInteraction& I) const { return (pair_ == I.pair_); }

  /** @brief Checks if this interaction is less than another interaction.
   * @param I The interaction to compare with.
   * [return] True if this interaction is less than the other, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool operator<(InnerBondInteraction& I) { return (pair_ < I.pair_); }

  /** @brief Checks if this interaction is less than another interaction.
   * @param I The interaction to compare with.
   * [return] True if this interaction is less than the other, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool operator<(const InnerBondInteraction& I) const { return (pair_ < I.pair_); }
};

/**
 * @brief Convert an active inner bond interaction into a vertex-vertex interaction.
 *
 * This function transforms an InnerBondInteraction into a generic Interaction
 * of type Vertex-Vertex. The resulting interaction preserves the pair information
 * and friction, while resetting the moment to zero.
 *
 * @param I The input inner bond interaction to convert.
 * @return Interaction The resulting vertex-vertex interaction.
 *
 * @note Assumes that the interaction is already broken (unbroken == false).
 */
ONIKA_HOST_DEVICE_FUNC inline Interaction broke_interaction(const InnerBondInteraction& I) {
  assert(I[attr::unbroken] == false);
  Interaction res;
  res.pair_ = I.pair_;
  res.pair_.type_ = InteractionTypeId::VertexVertex;
  res[attr::friction] = I[attr::friction];
  res[attr::moment] = exanb::Vec3d{0, 0, 0};
  return res;
}

static_assert(std::is_trivially_copyable_v<InnerBondInteraction>, "Interaction must remain trivially copyable");
}  // namespace exaDEM
