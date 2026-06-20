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

#include <exaDEM/interaction/interaction_pair.hpp>
#include <exaDEM/interface/rupture_criterion.hpp>
#include <iostream>

namespace exaDEM {
/**
 * @brief Structure representing an interaction in a Discrete Element Method (DEM) simulation.
 */
struct InnerBondInteraction {
  InteractionPair pair; /**< The InteractionPair structure containing information about the interacting particles and
                           the type of interaction. */

  // specialized members
  exanb::Vec3d friction = {0, 0, 0}; /**< Friction vector associated with the interaction. */
  double en;                         /**< Normal energy associated with the interaction. */
  exanb::Vec3d tds;                  /**< Tangential displacement vector associated with the interaction. */
  double et;                         /**< Tangential energy associated with the interaction. */
  double dn0;                        /**< Initial normal overlap associated with the interaction. */
  double weight = 1.0; /**< Weight associated with the interaction, used for scaling forces and moments. */
  RuptureCriteria
      criterion; /**< Criterion value associated with the interaction, used for determining bond breakage. */

  uint8_t unbroken = true; /**< Flag indicating whether the bond is unbroken (true) or broken (false). */

  /** @brief Get the first particle location.
   * [return] Reference to the first particle location.
   */
  ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& i() { return pair.pi; }
  ONIKA_HOST_DEVICE_FUNC inline const ParticleSubLocation& i() const { return pair.pi; }

  /** @brief Get the second particle location.
   * [return] Reference to the second particle location.
   */
  ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& j() { return pair.pj; }
  ONIKA_HOST_DEVICE_FUNC inline const ParticleSubLocation& j() const { return pair.pj; }

  /** @brief Get the type of the interaction.
   * [return] The type of the interaction.
   */
  ONIKA_HOST_DEVICE_FUNC inline uint16_t type() { return pair.type; }
  ONIKA_HOST_DEVICE_FUNC inline uint16_t type() const { return pair.type; }

  /** @brief Get the cell index associated with the interaction.
   * [return] The cell index.
   */
  ONIKA_HOST_DEVICE_FUNC inline uint32_t cell() { return pair.owner().cell; }

  /** @brief Get the pair information associated with the interaction.
   * [return] Reference to the interaction pair.
   */
  ONIKA_HOST_DEVICE_FUNC inline InteractionPair& pair_info() { return pair; }
  ONIKA_HOST_DEVICE_FUNC inline const InteractionPair& pair_info() const { return pair; }

  /** @brief Resets the Interaction structure by setting friction and moment vectors to zero.
   */
  ONIKA_HOST_DEVICE_FUNC void reset() {
    constexpr exanb::Vec3d null = {0, 0, 0};
    friction = null;
    dn0 = 0;
    en = 0;
    tds = exanb::Vec3d{0, 0, 0};
    et = 0;
    unbroken = true;
    // weight and criteria are not reset
  }

  /** @brief Checks if the interaction is active.
   * [return] True if the interaction is active (moment vector or friction vector is non-zero), false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool active() const {
    if (!pair.active()) {
      return false;
    }
    return true;
  }

  /** @brief Defines whether an interaction will be reconstructed or not.
   * [return] True if the interaction is persistent, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool persistent() { return unbroken; }

  /** @brief Defines whether an interaction will be reconstructed or not.
   * [return] True if the interaction is persistent, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool persistent() const { return unbroken; }

  /** @brief Defines whether to skip other interactions.
   * [return] True.
   */
  ONIKA_HOST_DEVICE_FUNC bool ignore_other_interactions() { return true; }

  /**
   * @brief Displays the Interaction data.
   */
  void print() {
    pair.print();
    std::cout << "Friction: " << friction << ", en: " << en << ", tds: " << tds << ", et: " << et << ", dn0: " << dn0
              << ", weight: " << weight << ", criterion: " << criterion << ")" << std::endl;
  }

  /**
   * @brief Displays the Interaction data.
   */
  void print() const {
    pair.print();
    std::cout << "Friction: " << friction << ", en: " << en << ", tds: " << tds << ", et: " << et << ", dn0: " << dn0
              << ", weight: " << weight << ", criterion: " << criterion << ")" << std::endl;
  }

  /** @brief Updates the interaction with values from another interaction.
   * @param I The interaction to update from.
   * [return] None.
   */
  ONIKA_HOST_DEVICE_FUNC void update(InnerBondInteraction& I) {
    this->friction = I.friction;
    this->en = I.en;
    this->tds = I.tds;
    this->et = I.et;
    this->dn0 = I.dn0;
    this->weight = I.weight;
    this->criterion = I.criterion;
    this->unbroken = I.unbroken;
  }

  /** @brief Checks if two interactions are equal.
   * @param I The interaction to compare with.
   * [return] True if the interactions are equal, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool operator==(InnerBondInteraction& I) { return (pair == I.pair); }

  /** @brief Checks if two interactions are equal.
   * @param I The interaction to compare with.
   * [return] True if the interactions are equal, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool operator==(const InnerBondInteraction& I) const { return (pair == I.pair); }

  /** @brief Checks if this interaction is less than another interaction.
   * @param I The interaction to compare with.
   * [return] True if this interaction is less than the other, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool operator<(InnerBondInteraction& I) { return (pair < I.pair); }

  /** @brief Checks if this interaction is less than another interaction.
   * @param I The interaction to compare with.
   * [return] True if this interaction is less than the other, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool operator<(const InnerBondInteraction& I) const { return (pair < I.pair); }
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
  assert(I.unbroken == false);
  Interaction res;
  res.pair = I.pair;
  res.pair.type = InteractionTypeId::VertexVertex;
  res.friction = I.friction;
  res.moment = exanb::Vec3d{0, 0, 0};
  return res;
}

static_assert(std::is_trivially_copyable_v<InnerBondInteraction>, "Interaction must remain trivially copyable");
}  // namespace exaDEM
