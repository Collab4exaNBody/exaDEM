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
  InteractionPair pair_; /**< The InteractionPair structure containing information about the interacting particles and
                           the type of interaction. */

  // specialized members
  exanb::Vec3d friction_ = {0, 0, 0}; /**< Friction vector associated with the interaction. */
  double en_;                         /**< Normal energy associated with the interaction. */
  exanb::Vec3d tds_;                  /**< Tangential displacement vector associated with the interaction. */
  double et_;                         /**< Tangential energy associated with the interaction. */
  double dn0_;                        /**< Initial normal overlap associated with the interaction. */
  double weight_ = 1.0; /**< Weight associated with the interaction, used for scaling forces and moments. */
  RuptureCriteria
      criterion_; /**< Criterion value associated with the interaction, used for determining bond breakage. */

  uint8_t unbroken_ = true; /**< Flag indicating whether the bond is unbroken (true) or broken (false). */

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
    friction_ = null;
    dn0_ = 0;
    en_ = 0;
    tds_ = exanb::Vec3d{0, 0, 0};
    et_ = 0;
    unbroken_ = true;
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
  ONIKA_HOST_DEVICE_FUNC bool persistent() { return unbroken_; }

  /** @brief Defines whether an interaction will be reconstructed or not.
   * [return] True if the interaction is persistent, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool persistent() const { return unbroken_; }

  /** @brief Defines whether to skip other interactions.
   * [return] True.
   */
  ONIKA_HOST_DEVICE_FUNC bool ignore_other_interactions() { return true; }

  /**
   * @brief Displays the Interaction data.
   */
  void print() {
    pair_.print();
    std::cout << "Friction: " << friction_ << ", en: " << en_ << ", tds: " << tds_ << ", et: " << et_ << ", dn0: " << dn0_
              << ", weight: " << weight_ << ", criterion: " << criterion_ << ")" << std::endl;
  }

  /**
   * @brief Displays the Interaction data.
   */
  void print() const {
    pair_.print();
    std::cout << "Friction: " << friction_ << ", en: " << en_ << ", tds: " << tds_ << ", et: " << et_ << ", dn0: " << dn0_
              << ", weight: " << weight_ << ", criterion: " << criterion_ << ")" << std::endl;
  }

  /** @brief Updates the interaction with values from another interaction.
   * @param I The interaction to update from.
   * [return] None.
   */
  ONIKA_HOST_DEVICE_FUNC void update(InnerBondInteraction& I) {
    this->friction_ = I.friction_;
    this->en_ = I.en_;
    this->tds_ = I.tds_;
    this->et_ = I.et_;
    this->dn0_ = I.dn0_;
    this->weight_ = I.weight_;
    this->criterion_ = I.criterion_;
    this->unbroken_ = I.unbroken_;
  }

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
  assert(I.unbroken_ == false);
  Interaction res;
  res.pair_ = I.pair_;
  res.pair_.type_ = InteractionTypeId::VertexVertex;
  res.friction_ = I.friction_;
  res.moment_ = exanb::Vec3d{0, 0, 0};
  return res;
}

static_assert(std::is_trivially_copyable_v<InnerBondInteraction>, "Interaction must remain trivially copyable");
}  // namespace exaDEM
