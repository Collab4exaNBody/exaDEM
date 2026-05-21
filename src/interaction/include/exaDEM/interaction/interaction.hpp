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
#include <iostream>

namespace exaDEM {
/**
 * @brief Structure representing an interaction in a Discrete Element Method (DEM) simulation.
 */
struct Interaction {
  InteractionPair pair; /**< The InteractionPair structure containing information about the interacting particles and
                           the type of interaction. */
  // specialized members
  exanb::Vec3d friction = {0, 0, 0}; /**< Friction vector associated with the interaction. */
  exanb::Vec3d moment = {0, 0, 0};   /**< Moment vector associated with the interaction. */

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

  /** @brief Get the owner particle location.
   * [return] Reference to the owner particle location.
   */
  ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& owner() { return pair.owner(); }

  /** @brief Get the partner particle location.
   * [return] Reference to the partner particle location.
   */
  ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& partner() { return pair.partner(); }

  /** @brief Get the driver particle location.
   * [return] Reference to the driver particle location.
   */
  ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& driver() { return j(); }

  /** @brief Get the type of the interaction.
   * [return] The type of the interaction.
   */
  ONIKA_HOST_DEVICE_FUNC inline uint16_t type() { return pair.type; }
  ONIKA_HOST_DEVICE_FUNC inline uint16_t type() const { return pair.type; }

  /** @brief Get the cell index associated with the interaction.
   * [return] The cell index.
   */
  ONIKA_HOST_DEVICE_FUNC inline uint32_t cell() { return pair.owner().cell; }

  /** @brief Get the driver particle ID.
   * [return] The driver particle ID.
   */
  ONIKA_HOST_DEVICE_FUNC inline uint64_t driver_id() { return pair.pj.id; }

  /** @brief Get the pair information associated with the interaction.
   * [return] Reference to the interaction pair.
   */
  ONIKA_HOST_DEVICE_FUNC inline InteractionPair& pair_info() { return pair; }
  ONIKA_HOST_DEVICE_FUNC inline const InteractionPair& pair_info() const { return pair; }

  /**
   * @brief Resets the Interaction structure by setting friction and moment vectors to zero.
   */
  ONIKA_HOST_DEVICE_FUNC void reset() {
    constexpr exanb::Vec3d null = {0, 0, 0};
    friction = null;
    moment = null;
  }

  /** @brief Check if the interaction is persistent.
   * A persistent interaction is an interaction that should be kept in the InteractionManager list even if it is not
   * active anymore. [return] False
   */
  ONIKA_HOST_DEVICE_FUNC bool persistent() { return false; }
  ONIKA_HOST_DEVICE_FUNC bool persistent() const { return false; }

  /** @brief Check if the interaction should ignore other interactions.
   * [return] False
   */
  ONIKA_HOST_DEVICE_FUNC bool ignore_other_interactions() { return false; }

  /**
   * @brief Checks if the interaction is active.
   *
   * This function checks if the interaction is active by examining the moment and friction vectors.
   * An interaction is considered active if either the moment vector or the friction vector is non-zero.
   *
   * @return True if the interaction is active (moment vector or friction vector is non-zero), false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool active() const {
    if (!pair.active()) {
      return false;
    }
    constexpr exanb::Vec3d null = {0, 0, 0};
    bool res = ((moment != null) || (friction != null));
    return res;
  }

  /**
   * @brief Displays the Interaction data.
   */
  void print() {
    pair.print();
    std::cout << "Friction: " << friction << ", Moment: " << moment << ")" << std::endl;
  }

  /**
   * @brief Displays the Interaction data.
   */
  void print() const {
    pair.print();
    std::cout << "Friction: " << friction << ", Moment: " << moment << ")" << std::endl;
  }

  /** @brief Updates the interaction with the values from another interaction.
   * [param] I The interaction to update from.
   */
  ONIKA_HOST_DEVICE_FUNC void update(Interaction& I) {
    this->friction = I.friction;
    this->moment = I.moment;
  }

  /** @brief Checks if two interactions are equal.
   * [param] I The interaction to compare with.
   * [return] True if the interactions are equal, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool operator==(Interaction& I) { return (pair == I.pair); }

  /** @brief Checks if two interactions are equal.
   * [param] I The interaction to compare with.
   * [return] True if the interactions are equal, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool operator==(const Interaction& I) const { return (pair == I.pair); }

  /** @brief Checks if one interaction is less than another.
   * [param] I The interaction to compare with.
   * [return] True if the current interaction is less than the other, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool operator<(Interaction& I) { return (pair < I.pair); }

  /** @brief Checks if one interaction is less than another.
   * [param] I The interaction to compare with.
   * [return] True if the current interaction is less than the other, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool operator<(const Interaction& I) const { return (pair < I.pair); }
};

static_assert(std::is_trivially_copyable_v<Interaction>, "Interaction must remain trivially copyable");
}  // namespace exaDEM
