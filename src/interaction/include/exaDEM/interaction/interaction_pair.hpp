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

#include <exaDEM/color_log.hpp>
#include <exaDEM/interaction/interaction_enum.hpp>
#include <iostream>

namespace exaDEM {
/** @brief Structure representing the location of a particle in the simulation domain. */
struct ParticleSubLocation {
  uint64_t id_;   /**< Id of the first particle */
  uint32_t cell_; /**< Index of the cell of the first particle involved in the interaction. */
  uint16_t p_;    /**< Index of the particle within its cell for the particle involved in the interaction. */
  uint32_t sub_;  /**< Sub-particle index for the particle involved in the interaction. */

  /** @brief Check if two particle sub-locations are equal.
   * @param P The other particle sub-location to compare with.
   * @return True if the sub-locations are equal, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool operator==(ParticleSubLocation& P) {
    return (id_ == P.id_ && cell_ == P.cell_ && p_ == P.p_ && sub_ == P.sub_);
  }

  /** @brief Check if two particle sub-locations are equal.
   * @param P The other particle sub-location to compare with.
   * @return True if the sub-locations are equal, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool operator==(const ParticleSubLocation& P) const {
    return (id_ == P.id_ && cell_ == P.cell_ && p_ == P.p_ && sub_ == P.sub_);
  }
};

/**
 * @brief Structure representing an interaction in a Discrete Element Method (DEM) simulation.
 */
struct InteractionPair {
  // means that the interaction is linked to two particles that are in the current subdomain
  static constexpr uint8_t NotGhost = 0;
  // means that the interaction is linked to a particle that is in the current subdomain
  static constexpr uint8_t OwnerGhost = 1;
  // means that the interaction is linked to a particle that is in the current subdomain
  static constexpr uint8_t PartnerGhost = 2;

  ParticleSubLocation pi_;                       /**< Sub-location of the first particle in the interaction. */
  ParticleSubLocation pj_;                       /**< Sub-location of the second particle in the interaction. */
  uint16_t type_ = InteractionTypeId::Undefined; /**< Type of the interaction (e.g., contact type). */
  uint8_t swap_ = false; /**< Flag indicating whether the order of the particles is swapped (i.e., if true, pi and pj are
                           swapped). */
  uint8_t ghost_ =
      NotGhost; /**< Flag indicating the ghost status of the interaction (e.g., whether it involves ghost particles). */

  /** @brief Get a reference to the owner particle sub-location.
   * @return A reference to the owner particle sub-location.
   */
  ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& owner() {
    if (!swap_) {
      return pi_;
    }
    return pj_;
  }

  /** @brief Get a reference to the partner particle sub-location.
   * @return A reference to the partner particle sub-location.
   */
  ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& partner() {
    if (!swap_) {
      return pj_;
    }
    return pi_;
  }

  /** @brief Get a reference to the owner particle sub-location.
   * @return A reference to the owner particle sub-location.
   */
  ONIKA_HOST_DEVICE_FUNC inline const ParticleSubLocation& owner() const {
    if (!swap_) {
      return pi_;
    }
    return pj_;
  }

  /** @brief Get a reference to the partner particle sub-location.
   * @return A reference to the partner particle sub-location.
   */
  ONIKA_HOST_DEVICE_FUNC inline const ParticleSubLocation& partner() const {
    if (!swap_) {
      return pj_;
    }
    return pi_;
  }

  /** @brief Check if the interaction is active.
   * @return True if the interaction is active, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC inline bool active() { return ghost_ != PartnerGhost; }
  ONIKA_HOST_DEVICE_FUNC inline bool active() const { return ghost_ != PartnerGhost; }

  /** @brief Check if the interaction is consistent by verifying the consistency of its InteractionPair.
   * Used for debugging purposes to ensure that the interaction data is well-formed and does not contain invalid values.
   * CPU only function, not intended to be called from device code.
   * [return] True if the interaction is consistent, false otherwise.
   */
  template <bool DisplayWarnings = true>
  inline bool consistent() const {
    std::string function_name = "InteractionPair::consistent()";
    bool res = true;
    if (type_ >= InteractionTypeId::NTypes) {
      if constexpr (DisplayWarnings) {
        color_log::warning(function_name, "type is undefined: " + std::to_string(type_));
      }
      res = false;
    }
    if (owner() == partner()) {
      if constexpr (DisplayWarnings) {
        color_log::warning(function_name, "owner ParticleSubLocation is equal to the partner()");
      }
      res = false;
    }
    if (ghost_ > InteractionPair::PartnerGhost) {
      if constexpr (DisplayWarnings) {
        color_log::warning(function_name, "ghost is undefined: " + std::to_string(ghost_));
      }
      res = false;
    }
    if (swap_ >= 2 /* not a boolean */) {
      if constexpr (DisplayWarnings) {
        color_log::warning(function_name, "swap is undefined: " + std::to_string(swap_));
      }
      res = false;
    }
    return res;
  }

  /**
   * @brief Displays the Interaction data.
   */
  void print() {
    auto [id_i, cell_i, p_i, sub_i] = owner();
    auto [id_j, cell_j, p_j, sub_j] = partner();
    std::cout << "Interaction(type = " << int(type_) << " [cell: " << cell_i << ", idx " << p_i
              << ", particle id: " << id_i << ", sub: " << sub_i << "] and"
              << " [cell: " << cell_j << ", idx " << p_j << ", particle id: " << id_j << ", sub: " << sub_j << "]"
              << " swap: " << int(swap_) << " ghost: " << int(ghost_) << ")" << std::endl;
  }

  /**
   * @brief Displays the Interaction data.
   */
  void print() const {
    auto [id_i, cell_i, p_i, sub_i] = owner();
    auto [id_j, cell_j, p_j, sub_j] = partner();
    std::cout << "Interaction(type = " << int(type_) << " [cell: " << cell_i << ", idx " << p_i
              << ", particle id: " << id_i << ", sub: " << sub_i << "] and"
              << " [cell: " << cell_j << ", idx " << p_j << ", particle id: " << id_j << ", sub: " << sub_j << "]"
              << " swap: " << int(swap_) << " ghost: " << int(ghost_) << ")" << std::endl;
  }

  ///////////////
  // Operators //
  ///////////////

  /** @brief Check if two interaction pairs are equal.
   * @param I The other interaction pair to compare with.
   * @return True if the interaction pairs are equal, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool operator==(InteractionPair& I) {
    auto& me_pi = this->pi_;
    auto& you_pi = I.pi_;
    auto& me_pj = this->pj_;
    auto& you_pj = I.pj_;

    if (me_pi.id_ == you_pi.id_ && me_pj.id_ == you_pj.id_ && me_pi.sub_ == you_pi.sub_ && me_pj.sub_ == you_pj.sub_ &&
        this->swap_ == I.swap_ && this->type_ == I.type_) {
      return true;
    } else {
      return false;
    }
  }

  /** @brief Check if two interaction pairs are equal.
   * @param I The other interaction pair to compare with.
   * @return True if the interaction pairs are equal, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool operator==(const InteractionPair& I) const {
    auto& me_pi = this->pi_;
    auto& you_pi = I.pi_;
    auto& me_pj = this->pj_;
    auto& you_pj = I.pj_;

    if (me_pi.id_ == you_pi.id_ && me_pj.id_ == you_pj.id_ && me_pi.sub_ == you_pi.sub_ && me_pj.sub_ == you_pj.sub_ &&
        this->swap_ == I.swap_ && this->type_ == I.type_) {
      return true;
    } else {
      return false;
    }
  }

  /** @brief Check if an interaction pair is less than another.
   * @param I The other interaction pair to compare with.
   * @return True if this interaction pair is less than the other, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool operator<(const InteractionPair& I) const {
    const auto& a = owner();
    const auto& b = partner();
    const auto& c = I.owner();
    const auto& d = I.partner();

    if (a.id_ != c.id_) {
      return a.id_ < c.id_;
    }
    if (b.id_ != d.id_) {
      return b.id_ < d.id_;
    }

    auto t1 = type_;
    auto t2 = I.type_;
    if (t1 != t2) {
      return t1 < t2;
    }

    if (swap_ != I.swap_) {
      return swap_ < I.swap_;
    }

    if (a.sub_ != c.sub_) {
      return a.sub_ < c.sub_;
    }
    return b.sub_ < d.sub_;
  }
};
}  // namespace exaDEM
