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

#include <iostream>
// #include <ostream>
#include <onika/math/basic_types.h>
#include <onika/math/basic_types_stream.h>
#include <exaDEM/interaction/interaction_pair.hpp>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/inner_bond_interaction.hpp>
#include <exaDEM/color_log.hpp>

namespace exaDEM {

/* @brief Filter out duplicate interactions based on their ghost status.
 * [param I] The interaction to check.
 * [return] True if the interaction is not a ghost, false otherwise.
 */
template <typename InteractionT>
inline bool filter_duplicates(const InteractionT& I) {
  return I.pair_.ghost_ != InteractionPair::PartnerGhost;
}

/** @brief Compute the maximum of two compile-time constants.
 * [param a] The first constant.
 * [param b] The second constant.
 * [return] The maximum of the two constants.
 */
static constexpr size_t constexpr_max(std::size_t a, std::size_t b) {
  return a > b ? a : b;
}

/**
 * @brief Compute the maximum size of a set of types.
 * [param T] The first type.
 * [param Ts] The remaining types.
 * [return] The maximum size of the types.
 */
template <typename T, typename... Ts>
static constexpr size_t max_sizeof() {
  if constexpr (sizeof...(Ts) == 0) {
    return sizeof(T);
  } else {
    return constexpr_max(sizeof(T), max_sizeof<Ts...>());
  }
}

/**
 * @brief Structure representing an interaction in a Discrete Element Method (DEM) simulation.
 */
struct PlaceholderInteraction {
  // Constants to define the size of the PlaceholderInteraction structure based on the interaction sizes (ex: Interaction, InnerBondInteraction ...).
  static constexpr int PlaceholderInteractionPairSize = sizeof(InteractionPair);
  static constexpr int PlaceholderInteractionSize =
      max_sizeof<exaDEM::Interaction, exaDEM::InnerBondInteraction, exaDEM::InnerBondInteraction>() -
      PlaceholderInteractionPairSize;
  static_assert(PlaceholderInteractionSize > 0);
  static constexpr size_t MaxAlign = std::max({alignof(Interaction), alignof(InnerBondInteraction)});
  static_assert(PlaceholderInteractionSize > 0);

  // members
  InteractionPair pair_;                                             /**< The InteractionPair structure containing information about the interacting particles and the type of interaction. */
  alignas(MaxAlign) uint8_t data_[PlaceholderInteractionSize] = {};  /**< The data buffer for storing interaction-specific information. */

  /** @brief Get the first particle location.
   * [return] A reference to the first particle location.
   */
  ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& i() {
    return pair_.pi_;
  }

  /** @brief Get the second particle location.
   * [return] A reference to the second particle location.
   */
  ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& j() {
    return pair_.pj_;
  }

  /** @brief Get the driver particle location.
   * [return] A reference to the driver particle location.
   */
  ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& driver() {
    return j();
  }

  /** @brief Get the type of the interaction.
  * [return] The type of the interaction.
  */
  ONIKA_HOST_DEVICE_FUNC inline uint16_t type() {
    return pair_.type_;
  }

  /** @brief Get the type of the interaction.
   * [return] The type of the interaction.
   */
  ONIKA_HOST_DEVICE_FUNC inline uint16_t type() const {
    return pair_.type_;
  }

  /** @brief Get the cell index.
   * [return] The cell index.
   */
  ONIKA_HOST_DEVICE_FUNC inline uint32_t cell() {
    return pair_.owner().cell_;
  }

  /** @brief Get the owner particle location.
   * [return] A reference to the owner particle location.
   */
  ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& owner() {
    return pair_.owner();
  }

  /** @brief Get the owner particle location.
   * [return] A reference to the owner particle location.
   */
  ONIKA_HOST_DEVICE_FUNC inline const ParticleSubLocation& owner() const {
    return pair_.owner();
  }

  /** @brief Get the partner particle location.
   * [return] A reference to the partner particle location.
   */
  ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& partner() {
    return pair_.partner();
  }

  /** @brief Get the interaction pair information.
   * [return] A reference to the interaction pair information.
   */
  ONIKA_HOST_DEVICE_FUNC inline InteractionPair& pair_info() {
    return pair_;
  }

  /** @brief Get the interaction pair information.
   * [return] A reference to the interaction pair information.
   */
  ONIKA_HOST_DEVICE_FUNC inline const InteractionPair& pair_info() const {
    return pair_;
  }

  /** @brief Check if the interaction is consistent by verifying the consistency of its InteractionPair.
   * Used for debugging purposes to ensure that the interaction data is well-formed and does not contain invalid values.
   * CPU only function, not intended to be called from device code.
   * [return] True if the interaction is consistent, false otherwise.
   */
  template <bool DisplayWarnings = true>
  inline bool consistent() const {
    return pair_.consistent<DisplayWarnings>();
  }

  /** @brief Displays the Interaction data.
   */
  void print() {
    if (type() < InteractionTypeId::NTypesParticleParticle) {
      return this->as<Interaction>().print();
    } else if (type() == InteractionTypeId::InnerBond) {
      return this->as<InnerBondInteraction>().print();
    }
  }

  /** @brief Displays the Interaction data.
   */
  void print() const {
    if (type() < InteractionTypeId::NTypesParticleParticle) {
      this->as<Interaction>().print();
    } else if (type() == InteractionTypeId::InnerBond) {
      this->as<InnerBondInteraction>().print();
    }
  }

  /** @brief Updates the placeholder interaction data by copying from another interaction.
   * [param in] The PlaceholderInteraction to copy data from.
   */
  void update(PlaceholderInteraction& in) {
    std::memcpy(data_, in.data_, PlaceholderInteractionSize * sizeof(uint8_t));
  }

  /** @brief Clears the placeholder interaction data.
   * [return] A reference to the cleared placeholder interaction data.
   */
  void clear_placeholder() {
    memset(data_, 0, PlaceholderInteractionSize);
  }


  ONIKA_HOST_DEVICE_FUNC bool active() const {
    if (type() < InteractionTypeId::NTypesParticleParticle) {
      return this->as<Interaction>().active();
    } else if (type() == InteractionTypeId::InnerBond) {
      return this->as<InnerBondInteraction>().active();
    }
#ifndef ONIKA_CUDA_VERSION
    pair_.print();
    color_log::mpi_error("PlaceholderInteraction::active",
                         "The type value of this interaction is invalid: " + std::to_string(type()));
    std::exit(EXIT_FAILURE);
#endif
    return false;  // default
  }

  /** @brief Check if the interaction is persistent.
   * A persistent interaction is an interaction that should be kept in the InteractionManager list even if it is not active anymore.
   * [return] True if the interaction is persistent, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool persistent() {
    if (type() < InteractionTypeId::NTypesParticleParticle) {
      return this->as<Interaction>().persistent();
    } else if (type() == InteractionTypeId::InnerBond) {
      return this->as<InnerBondInteraction>().persistent();
    }
#ifndef ONIKA_CUDA_VERSION
    if(type() == InteractionTypeId::Undefined) {
      color_log::mpi_error("PlaceholderInteraction::persistent",
                           "The interaction is undefined, please define it.");
    }
    pair_.print();
    color_log::mpi_error("PlaceholderInteraction::persistent",
                         "The type value of this interaction is invalid: "
                         + std::to_string(static_cast<int>(type())));
    std::exit(EXIT_FAILURE);
#endif
    return false;  // default
  }

  /** @brief Check if the interaction is persistent.
   * A persistent interaction is an interaction that should be kept in the InteractionManager list even if it is not active anymore.
   * [return] True if the interaction is persistent, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool persistent() const {
    if (type() < InteractionTypeId::NTypesParticleParticle) {
      return this->as<Interaction>().persistent();
    } else if (type() == InteractionTypeId::InnerBond) {
      return this->as<InnerBondInteraction>().persistent();
    }
#ifndef ONIKA_CUDA_VERSION
    if(type() == InteractionTypeId::Undefined) {
      color_log::mpi_error("PlaceholderInteraction::persistent",
                           "The interaction is undefined, please define it.");
    }
    pair_.print();
    color_log::mpi_error("PlaceholderInteraction::persistent",
                         "The type value of this interaction is invalid: "
                         + std::to_string(static_cast<int>(type())));
    std::exit(EXIT_FAILURE);
#endif
    return false;  // default
    }

  /** @brief Check if the interaction should ignore other interactions.
   * [return] True if the interaction should ignore other interactions, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool ignore_other_interactions() {
    if (type() < InteractionTypeId::NTypesParticleParticle) {
      return this->as<Interaction>().ignore_other_interactions();
    } else if (type() == InteractionTypeId::InnerBond) {
      return this->as<InnerBondInteraction>().ignore_other_interactions();
    }
#ifndef ONIKA_CUDA_VERSION
    color_log::mpi_error("PlaceholderInteraction::ignore_other_interactions",
                         "The type value of this interaction is invalid");
#endif
    return false;
  }

  /** @brief Reset the interaction.
   */
  ONIKA_HOST_DEVICE_FUNC void reset() {
    // Do not use it via a placeholder_interaction
    if (type() < InteractionTypeId::NTypesParticleParticle) {
      this->as<Interaction>().reset();
    } else if (type() == InteractionTypeId::InnerBond) {
      this->as<InnerBondInteraction>().reset();
    }
#ifndef ONIKA_CUDA_VERSION
    else {
      color_log::mpi_error("PlaceholderInteraction::reset",
                           "The type value of this interaction is invalid: " + std::to_string(type()));
    }
#endif
  }

  /** @brief Convert the placeholder interaction to a specific interaction type.
   * @tparam T The type to convert to.
   * @return A reference to the converted interaction.
   */
  template <typename T>
  ONIKA_HOST_DEVICE_FUNC T& as() {
    // Ensure that the type T is compatible with the PlaceholderInteraction structure.
    static_assert(std::is_standard_layout_v<T>, "as<T>() requires a standard-layout type");
    // Ensure that the size of T is at least as large as the InteractionPair, since PlaceholderInteraction contains an InteractionPair as its first member.
    static_assert(sizeof(T) >= sizeof(InteractionPair), "Type T must contain at least InteractionPair");
    // Ensure that the alignment of T does not exceed the alignment of PlaceholderInteraction, since we will be reinterpreting the memory of PlaceholderInteraction as type T.
    static_assert(alignof(T) <= alignof(PlaceholderInteraction),
                  "Type T alignment exceeds PlaceholderInteraction alignment");
    return *reinterpret_cast<T*>(this);
  }

  /** @brief Convert the placeholder interaction to a specific interaction type.
   * @tparam T The type to convert to.
   * @return A reference to the converted interaction.
   */
  template <typename T>
  ONIKA_HOST_DEVICE_FUNC const T& as() const {
    static_assert(std::is_standard_layout_v<T>, "as<T>() requires a standard-layout type");
    static_assert(sizeof(T) >= sizeof(InteractionPair), "Type T must contain at least InteractionPair");
    static_assert(alignof(T) <= alignof(PlaceholderInteraction),
                  "Type T alignment exceeds PlaceholderInteraction alignment");

    return *reinterpret_cast<const T*>(this);
  }

  /** @brief Convert the placeholder interaction to a specific interaction type.
   * @tparam T The type to convert to.
   * @return A reference to the converted interaction.
   */
  template <InteractionType IT>
  auto& convert() {
    if constexpr (IT == ParticleParticle) {
      return as<Interaction>();
    } else if constexpr (IT == ParticleDriver) {
      return as<Interaction>();
    } if constexpr (IT == InnerBond) {
      return as<InnerBondInteraction>();
    }
    color_log::mpi_error("PlaceholderInteraction::as<InteractionType>",
                         "Error, no Interaction type is defined for this value of InteractionType");
  }

  /** @brief Check if two placeholder interactions are equal.
   * @param I The other placeholder interaction to compare with.
   * @return True if the interactions are equal, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool operator==(PlaceholderInteraction& I) {
    return (pair_ == I.pair_);
  }

  /** @brief Check if two placeholder interactions are equal.
   * @param I The other placeholder interaction to compare with.
   * @return True if the interactions are equal, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool operator==(const PlaceholderInteraction& I) const {
    return (pair_ == I.pair_);
  }

  /** @brief Check if a placeholder interaction is less than another.
   * @param I The other placeholder interaction to compare with.
   * @return True if this interaction is less than the other, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool operator<(PlaceholderInteraction& I) {
    return (pair_ < I.pair_);
  }

  /** @brief Check if a placeholder interaction is less than another.
   * @param I The other placeholder interaction to compare with.
   * @return True if this interaction is less than the other, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool operator<(const PlaceholderInteraction& I) const {
    return (pair_ < I.pair_);
  }

  /**
   * @brief Comparison operator used to sort interactions by owner particle index.
   * Ordering rule:
   * 1. Compare owner particle local index (p)
   * 2. If equal, fallback to pair comparison
   * @param I Interaction to compare with
   */
  ONIKA_HOST_DEVICE_FUNC bool sort_by_owner_p(const PlaceholderInteraction& I) const {
    if (this->owner().p_ != I.owner().p_) {
      return (this->owner().p_ < I.owner().p_);
    }
    return (this->pair_ < I.pair_);
  }

  /**
   * @brief Assign from an  Interaction.
   *
   * Reinterprets the internal storage as Interaction and performs
   * a copy assignment. Allows implicit conversion from Interaction
   * to PlaceholderInteraction.
   *
   * @param I Source interaction
   * @return Reference to this PlaceholderInteraction
   */
  ONIKA_HOST_DEVICE_FUNC inline
      PlaceholderInteraction& operator=(const Interaction& I) {
        Interaction& AsI = as<Interaction>();
        AsI = I;
        return *this;
      }

  /**
   * @brief Assign from an InnerBondInteraction.
   *
   * Reinterprets the internal storage as InnerBondInteraction and
   * performs a copy assignment.
   *
   * @param I Source inner bond interaction
   * @return Reference to this PlaceholderInteraction
   */
  ONIKA_HOST_DEVICE_FUNC inline
      PlaceholderInteraction& operator=(const InnerBondInteraction& I) {
        InnerBondInteraction& AsI = as<InnerBondInteraction>();
        AsI = I;
        return *this;
      }
};

/** @brief Update the interactions based on their history.
 * @param interactions The vector of interactions to update.
 * @param history The vector of historical interactions.
 */
inline void update(std::vector<PlaceholderInteraction>& interactions,
  std::vector<PlaceholderInteraction>& history) {
  for (size_t it = 0; it < interactions.size(); it++) {
    auto& item = interactions[it];
    auto lower = std::lower_bound(history.begin(), history.end(), item);
    if (lower != history.end()) {
      if (item == *lower) {
        item.update(*lower);  // update interaction with specific interaction data (history).
      }
    }
  }
}
}  // namespace exaDEM
