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

template <typename InteractionT>
inline bool filter_duplicates(const InteractionT& I) {
  return I.pair.ghost != InteractionPair::PartnerGhost;
}

static constexpr size_t constexpr_max(std::size_t a, std::size_t b) {
  return a > b ? a : b;
}

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
  static constexpr int PlaceholderInteractionPairSize = sizeof(InteractionPair);
  static constexpr int PlaceholderInteractionSize =
      max_sizeof<exaDEM::Interaction, exaDEM::InnerBondInteraction, exaDEM::InnerBondInteraction>() -
      PlaceholderInteractionPairSize;

  static_assert(PlaceholderInteractionSize > 0);

  static constexpr size_t MaxAlign = std::max({alignof(Interaction), alignof(InnerBondInteraction)});

  static_assert(PlaceholderInteractionSize > 0);

  // members
  InteractionPair pair;
  alignas(MaxAlign) uint8_t data[PlaceholderInteractionSize];

  ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& i() {
    return pair.pi;
  }

  ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& j() {
    return pair.pj;
  }

  ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& driver() {
    return j();
  }

  ONIKA_HOST_DEVICE_FUNC inline uint16_t type() {
    return pair.type;
  }

  ONIKA_HOST_DEVICE_FUNC inline uint16_t type() const {
    return pair.type;
  }

  ONIKA_HOST_DEVICE_FUNC inline uint32_t cell() {
    return pair.owner().cell;
  }  // associate cell -> cell_i

  ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& owner() {
    return pair.owner();
  }

  ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& partner() {
    return pair.partner();
  }

  ONIKA_HOST_DEVICE_FUNC inline InteractionPair& pair_info() {
    return pair;
  }

  ONIKA_HOST_DEVICE_FUNC inline const InteractionPair& pair_info() const {
    return pair;
  }

  /**
   * @brief Displays the Interaction data.
   */
  void print() {
    if (type() < InteractionTypeId::NTypesParticleParticle) {
      return this->as<Interaction>().print();
    } else if (type() == InteractionTypeId::InnerBond) {
      return this->as<InnerBondInteraction>().print();
    }
  }

  /**
   * @brief Displays the Interaction data.
   */
  void print() const {
    if (type() < InteractionTypeId::NTypesParticleParticle) {
      return this->as<Interaction>().print();
    } else if (type() == InteractionTypeId::InnerBond) {
      return this->as<InnerBondInteraction>().print();
    }
  }

  void update(PlaceholderInteraction& in) {
    std::memcpy(data, in.data, PlaceholderInteractionSize * sizeof(uint8_t));
  }

  void clear_placeholder() {
    memset(data, 0, PlaceholderInteractionSize);
  }

  ONIKA_HOST_DEVICE_FUNC bool active() const {
    if (type() < InteractionTypeId::NTypesParticleParticle) {
      return this->as<Interaction>().active();
    } else if (type() == InteractionTypeId::InnerBond) {
      return this->as<InnerBondInteraction>().active();
    }
#ifndef ONIKA_CUDA_VERSION
    pair.print();
    color_log::mpi_error("PlaceholderInteraction::active",
                         "The type value of this interaction is invalid: " + std::to_string(type()));
    std::exit(EXIT_FAILURE);
#endif
    return false;  // default
  }

  // Defines whether an interaction will be reconstructed or not.
  ONIKA_HOST_DEVICE_FUNC bool persistent() {
    if (type() < InteractionTypeId::NTypesParticleParticle) {
      return this->as<Interaction>().persistent();
    } else if (type() == InteractionTypeId::InnerBond) {
      return this->as<InnerBondInteraction>().persistent();
    }
#ifndef ONIKA_CUDA_VERSION
    color_log::mpi_error("PlaceholderInteraction::persistent",
                         "The type value of this interaction is invalid: " + std::to_string(type()));
    std::exit(EXIT_FAILURE);
#endif
    return false;  // default
  }

  ONIKA_HOST_DEVICE_FUNC bool ignore_other_interactions() {
    if (type() < InteractionTypeId::NTypesParticleParticle) {
      return this->as<Interaction>().ignore_other_interactions();
    } else if (type() == InteractionTypeId::InnerBond) {
      return this->as<InnerBondInteraction>().ignore_other_interactions();
    }
#ifndef ONIKA_CUDA_VERSION
    color_log::mpi_error("PlaceholderInteraction::ignore_other_interactions",
                         "The type value of this interaction is invalid");
    std::exit(EXIT_FAILURE);
#endif
    return false;
  }

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
      std::exit(EXIT_FAILURE);
    }
#endif
  }

  template <typename T>
  ONIKA_HOST_DEVICE_FUNC T& as() {
    static_assert(std::is_standard_layout_v<T>, "as<T>() requires a standard-layout type");
    static_assert(sizeof(T) >= sizeof(InteractionPair), "Type T must contain at least InteractionPair");
    static_assert(alignof(T) <= alignof(PlaceholderInteraction),
                  "Type T alignment exceeds PlaceholderInteraction alignment");
    return *reinterpret_cast<T*>(this);
  }

  template <typename T>
  ONIKA_HOST_DEVICE_FUNC const T& as() const {
    static_assert(std::is_standard_layout_v<T>, "as<T>() requires a standard-layout type");
    static_assert(sizeof(T) >= sizeof(InteractionPair), "Type T must contain at least InteractionPair");
    static_assert(alignof(T) <= alignof(PlaceholderInteraction),
                  "Type T alignment exceeds PlaceholderInteraction alignment");

    return *reinterpret_cast<const T*>(this);
  }

  template <InteractionType IT>
  auto& convert() {
    if constexpr (IT == ParticleParticle) {
      return as<Interaction>();
    }
    if constexpr (IT == InnerBond) {
      return as<InnerBondInteraction>();
    }
    color_log::mpi_error("PlaceholderInteraction::as<InteractionType>",
                         "Error, no Interaction type is defined for this value of InteractionType");
  }

  ONIKA_HOST_DEVICE_FUNC bool operator==(PlaceholderInteraction& I) {
    return (pair == I.pair);
  }

  ONIKA_HOST_DEVICE_FUNC bool operator==(const PlaceholderInteraction& I) const {
    return (pair == I.pair);
  }

  ONIKA_HOST_DEVICE_FUNC bool operator<(PlaceholderInteraction& I) {
    return (pair < I.pair);
  }

  ONIKA_HOST_DEVICE_FUNC bool operator<(const PlaceholderInteraction& I) const {
    return (pair < I.pair);
  }
};

inline void update(std::vector<PlaceholderInteraction>& interactions, std::vector<PlaceholderInteraction>& history) {
  for (size_t it = 0; it < interactions.size(); it++) {
    auto& item = interactions[it];
    auto lower = std::lower_bound(history.begin(), history.end(), item);
    if (lower != history.end()) {
      if (item == *lower) {
        item.update(*lower);
      }
    }
  }
}
}  // namespace exaDEM
