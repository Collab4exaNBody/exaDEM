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
#include <onika/math/basic_types.h>
#include <onika/math/basic_types_stream.h>
#include <exaDEM/interaction/interaction_pair.hpp>

namespace exaDEM {
/**
 * @brief Structure representing an interaction in a Discrete Element Method (DEM) simulation.
 */
struct InnerBondInteraction {
  InteractionPair pair;
  // specialized members
  exanb::Vec3d friction = {0, 0, 0}; /**< Friction vector associated with the interaction. */
  double en;
  exanb::Vec3d tds;
  double et;
  double dn0;
  double weight = 1.0;
  double criterion;  // interface fracture criterion
  uint8_t unbroken = true;

  ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& i() {
    return pair.pi;
  }

  ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& j() {
    return pair.pj;
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

  ONIKA_HOST_DEVICE_FUNC inline InteractionPair& pair_info() {
    return pair;
  }

  ONIKA_HOST_DEVICE_FUNC inline const InteractionPair& pair_info() const {
    return pair;
  }

  /**
   * @brief Resets the Interaction structure by setting friction and moment vectors to zero.
   */
  ONIKA_HOST_DEVICE_FUNC void reset() {
    constexpr exanb::Vec3d null = {0, 0, 0};
    friction = null;
    dn0 = 0;
    en = 0;
    tds = exanb::Vec3d{0,0,0};
    et = 0;
    unbroken = true;
    // weight and criterion are not reset
  }

  /**
   * @brief Checks if the interaction is active.
   * @return True if the interaction is active (moment vector or friction vector is non-zero), false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC bool active() const {
    if (!pair.active()) {
      return false;
    }
    return true;
  }

  // Defines whether an interaction will be reconstructed or not.
  ONIKA_HOST_DEVICE_FUNC bool persistent() {
    return unbroken;
  }
  // Skip other interactions if this interaction is defined
  ONIKA_HOST_DEVICE_FUNC bool ignore_other_interactions() {
    return true;
  }

  /**
   * @brief Displays the Interaction data.
   */
  void print() {
    pair.print();
    std::cout << "Friction: " << friction << ", en: " << en <<", tds: " << tds << ", et: " << et << ", dn0: " << dn0
              << ", weight: " << weight << ", criterion: " << criterion << ")" << std::endl;
  }

  /**
   * @brief Displays the Interaction data.
   */
  void print() const {
    pair.print();
    std::cout << "Friction: " << friction << ", en: " << en <<", tds: " << tds << ", et: " << et << ", dn0: " << dn0
              << ", weight: " << weight << ", criterion: " << criterion << ")" << std::endl;
  }

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

  ONIKA_HOST_DEVICE_FUNC bool operator==(InnerBondInteraction& I) {
    return (pair == I.pair);
  }

  ONIKA_HOST_DEVICE_FUNC bool operator==(const InnerBondInteraction& I) const {
    return (pair == I.pair);
  }

  ONIKA_HOST_DEVICE_FUNC bool operator<(InnerBondInteraction& I) {
    return (pair < I.pair);
  }

  ONIKA_HOST_DEVICE_FUNC bool operator<(const InnerBondInteraction& I) const {
    return (pair < I.pair);
  }
};

}  // namespace exaDEM
