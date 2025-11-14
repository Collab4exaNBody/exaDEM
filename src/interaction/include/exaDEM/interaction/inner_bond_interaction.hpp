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
//#include <ostream>
#include <onika/math/basic_types.h>
#include <onika/math/basic_types_stream.h>
#include <exaDEM/interaction/interaction_pair.hpp>

namespace exaDEM
{
  /**
   * @brief Structure representing an interaction in a Discrete Element Method (DEM) simulation.
   */
  struct InnerBondInteraction
  {
    InteractionPair pair;
    exanb::Vec3d friction = {0, 0, 0}; /**< Friction vector associated with the interaction. */
    double en;
    double et;
    double dn0;
    double S; // surface of the interface

    ParticleSubLocation& i() { return pair.pi; }
    ParticleSubLocation& j() { return pair.pj; }

    uint16_t type() { return pair.type; } 
    uint16_t type() const { return pair.type; } 
    uint32_t cell() { return pair.pi.cell; } // associate cell -> cell_i
    uint32_t partner_cell() { return pair.pj.cell; } // associate cell -> cell_j
    InteractionPair& pair_info() { return pair; }
    const InteractionPair& pair_info() const { return pair; }
    double surface() { return S; } // associate cell -> cell_j

    /**
     * @brief Resets the Interaction structure by setting friction and moment vectors to zero.
     */
    ONIKA_HOST_DEVICE_FUNC void reset()
    {
      constexpr exanb::Vec3d null = {0, 0, 0};
      friction = null;
      dn0 = 0;
      en = 0;
      et = 0;
    }

    /**
     * @brief Checks if the interaction is active.
     *
     * This function checks if the interaction is active by examining the moment and friction vectors.
     * An interaction is considered active if either the moment vector or the friction vector is non-zero.
     *
     * @return True if the interaction is active (moment vector or friction vector is non-zero), false otherwise.
     */
    ONIKA_HOST_DEVICE_FUNC bool is_active() const
    {
      return true;
    }

    // Defines whether an interaction will be reconstructed or not.
    ONIKA_HOST_DEVICE_FUNC bool persistent() { return true; }
    // Skip other interactions if this interaction is defined
    ONIKA_HOST_DEVICE_FUNC bool ignore_other_interactions() { return true; } 

    /**
     * @brief Displays the Interaction data.
     */
    void print()
    {
      pair.print();
      std::cout << "Friction: " << friction 
        << ", en: " << en 
        << ", et: " << et 
        << ", dn0: " << dn0 
        << ")" << std::endl;
    }

    /**
     * @brief Displays the Interaction data.
     */
    void print() const
    {
      pair.print();
      std::cout << "Friction: " << friction 
        << ", en: " << en 
        << ", et: " << et 
        << ", dn0: " << dn0 
        << ")" << std::endl;
    }


    ONIKA_HOST_DEVICE_FUNC void update(InnerBondInteraction &I)
    {
      this->friction = I.friction;
      this->en = I.en;
      this->et = I.et;
      this->dn0 = I.dn0;
    }


    ONIKA_HOST_DEVICE_FUNC bool operator==(InnerBondInteraction& I)
    {
      return (pair == I.pair);
    }

    ONIKA_HOST_DEVICE_FUNC bool operator==(const InnerBondInteraction& I) const
    {
      return (pair == I.pair);
    }

    ONIKA_HOST_DEVICE_FUNC bool operator<(InnerBondInteraction& I)
    {
      return (pair < I.pair);
    }

    ONIKA_HOST_DEVICE_FUNC bool operator<(const InnerBondInteraction& I) const
    {
      return (pair < I.pair);
    }
  };

} // namespace exaDEM
