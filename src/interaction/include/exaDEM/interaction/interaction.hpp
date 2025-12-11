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
  struct Interaction 
  {
    InteractionPair pair;
    exanb::Vec3d friction = {0, 0, 0}; /**< Friction vector associated with the interaction. */
    exanb::Vec3d moment = {0, 0, 0};   /**< Moment vector associated with the interaction. */

    ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& i() { return pair.pi; }
    ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& j() { return pair.pj; }
    ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& owner() { return pair.owner(); }
    ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& partner() { return pair.partner(); }
    ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& driver() { return j(); }
    ONIKA_HOST_DEVICE_FUNC inline uint16_t type() { return pair.type; } 
    ONIKA_HOST_DEVICE_FUNC inline uint16_t type() const { return pair.type; } 
    ONIKA_HOST_DEVICE_FUNC inline uint32_t cell() { return pair.owner().cell; } // associate cell -> cell_i
    ONIKA_HOST_DEVICE_FUNC inline uint64_t driver_id() { return pair.pj.id; }
    ONIKA_HOST_DEVICE_FUNC inline InteractionPair& pair_info() { return pair; }
    ONIKA_HOST_DEVICE_FUNC inline const InteractionPair& pair_info() const { return pair; }

    /**
     * @brief Resets the Interaction structure by setting friction and moment vectors to zero.
     */
    ONIKA_HOST_DEVICE_FUNC void reset()
    {
      constexpr exanb::Vec3d null = {0, 0, 0};
      friction = null;
      moment = null;
    }

    // Defines whether an interaction will be reconstructed or not.
    ONIKA_HOST_DEVICE_FUNC bool persistent() { return false; }
    // Skip other interactions if this interaction is defined
    //ONIKA_HOST_DEVICE_FUNC bool ignore_other_interactions() { return true; } 
    ONIKA_HOST_DEVICE_FUNC bool ignore_other_interactions() { return false; } 

    /**
     * @brief Checks if the interaction is active.
     *
     * This function checks if the interaction is active by examining the moment and friction vectors.
     * An interaction is considered active if either the moment vector or the friction vector is non-zero.
     *
     * @return True if the interaction is active (moment vector or friction vector is non-zero), false otherwise.
     */
    ONIKA_HOST_DEVICE_FUNC bool active() const
    {
      if( !pair.active() ) return false;
      constexpr exanb::Vec3d null = {0, 0, 0};
      bool res = ((moment != null) || (friction != null));
      return res;
    }

    /**
     * @brief Displays the Interaction data.
     */
    void print()
    {
      pair.print();
      std::cout << "Friction: " << friction << ", Moment: " << moment << ")" << std::endl;
    }

    /**
     * @brief Displays the Interaction data.
     */
    void print() const
    {
      pair.print();
      std::cout << "Friction: " << friction << ", Moment: " << moment << ")" << std::endl;
    }


    ONIKA_HOST_DEVICE_FUNC void update(Interaction &I)
    {
      this->friction = I.friction;
      this->moment = I.moment;
    }

    ONIKA_HOST_DEVICE_FUNC bool operator==(Interaction& I)
    {
      return (pair == I.pair);
    }

    ONIKA_HOST_DEVICE_FUNC bool operator==(const Interaction& I) const
    {
      return (pair == I.pair);
    }

    ONIKA_HOST_DEVICE_FUNC bool operator<(Interaction& I)
    {
      return (pair < I.pair);
    }

    ONIKA_HOST_DEVICE_FUNC bool operator<(const Interaction& I) const
    {
      return (pair < I.pair);
    }
  };
} // namespace exaDEM
