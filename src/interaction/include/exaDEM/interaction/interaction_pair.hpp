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
#include <exaDEM/interaction/interaction_enum.hpp>

namespace exaDEM
{
  struct ParticleSubLocation
  {
    uint64_t id;                     /**< Id of the first particle */
    uint32_t cell;                   /**< Index of the cell of the first particle involved in the interaction. */
    uint16_t p;                      /**< Index of the particle within its cell for the particle involved in the interaction. */
    uint32_t sub;                    /**< Sub-particle index for the particle involved in the interaction. */

    ONIKA_HOST_DEVICE_FUNC bool operator==(ParticleSubLocation& P)
    {
      return (id == P.id && cell == P.cell && p == P.p && sub == P.sub);
    }

    ONIKA_HOST_DEVICE_FUNC bool operator==(const ParticleSubLocation& P) const
    {
      return (id == P.id && cell == P.cell && p == P.p && sub == P.sub);
    }
  };

  /**
   * @brief Structure representing an interaction in a Discrete Element Method (DEM) simulation.
   */
  struct InteractionPair
  {
    static constexpr uint8_t NotGhost = 0; // means that the interaction is linked to two particles that are in the current subdomain
    static constexpr uint8_t OwnerGhost = 1; // means that the interaction is linked to a particle that is in the current subdomain
    static constexpr uint8_t PartnerGhost = 2; // means that the interaction is linked to a particle that is in the current subdomain

    ParticleSubLocation pi;
    ParticleSubLocation pj;
    uint16_t type = InteractionTypeId::Undefined; /**< Type of the interaction (e.g., contact type). */
    uint8_t swap = false;
    uint8_t ghost = NotGhost;

    ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& owner()
    {
      if(!swap) return pi;
      return pj; 
    } 

    ONIKA_HOST_DEVICE_FUNC inline ParticleSubLocation& partner()
    {
      if(!swap) return pj;
      return pi; 
    } 

    ONIKA_HOST_DEVICE_FUNC inline const ParticleSubLocation& owner() const
    {
      if(!swap) return pi;
      return pj; 
    } 

    ONIKA_HOST_DEVICE_FUNC inline const ParticleSubLocation& partner() const
    {
      if(!swap) return pj;
      return pi; 
    } 

    ONIKA_HOST_DEVICE_FUNC inline bool active() { return ghost != PartnerGhost; }
    ONIKA_HOST_DEVICE_FUNC inline bool active() const { return ghost != PartnerGhost; }

    /**
     * @brief Displays the Interaction data.
     */
    void print()
    {
      auto [id_i, cell_i, p_i, sub_i] = owner();
      auto [id_j, cell_j, p_j, sub_j] = partner();
      std::cout << "Interaction(type = " << int(type)
        << " [cell: " << cell_i << ", idx " << p_i << ", particle id: " << id_i << ", sub: "<< sub_i <<"] and"
        << " [cell: " << cell_j << ", idx " << p_j << ", particle id: " << id_j << ", sub: "<< sub_j <<"]" 
        << " swap: " << int(swap) << " ghost: " << int(ghost) << ")" <<std::endl;
    }

    /**
     * @brief Displays the Interaction data.
     */
    void print() const
    {
      auto [id_i, cell_i, p_i, sub_i] = owner();
      auto [id_j, cell_j, p_j, sub_j] = partner();
      std::cout << "Interaction(type = " << int(type)
        << " [cell: " << cell_i << ", idx " << p_i << ", particle id: " << id_i << ", sub: "<< sub_i <<"] and"
        << " [cell: " << cell_j << ", idx " << p_j << ", particle id: " << id_j << ", sub: "<< sub_j <<"]" 
        << " swap: " << int(swap) << " ghost: " << int(ghost) << ")" <<std::endl;
    }

    /**
     * @brief return true if particles id and particles sub id are equals.
     */
    ONIKA_HOST_DEVICE_FUNC bool operator==(InteractionPair& I)
    {
      auto& me_pi = this->pi;
      auto& you_pi = I.pi;
      auto& me_pj = this->pj;
      auto& you_pj = I.pj;

      if (me_pi.id == you_pi.id 
          && me_pj.id == you_pj.id
          && me_pi.sub == you_pi.sub
          && me_pj.sub == you_pj.sub
          && this->swap == I.swap
          && this->type == I.type)
      {
        return true;
      }
      else
      {
        return false;
      }
    }

    /**
     * @brief return true if particles id and particles sub id are equals.
     */
    ONIKA_HOST_DEVICE_FUNC bool operator==(const InteractionPair& I) const
    {
      auto& me_pi = this->pi;
      auto& you_pi = I.pi;
      auto& me_pj = this->pj;
      auto& you_pj = I.pj;

      if (me_pi.id == you_pi.id 
          && me_pj.id == you_pj.id
          && me_pi.sub == you_pi.sub
          && me_pj.sub == you_pj.sub
          && this->swap == I.swap
          && this->type == I.type)
      {
        return true;
      }
      else
      {
        return false;
      }
    }

    ONIKA_HOST_DEVICE_FUNC bool operator<(const InteractionPair& I) const
    {
      const auto &a = owner();
      const auto &b = partner();
      const auto &c = I.owner();
      const auto &d = I.partner();

      if (a.id != c.id) return a.id < c.id;
      if (b.id != d.id) return b.id < d.id;

      auto t1 = type;
      auto t2 = I.type;
      if (t1 != t2) return t1 < t2;

      if(swap != I.swap) return swap < I.swap;

      if (a.sub != c.sub) return a.sub < c.sub;
      return b.sub < d.sub;
    }
  };
} // namespace exaDEM
