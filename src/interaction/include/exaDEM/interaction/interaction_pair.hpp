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

    static constexpr uint8_t NotGhost = 0; // this interaction doesn't occurs 
    static constexpr uint8_t OwnerGhost = 1; // means that the interaction is linked to a particle that is in the current subdomain
    static constexpr uint8_t PartnerGhost = 2; // means that the interaction is linked to a particle that is in the current subdomain

    ParticleSubLocation pi;
    ParticleSubLocation pj;
    uint16_t type;                     /**< Type of the interaction (e.g., contact type). */
    uint8_t swap = false;
    uint8_t ghost = NotGhost;

    ParticleSubLocation& owner()
    {
      if(!swap) return pi;
      return pj; 
    } 

    ParticleSubLocation& partner()
    {
      if(!swap) return pj;
      return pi; 
    } 

    const ParticleSubLocation& owner() const
    {
      if(!swap) return pi;
      return pj; 
    } 

    const ParticleSubLocation& partner() const
    {
      if(!swap) return pj;
      return pi; 
    } 

    bool active() { return ghost != PartnerGhost; }
    bool active() const { return ghost != PartnerGhost; }

    /**
     * @brief Displays the Interaction data.
     */
    void print()
    {
      auto [id_i, cell_i, p_i, sub_i] = owner();
      auto [id_j, cell_j, p_j, sub_j] = partner();
      std::cout << "Interaction(type = " << int(type)
        << " [cell: " << cell_i << ", idx " << p_i << ", particle id: " << id_i << "] and"
        << " [cell: " << cell_j << ", idx " << p_j << ", particle id: " << id_j << "]" 
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
        << " [cell: " << cell_i << ", idx " << p_i << ", particle id: " << id_i << "] and"
        << " [cell: " << cell_j << ", idx " << p_j << ", particle id: " << id_j << "]" 
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

      if (me_pi == you_pi && me_pj == you_pj && this->type == I.type)
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

      if (me_pi == you_pi && me_pj == you_pj && this->type == I.type)
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
/*      auto me_id_i = owner().id;
      auto me_id_j = partner().id;
      auto me_sub_i = pi.sub;
      auto me_sub_j = pj.sub;

      auto you_id_i = I.owner().id;
      auto you_id_j = I.partner().id;
      auto you_sub_i = I.pi.sub;
      auto you_sub_j = I.pj.sub;
*/
      auto me_id_i = owner().id;
      auto you_id_i = I.owner().id;
      if (me_id_i < you_id_i)  { return true; }
//      else if (me_id_i == you_id_i && me_id_j < you_id_j) { return true; }
//      else if (me_id_i == you_id_i && me_id_j == you_id_j && me_sub_i < you_sub_i) { return true; }
//      else if (me_id_i == you_id_i && me_id_j == you_id_j && me_sub_i == you_sub_i && me_sub_j < you_sub_j) {  return true; }
//      else if (me_id_i == you_id_i && me_id_j == you_id_j && me_sub_i == you_sub_i && me_sub_j == you_sub_j && type < I.type) {  return true; }
      else return false;
/*      auto& me_pi = this->pi;
      auto& you_pi = I.pi;
      auto& me_pj = this->pj;
      auto& you_pj = I.pj;
      if (me_pi.id < you_pi.id)  { return true; }
      else if (me_pi.id == you_pi.id && me_pj.id < you_pj.id) { return true; }
      else if (me_pi.id == you_pi.id && me_pj.id == you_pj.id && me_pi.sub < you_pi.sub) { return true; }
      else if (me_pi.id == you_pi.id && me_pj.id == you_pj.id && me_pi.sub == you_pi.sub && me_pj.sub < you_pj.sub) {  return true; }
      else if (me_pi.id == you_pi.id && me_pj.id == you_pj.id && me_pi.sub == you_pi.sub && me_pj.sub == you_pj.sub && type < I.type) {  return true; }
      else return false;
*/
    }

    ONIKA_HOST_DEVICE_FUNC void update(InteractionPair& I)
    {
      this->pi.cell = I.pi.cell;
      this->pj.cell = I.pj.cell;
      this->pi.p = I.pi.p;
      this->pj.p = I.pj.p;
    }
  };
} // namespace exaDEM
