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
    ParticleSubLocation pi;
    ParticleSubLocation pj;
    uint16_t type;                     /**< Type of the interaction (e.g., contact type). */

    /**
     * @brief Displays the Interaction data.
     */
    void print()
    {
      auto [id_i, cell_i, p_i, sub_i] = pi;
      auto [id_j, cell_j, p_j, sub_j] = pj;
			std::cout << "Interaction(type = " << int(type)
				<< " [cell: " << cell_i << ", idx " << p_i << ", particle id: " << id_i << "] and"
				<< " [cell: " << cell_j << ", idx " << p_j << ", particle id: " << id_j << "]" <<std::endl;
		}

		/**
		 * @brief Displays the Interaction data.
		 */
		void print() const
		{
      const auto [id_i, cell_i, p_i, sub_i] = pi;
      const auto [id_j, cell_j, p_j, sub_j] = pj;
			std::cout << "Interaction(type = " << int(type)
				<< " [cell: " << cell_i << ", idx " << p_i << ", particle id: " << id_i << "] and"
				<< " [cell: " << cell_j << ", idx " << p_j << ", particle id: " << id_j << "]" <<std::endl;
		}

		/**
		 * @brief return true if particles id and particles sub id are equals.
		 */
		ONIKA_HOST_DEVICE_FUNC bool operator==(InteractionPair& I)
		{
			if (this->pi == I.pi && this->pj == I.pj && this->type == I.type)
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
			if (this->pi == I.pi && this->pj == I.pj && this->type == I.type)
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
			if (this->pi.id < I.pi.id)
			{
				return true;
			}
			else if (this->pi.id == I.pi.id && this->pj.id < I.pj.id)
			{
				return true;
			}
			else if (this->pi.id == I.pi.id && this->pj.id == I.pj.id && this->pi.sub < I.pi.sub)
			{
				return true;
			}
			else if (this->pi.id == I.pi.id && this->pj.id == I.pj.id && this->pi.sub == I.pi.sub && this->pj.sub < I.pj.sub)
			{
				return true;
			}
			else if (this->pi.id == I.pi.id && this->pj.id == I.pj.id && this->pi.sub == I.pi.sub && this->pj.sub == I.pj.sub && this->type < I.type)
			{
				return true;
			}
			else
				return false;
		}

		ONIKA_HOST_DEVICE_FUNC void update(InteractionPair& I)
		{
			this->pi.cell = I.pi.cell;
			this->pj.cell = I.pj.cell;
			this->pi.p = I.pi.p;
			this->pj.p = I.pj.p;
		}
	};

  template<typename InteractionT>
	inline std::pair<bool, InteractionT&> get_interaction(std::vector<InteractionT> &list, InteractionT &I)
	{
		auto iterator = std::find(list.begin(), list.end(), I);
		// assert(iterator == std::end(list) && "This interaction is NOT in the list");
		bool exist = iterator == std::end(list);
		return {exist, *iterator};
	}

	// sequential
	template<typename InteractionT>
		inline void extract_history(std::vector<InteractionT> &local, const InteractionT *data, const unsigned int size)
		{
			local.clear();
			for (size_t i = 0; i < size; i++)
			{
				const auto &item = data[i];
				if (item.is_active())
				{
					local.push_back(item);
				}
			}
		}
} // namespace exaDEM
