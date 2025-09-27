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
  /**
   * @brief Structure representing an interaction in a Discrete Element Method (DEM) simulation.
   */
  struct Interaction
  {
    exanb::Vec3d friction = {0, 0, 0}; /**< Friction vector associated with the interaction. */
    exanb::Vec3d moment = {0, 0, 0};   /**< Moment vector associated with the interaction. */
    uint64_t id_i;                     /**< Id of the first particle */
    uint64_t id_j;                     /**< Id of the second particle */
    uint32_t cell_i;                   /**< Index of the cell of the first particle involved in the interaction. */
    uint32_t cell_j;                   /**< Index of the cell of the second particle involved in the interaction. */
    uint16_t p_i;                      /**< Index of the particle within its cell for the first particle involved in the interaction. */
    uint16_t p_j;                      /**< Index of the particle within its cell for the second particle involved in the interaction. */
    uint16_t sub_i;                    /**< Sub-particle index for the first particle involved in the interaction. */
    uint16_t sub_j;                    /**< Sub-particle index for the second particle involved in the interaction. */
    uint16_t type;                     /**< Type of the interaction (e.g., contact type). */

    /**
     * @brief Resets the Interaction structure by setting friction and moment vectors to zero.
     */
    ONIKA_HOST_DEVICE_FUNC void reset()
    {
      constexpr exanb::Vec3d null = {0, 0, 0};
      friction = null;
      moment = null;
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
      constexpr exanb::Vec3d null = {0, 0, 0};
	  bool is_ghost = (id_i >= id_j && type <= 3);
      bool res = ((moment != null) || (friction != null)) && (!is_ghost);
      return res;
    }

    /**
     * @brief Displays the Interaction data.
     */
    void print()
    {
      std::cout << "Interaction(type = " << int(type) << " [cell: " << cell_i << ", idx " << p_i << ", particle id: " << id_i << "] and"
                << " [cell: " << cell_j << ", idx " << p_j << ", particle id: " << id_j << "] : (friction: " << friction << ", moment: " << moment << ")" << std::endl;
    }

    ONIKA_HOST_DEVICE_FUNC void PrintF()
    {
      printf("Interaction(type = %d [cell: %d, idx %d, particle id: %d] and [cell: %d , idx %d, particle id: %d]\n", 
             int(type), int(cell_i), int(p_i), int(id_i), int(cell_j), int(p_j), int(id_j));
    }

    /**
     * @brief Displays the Interaction data.
     */
    void print() const
    {
      std::cout << "Interaction(type = " << int(type) << " [cell: " << cell_i << ", idx " << id_i << ", particle id: " << p_i << "] and"
                << " [cell: " << cell_j << ", idx " << id_j << ", particle id: " << p_j << "] : (friction: " << friction << ", moment: " << moment << ")" << std::endl;
    }

    /**
     * @brief return true if particles id and particles sub id are equals.
     */
    ONIKA_HOST_DEVICE_FUNC bool operator==(Interaction &I)
		{
			if (this->id_i == I.id_i && 
					this->id_j == I.id_j && 
					this->sub_i == I.sub_i && 
					this->sub_j == I.sub_j && 
					this->type == I.type)
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
		ONIKA_HOST_DEVICE_FUNC bool operator==(const Interaction &I) const
		{
			if (this->id_i == I.id_i && 
					this->id_j == I.id_j && 
					this->sub_i == I.sub_i && 
					this->sub_j == I.sub_j && 
					this->type == I.type)
			{
				return true;
			}
			else
			{
				return false;
			}
		}

		ONIKA_HOST_DEVICE_FUNC bool operator<(const Interaction &I) const
		{
			if (this->id_i < I.id_i)
			{
				return true;
			}
			else if (this->id_i == I.id_i && this->id_j < I.id_j)
			{
				return true;
			}
			else if (this->id_i == I.id_i && this->id_j == I.id_j && this->sub_i < I.sub_i)
			{
				return true;
			}
			else if (this->id_i == I.id_i && this->id_j == I.id_j && this->sub_i == I.sub_i && this->sub_j < I.sub_j)
			{
				return true;
			}
			else if (this->id_i == I.id_i && this->id_j == I.id_j && this->sub_i == I.sub_i && this->sub_j == I.sub_j && this->type < I.type)
			{
				return true;
			}
			else
				return false;
		}

		ONIKA_HOST_DEVICE_FUNC void update(Interaction &I)
		{
			this->cell_i = I.cell_i;
			this->cell_j = I.cell_j;
			this->p_i = I.p_i;
			this->p_j = I.p_j;
		}

		ONIKA_HOST_DEVICE_FUNC void update_friction_and_moment(Interaction &I)
		{
			this->friction = I.friction;
			this->moment = I.moment;
		}
	};

	inline std::pair<bool, Interaction &> get_interaction(std::vector<Interaction> &list, Interaction &I)
	{
		auto iterator = std::find(list.begin(), list.end(), I);
		// assert(iterator == std::end(list) && "This interaction is NOT in the list");
		bool exist = iterator == std::end(list);
		return {exist, *iterator};
	}

	inline std::vector<Interaction> extract_history_omp(std::vector<Interaction> &interactions)
	{
		std::vector<Interaction> ret;
#   pragma omp parallel
		{
			std::vector<Interaction> tmp;
#     pragma omp for
			for (size_t i = 0; i < interactions.size(); i++)
			{
				if (interactions[i].is_active())
				{
					tmp.push_back(interactions[i]);
				}
			}

			if (tmp.size() > 0)
			{
#       pragma omp critical
				{
					ret.insert(ret.end(), tmp.begin(), tmp.end());
				}
			}
		}

		return ret;
	}

	inline void update_friction_moment_omp(std::vector<Interaction> &interactions, std::vector<Interaction> &history)
	{
#   pragma omp parallel for
		for (size_t it = 0; it < interactions.size(); it++)
		{
			auto &item = interactions[it];
			auto lower = std::lower_bound(history.begin(), history.end(), item);
			if (lower != history.end())
			{
				if (item == *lower)
				{
					item.update_friction_and_moment(*lower);
				}
			}
		}
	}

	inline void update_friction_moment(std::vector<Interaction> &interactions, std::vector<Interaction> &history)
	{
		for (size_t it = 0; it < interactions.size(); it++)
		{
			auto &item = interactions[it];
			auto lower = std::lower_bound(history.begin(), history.end(), item);
			if (lower != history.end())
			{
				if (item == *lower)
				{
					item.update_friction_and_moment(*lower);
				}
			}
		}
	}

	// sequential
	inline void extract_history(std::vector<Interaction> &local, const Interaction *data, const unsigned int size)
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
