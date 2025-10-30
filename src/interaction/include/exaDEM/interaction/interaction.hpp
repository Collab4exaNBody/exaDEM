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

    ParticleSubLocation& i() { return pair.pi; }
    ParticleSubLocation& j() { return pair.pj; }
    ParticleSubLocation& driver() { return j(); }
    uint16_t type() { return pair.type; } 
    uint16_t type() const { return pair.type; } 
    uint16_t cell() { return pair.pi.cell; } // associate cell -> cell_i
    uint16_t partner_cell() { return pair.pj.cell; } // associate cell -> cell_i
    uint64_t driver_id() { return pair.pj.id; }

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
			bool is_ghost = (pair.pi.id >= pair.pj.id && pair.type <= 3);
			bool res = ((moment != null) || (friction != null)) && (!is_ghost);
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


		ONIKA_HOST_DEVICE_FUNC void update_friction_and_moment(Interaction &I)
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
} // namespace exaDEM
