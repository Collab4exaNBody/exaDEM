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
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_stream.h>
#include <exaDEM/interaction/interaction.hpp>

namespace exaDEM
{
	/**
	 * @brief Structure representing the Arrays of Structure data structure for the interactions in a Discrete Element Method (DEM) simulation.
	 */
	struct InteractionAOS
	{
		template <typename T> using VectorT =  onika::memory::CudaMMVector<T>;
		
		VectorT<exaDEM::Interaction> interactions;  /**<  List of interactions.  */
		
		/**
		 *@briefs Clears the interactions.
		 */
		void clear()
		{
			interactions.clear();
		}
		
		/**
		 *@briefs Returns the number of interactions.
		 */
		const size_t size() const
		{
			return interactions.size();
		}
		
		size_t size()
		{
			return interactions.size();
		}
		
		
		/**
		 *@briefs Fills the list of interactions.
		 */
		void insert(std::vector<exaDEM::Interaction> tmp)
		{
			interactions.insert(interactions.end(), tmp.begin(), tmp.end());
		}
		
		
		/**
		 *@briefs Returns an interaction for a given index of the interactions's list.
		 */
		ONIKA_HOST_DEVICE_FUNC exaDEM::Interaction operator[](uint64_t id) const
		{
			auto* ints = onika::cuda::vector_data(interactions);
			exaDEM::Interaction item = ints[id];
			return item;
		}
		
		/**
		 *@briefs Updates the friction and moment of a givne interaction.
		 */
		ONIKA_HOST_DEVICE_FUNC void update(size_t id, exaDEM::Interaction item)
		{
			exaDEM::Interaction& item2 = onika::cuda::vector_data(interactions)[id];
			item2.friction = item.friction;
			item2.moment = item.moment;
		}
		

		
		
	};
}
