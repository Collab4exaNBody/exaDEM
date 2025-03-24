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
#include <exaDEM/interaction/interaction.hpp>

namespace exaDEM
{
  /**
   * @brief Structure representing the Arrays of Structure data structure for the m_data in a Discrete Element Method (DEM) simulation.
   */
  struct InteractionAOS
  {
    template <typename T> using VectorT = onika::memory::CudaMMVector<T>;

    VectorT<exaDEM::Interaction> m_data; /**<  List of m_data.  */

    /**
     *@briefs Clears the m_data.
     */
    void clear() { m_data.clear(); }

    /**
     *@briefs Returns the number of m_data.
     */
    size_t size() const { return m_data.size(); }

    size_t size() { return m_data.size(); }

    /**
     *@briefs Fills the list of m_data.
     */
    void insert(const std::vector<exaDEM::Interaction> &tmp, int w) { m_data.insert(m_data.end(), tmp.begin(), tmp.end()); }

    /**
     *@briefs Returns an interaction for a given index of the m_data's list.
     */
    ONIKA_HOST_DEVICE_FUNC exaDEM::Interaction operator[](uint64_t id) const
    {
      auto *ints = onika::cuda::vector_data(m_data);
      exaDEM::Interaction item = ints[id];
      return item;
    }

    /**
     *@briefs Updates the friction and moment of a givne interaction.
     */
    ONIKA_HOST_DEVICE_FUNC void update(size_t id, exaDEM::Interaction &item)
    {
      exaDEM::Interaction &item2 = onika::cuda::vector_data(m_data)[id];
      item2.friction = item.friction;
      item2.moment = item.moment;
    }
  };
} // namespace exaDEM
