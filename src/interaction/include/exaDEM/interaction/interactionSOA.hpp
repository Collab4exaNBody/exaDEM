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
   * @brief Structure representing the Structure of Arrays data structure for the interactions in a Discrete Element Method (DEM) simulation.
   */
  struct InteractionSOA
  {
    template <typename T> using VectorT = onika::memory::CudaMMVector<T>;

    VectorT<double> ft_x; /**< List of the x coordinate for the friction.  */
    VectorT<double> ft_y; /**< List of the y coordinate for the friction.  */
    VectorT<double> ft_z; /**< List of the z coordinate for the friction.  */

    VectorT<double> mom_x; /**< List of the x coordinate for the moment.  */
    VectorT<double> mom_y; /**< List of the y coordinate for the moment.  */
    VectorT<double> mom_z; /**< List of the z coordinate for the moment.  */

    VectorT<uint64_t> id_i; /**< List of the ids of the first particle involved in the interaction.  */
    VectorT<uint64_t> id_j; /**< List of the ids of the second particle involved in the interaction.  */

    VectorT<uint32_t> cell_i; /**< List of the indexes of the cell for the first particle involved in the interaction.  */
    VectorT<uint32_t> cell_j; /**< List of the indexes of the cell for the second particle involved in the interaction.  */

    VectorT<uint16_t> p_i; /**< List of the indexes of the particle within its cell for the first particle involved in the interaction. */
    VectorT<uint16_t> p_j; /**< List of the indexes of the particle within its cell for the second particle involved in the interaction.  */

    VectorT<uint16_t> sub_i; /**< List of the sub-particle indexes for the first particle involved in the interaction.  */
    VectorT<uint16_t> sub_j; /**< List of the sub-particle indexes for the first particle involved in the interaction.  */

    uint16_t type; /**< Type of the interaction (e.g., contact type). */

    /**
     *@briefs CLears all the lists.
     */
    void clear()
    {
      ft_x.clear();
      ft_y.clear();
      ft_z.clear();

      mom_x.clear();
      mom_y.clear();
      mom_z.clear();

      id_i.clear();
      id_j.clear();

      cell_i.clear();
      cell_j.clear();

      p_i.clear();
      p_j.clear();

      sub_i.clear();
      sub_j.clear();
    }

    /**
     *briefs Returns the number of interactions.
     */
    ONIKA_HOST_DEVICE_FUNC size_t size() const { return onika::cuda::vector_size(ft_x); }

    ONIKA_HOST_DEVICE_FUNC size_t size() { return onika::cuda::vector_size(ft_x); }

    /**
     *@briefs Fills the lists.
     */
    void insert(std::vector<exaDEM::Interaction> &tmp, int w)
    {
      for (auto interaction : tmp)
      {
        ft_x.push_back(interaction.friction.x);
        ft_y.push_back(interaction.friction.y);
        ft_z.push_back(interaction.friction.z);

        mom_x.push_back(interaction.moment.x);
        mom_y.push_back(interaction.moment.y);
        mom_z.push_back(interaction.moment.z);

        id_i.push_back(interaction.id_i);
        id_j.push_back(interaction.id_j);

        cell_i.push_back(interaction.cell_i);
        cell_j.push_back(interaction.cell_j);

        p_i.push_back(interaction.p_i);
        p_j.push_back(interaction.p_j);

        sub_i.push_back(interaction.sub_i);
        sub_j.push_back(interaction.sub_j);
      }

      type = w;
    }

    /**
     *@briefs Return the interaction for a given list.
     */
    ONIKA_HOST_DEVICE_FUNC exaDEM::Interaction operator[](uint64_t id) { return {{onika::cuda::vector_data(ft_x)[id], onika::cuda::vector_data(ft_y)[id], onika::cuda::vector_data(ft_z)[id]}, {onika::cuda::vector_data(mom_x)[id], onika::cuda::vector_data(mom_y)[id], onika::cuda::vector_data(mom_z)[id]}, onika::cuda::vector_data(id_i)[id], onika::cuda::vector_data(id_j)[id], onika::cuda::vector_data(cell_i)[id], onika::cuda::vector_data(cell_j)[id], onika::cuda::vector_data(p_i)[id], onika::cuda::vector_data(p_j)[id], onika::cuda::vector_data(sub_i)[id], onika::cuda::vector_data(sub_j)[id], type}; }

    /**
     *@briefs Updates the friction and moment of a given interaction.
     */
    ONIKA_HOST_DEVICE_FUNC void update(size_t id, exaDEM::Interaction &item)
    {
      onika::cuda::vector_data(ft_x)[id] = item.friction.x;
      onika::cuda::vector_data(ft_y)[id] = item.friction.y;
      onika::cuda::vector_data(ft_z)[id] = item.friction.z;

      onika::cuda::vector_data(mom_x)[id] = item.moment.x;
      onika::cuda::vector_data(mom_y)[id] = item.moment.y;
      onika::cuda::vector_data(mom_z)[id] = item.moment.z;
    }
  };
} // namespace exaDEM
