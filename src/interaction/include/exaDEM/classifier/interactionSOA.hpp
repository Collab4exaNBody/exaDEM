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
#include <exaDEM/color_log.hpp>
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

    template<typename Func>
    void for_all (Func& func)
    {
      func(ft_x);
      func(ft_y);
      func(ft_z);
      func(mom_x);
      func(mom_y);
      func(mom_z);
      func(id_i);
      func(id_j);
      func(cell_i);
      func(cell_j);
      func(p_i);
      func(p_j);
      func(sub_i);
      func(sub_j);
    }

    struct ClearFunctor{
      template<typename T> inline void operator()(T& vec) {vec.clear();}
    };

    /**
     *@briefs CLears all the lists.
     */
    void clear()
    {
      ClearFunctor func;
      for_all(func);
    }

    struct ResizeFunctor{
      const size_t size;
      template<typename T> inline void operator()(T& vec) {vec.resize(size);}
    };
    /**
     *@briefs Resize all the lists.
     */
    void resize(const size_t size)
    {
      ResizeFunctor func = {size};
      for_all(func);
    }

    /**
     *briefs Returns the number of interactions.
     */
    ONIKA_HOST_DEVICE_FUNC size_t size() const { return onika::cuda::vector_size(ft_x); }

    ONIKA_HOST_DEVICE_FUNC size_t size() { return onika::cuda::vector_size(ft_x); }

    struct PrefetchMemoryFunctor
    {
      size_t size;
      int deviceId;
      onikaStream_t stream;

      template<typename T> inline void operator()(T& vec)
      {
        auto* data = onika::cuda::vector_data(vec);
        ONIKA_CU_MEM_PREFETCH(data, size, deviceId, stream);
      }
    };

    void prefetch_memory_on_gpu(int device_id, onikaStream_t stream)
    {
      PrefetchMemoryFunctor func = {this->size(), device_id, stream};
      for_all(func);
    }

		void set(size_t idx, exaDEM::Interaction& interaction)
		{
			ft_x[idx] = interaction.friction.x;
			ft_y[idx] = interaction.friction.y;
			ft_z[idx] = interaction.friction.z;

			mom_x[idx] = interaction.moment.x;
			mom_y[idx] = interaction.moment.y;
			mom_z[idx] = interaction.moment.z;

			auto& [pi, pj, type] = interaction.pair;

			id_i[idx] = pi.id;
			id_j[idx] = pj.id;

			cell_i[idx] = pi.cell;
			cell_j[idx] = pj.cell;

			p_i[idx] = pi.p;
			p_j[idx] = pj.p;

			sub_i[idx] = pi.sub;
			sub_j[idx] = pj.sub;
		}

		/**
		 *@briefs Fills the lists.
		 */
		void insert(std::vector<exaDEM::Interaction> &tmp, int w)
		{
			const size_t new_elements = tmp.size();
			const size_t old_size = this->size();
			this->resize(old_size + new_elements);

			type = w;

			for (size_t i = 0 ; i < new_elements ; i++)
			{
				const size_t idx = old_size + i;
				auto& interaction = tmp[i];
        set(idx, interaction);
			}
		}

		void copy(size_t start, size_t size, std::vector<exaDEM::Interaction> &tmp, int w)
		{
			if( tmp.size() != size ) 
			{
				color_log::error("Classifier::copy", "When resizing wave: " +std::to_string(w));
			}
			type = w;

			for (size_t i = 0 ; i < size ; i++)
			{
				const size_t idx = start + i;
				auto& interaction = tmp[i];
				set(idx, interaction);
			}
		}

		/**
		 *@briefs Return the interaction for a given list.
		 */
		ONIKA_HOST_DEVICE_FUNC exaDEM::Interaction operator[](uint64_t id) 
		{
			using namespace onika::cuda;
			InteractionPair ip = {
				// pi
				{	vector_data(id_i)[id],
					vector_data(cell_i)[id],
					vector_data(p_i)[id],
					vector_data(sub_i)[id]},
				// pj
				{ vector_data(id_j)[id],
					vector_data(cell_j)[id],
					vector_data(p_j)[id],
					vector_data(sub_j)[id]},
				// type
				type};

			exaDEM::Interaction res{ ip,
				{vector_data(ft_x)[id],
					vector_data(ft_y)[id], 
					vector_data(ft_z)[id]},
				{vector_data(mom_x)[id],
					vector_data(mom_y)[id],
					vector_data(mom_z)[id]}};
				return res;
		}

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
