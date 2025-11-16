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
#include <onika/cuda/stl_adaptors.h>
#include <exaDEM/color_log.hpp>
#include <exaDEM/interaction/placeholder_interaction.hpp>

namespace exaDEM
{

  /**
   * @brief Structure representing the Structure of Arrays data structure for the interactions in a Discrete Element Method (DEM) simulation.
   */

  template<InteractionType IT>
    struct ClassifierContainer
    {
      template <typename T> using VectorT = onika::memory::CudaMMVector<T>;

      VectorT<double> ft_x; /**< List of the x coordinate for the friction.  */
      VectorT<double> ft_y; /**< List of the y coordinate for the friction.  */
      VectorT<double> ft_z; /**< List of the z coordinate for the friction.  */

      VectorT<double> mom_x; /**< List of the x coordinate for the moment.  */
      VectorT<double> mom_y; /**< List of the y coordinate for the moment.  */
      VectorT<double> mom_z; /**< List of the z coordinate for the moment.  */

      VectorT<double> en; /**< List of the en.  */
      VectorT<double> et; /**< List of the et.  */
      VectorT<double> dn0; /**< List of the dn0.  */
      VectorT<double> criterion; /**< List of the criterion.  */
      VectorT<uint8_t> unbroken; /**< List of the sticked interactions are unbroken.  */

      VectorT<uint64_t> id_i; /**< List of the ids of the first particle involved in the interaction.  */
      VectorT<uint64_t> id_j; /**< List of the ids of the second particle involved in the interaction.  */

      VectorT<uint32_t> cell_i; /**< List of the indexes of the cell for the first particle involved in the interaction.  */
      VectorT<uint32_t> cell_j; /**< List of the indexes of the cell for the second particle involved in the interaction.  */

      VectorT<uint16_t> p_i; /**< List of the indexes of the particle within its cell for the first particle involved in the interaction. */
      VectorT<uint16_t> p_j; /**< List of the indexes of the particle within its cell for the second particle involved in the interaction.  */

      VectorT<uint32_t> sub_i; /**< List of the sub-particle indexes for the first particle involved in the interaction.  */
      VectorT<uint32_t> sub_j; /**< List of the sub-particle indexes for the first particle involved in the interaction.  */

      uint16_t type; /**< Type of the interaction (e.g., contact type). */

      template<typename Func, typename Field>
        void apply_on_field (Func& func, Field& field)
        {
          func(field);
        }

      template<typename Func, typename Field, typename... Fields>
        void apply_on_fields (Func& func, Field& field, Fields& ... fields)
        {
          apply_on_field(func, field);
          if constexpr ( sizeof...(fields) > 0) apply_on_fields(func, fields...);
        }


      template<typename Func>
        void apply_on_fields (Func& func)
        {
          //ldbg << "Func applied on [id_i, id_j, cell_i, cell_j, p_i, p_j, sub_i, sub_j]" << std::endl;
          apply_on_fields(func, id_i, id_j, cell_i, cell_j, p_i, p_j, sub_i, sub_j);
          if constexpr (IT == InteractionType::ParticleParticle || IT == InteractionType::ParticleDriver) 
          {
            //ldbg << "Func applied on [ft_x, ft_y, ft_z, mom_x, mom_y, mom_z]" << std::endl;
            apply_on_fields(func, ft_x, ft_y, ft_z, mom_x, mom_y, mom_z);
          }
          if constexpr (IT == InteractionType::StickedParticles)
          {
            //ldbg << "Func applied on [ft_x, ft_y, ft_z, en, et, dn0, s(surface)]" << std::endl;
            apply_on_fields(func, ft_x, ft_y, ft_z, en, et, dn0, criterion, unbroken);
          }
        }

      struct ClearFunctor{
        template<typename T> inline void operator()(T& vec) { vec.clear(); }
      };
      /**
       *@briefs CLears all the lists.
       */
      void clear()
      {

        ClearFunctor func;
        apply_on_fields(func);
      }

      struct ResizeFunctor{
        const size_t size;
        template<typename T> inline void operator()(T& vec) {vec.resize(size);}
      };

      /**
       * briefs Resize all the lists.
       */
      void resize(const size_t size)
      {

        ResizeFunctor func = {size};
        apply_on_fields(func);
      }

      /**
       * briefs Returns the number of interactions.
       */
      ONIKA_HOST_DEVICE_FUNC size_t size() const { return onika::cuda::vector_size(ft_x); }
      ONIKA_HOST_DEVICE_FUNC size_t size() { return onika::cuda::vector_size(ft_x); }

      // Some accessors
      ONIKA_HOST_DEVICE_FUNC uint64_t particle_id_i(size_t idx) const { return id_i[idx]; }
      ONIKA_HOST_DEVICE_FUNC uint64_t particle_id_j(size_t idx) const { return id_j[idx]; }


			ONIKA_HOST_DEVICE_FUNC void set(
					size_t idx, 
					exaDEM::PlaceholderInteraction& interaction)
			{
				if constexpr (IT == InteractionType::ParticleParticle || IT == InteractionType::ParticleDriver)
				{ 
					auto& I = (Interaction&) (interaction);
					ft_x[idx] = I.friction.x;
					ft_y[idx] = I.friction.y;
					ft_z[idx] = I.friction.z;

					mom_x[idx] = I.moment.x;
					mom_y[idx] = I.moment.y;
					mom_z[idx] = I.moment.z;
				}

				if constexpr (IT == InteractionType::StickedParticles) 
				{
					auto& I = (InnerBondInteraction&) (interaction);
					ft_x[idx] = I.friction.x;
					ft_y[idx] = I.friction.y;
					ft_z[idx] = I.friction.z;

					en[idx] = I.en;
					et[idx] = I.et;
					dn0[idx] = I.dn0;
					criterion[idx] = I.criterion;
					unbroken[idx] = I.unbroken;
				}

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
			void insert(std::vector<exaDEM::PlaceholderInteraction> &tmp, int w)
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

			void copy(size_t start, size_t size, std::vector<exaDEM::PlaceholderInteraction> &tmp, int w)
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
			ONIKA_HOST_DEVICE_FUNC auto operator[](uint64_t id) 
			{
				using namespace onika::cuda;
				InteractionPair ip = {
					// pi
					{  vector_data(id_i)[id],
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

				if constexpr (IT == InteractionType::ParticleParticle || IT == InteractionType::ParticleDriver) 
				{
					exaDEM::Interaction res{ ip,
						{vector_data(ft_x)[id],
							vector_data(ft_y)[id], 
							vector_data(ft_z)[id]},
						{vector_data(mom_x)[id],
							vector_data(mom_y)[id],
							vector_data(mom_z)[id]}};
					return res;
				}
				else if constexpr (IT == InteractionType::StickedParticles)
				{
					exaDEM::InnerBondInteraction res{ ip,
						{vector_data(ft_x)[id],
							vector_data(ft_y)[id], 
							vector_data(ft_z)[id]},
						vector_data(en)[id],
						vector_data(et)[id],
						vector_data(dn0)[id],
						vector_data(criterion)[id],
						vector_data(unbroken)[id]
					};
					return res;

				}
			}

			/**
			 *@briefs Updates the friction and moment of a given interaction.
			 */
			ONIKA_HOST_DEVICE_FUNC void update(size_t id, PlaceholderInteraction &item)
			{
				using namespace onika::cuda;

				if constexpr (IT == InteractionType::ParticleParticle || IT == InteractionType::ParticleDriver) 
				{
					auto& I = (Interaction&) (item);
					vector_data(ft_x)[id] = I.friction.x;
					vector_data(ft_y)[id] = I.friction.y;
					vector_data(ft_z)[id] = I.friction.z;

					vector_data(mom_x)[id] = I.moment.x;
					vector_data(mom_y)[id] = I.moment.y;
					vector_data(mom_z)[id] = I.moment.z;
				}

				if constexpr (IT == InteractionType::StickedParticles) 
				{
					auto& I = (InnerBondInteraction&) (item);
					vector_data(ft_x)[id] = I.friction.x;
					vector_data(ft_y)[id] = I.friction.y;
					vector_data(ft_z)[id] = I.friction.z;

					vector_data(en)[id] = I.en;
					vector_data(et)[id] = I.et;
					vector_data(dn0)[id] = I.dn0;
					vector_data(criterion)[id] = I.criterion;
					vector_data(unbroken)[id] = I.unbroken;
				}
			}
		};
} // namespace exaDEM
