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

#include <exaDEM/classifier/classifier.hpp>

namespace exaDEM
{
  /**
   * @brief Classifies interactions into categorized waves based on their types.
   *
   * This function categorizes interactions into different waves based on their types,
   * utilizing the `waves` vector in the `Classifier` struct. It resets existing waves,
   * calculates the number of interactions per wave, and then stores interactions
   * accordingly.
   *
   * @param ges Reference to the GridCellParticleInteraction object containing interactions to classify.
   */
  inline void classify(Classifier& classifier, GridCellParticleInteraction &ges, size_t *idxs, size_t size)
  {
    using namespace onika::cuda;
    constexpr int spti = InteractionTypeId::InnerBond;
    constexpr int ntypes = InteractionTypeId::NTypes;

    classifier.reset_waves();          // Clear existing waves
    auto &ces = ges.m_data; // Reference to cells containing interactions

    size_t n_threads;
#     pragma omp parallel
    {
      n_threads = omp_get_num_threads();
    }

    std::vector< std::array<std::pair<size_t,size_t>, ntypes> > bounds;
    bounds.resize(n_threads);

#     pragma omp parallel
    {
      size_t threads = omp_get_thread_num();
      std::array<std::vector<exaDEM::PlaceholderInteraction>, ntypes> tmp; ///< Storage for interactions categorized by type.

#       pragma omp for schedule(static) nowait
      for (size_t c = 0; c < size; c++)
      {
        auto &interactions = ces[idxs[c]];
        const unsigned int n_interactions_in_cell = interactions.m_data.size();
        auto * const __restrict__ data_ptr = interactions.m_data.data();
        // Place interactions into their respective waves
        for (size_t it = 0; it < n_interactions_in_cell; it++)
        {
          auto& item = data_ptr[it];
          const int t = item.type();
          tmp[t].push_back(item);
          item.reset();
        }
      }

      for (int interaction_type = 0; interaction_type < ntypes; interaction_type++)
      {
        bounds[threads][interaction_type].second = tmp[interaction_type].size();
      }

#pragma omp barrier   

      // All
      auto& bound = bounds[threads];
      for (int interaction_type = 0; interaction_type < ntypes; interaction_type++) 
      {
        size_t start = 0;
        for ( size_t i = 0 ; i < threads ; i++)
        {
          start += bounds[i][interaction_type].second;
        }
        bound[interaction_type].first = start;
      }

#pragma omp barrier

      // Partial
#pragma omp for
      for (int interaction_type = 0; interaction_type<ntypes ; interaction_type++)
      {
        if( interaction_type < InteractionTypeId::NTypesParticleParticle )
        {
          size_t size = bounds[n_threads-1][interaction_type].first + bounds[n_threads-1][interaction_type].second;
          auto& data = classifier.get_data<ParticleParticle>(interaction_type);
          data.resize(size);
        }
        else if (interaction_type == spti)
        {
          size_t size = bounds[n_threads-1][spti].first + bounds[n_threads-1][spti].second;
          classifier.get_data<InnerBond>(spti).resize(size);
        }
      }
#pragma omp barrier

      // All
      for (int interaction_type = 0; interaction_type < InteractionTypeId::NTypesParticleParticle; interaction_type++)
      {
        auto& data = classifier.get_data<ParticleParticle>(interaction_type);
        data.copy(
            bound[interaction_type].first, 
            bound[interaction_type].second, 
            tmp[interaction_type], 
            interaction_type);
      }
      classifier.get_data<InnerBond>(spti).copy(
          bound[spti].first, 
          bound[spti].second, 
          tmp[spti], 
          spti);
    }
  }



  template<template<InteractionType> class Wave, InteractionType IT>
    void unclassify_core(GridCellParticleInteraction &ges, Wave<IT>& data, size_t n1)
    {
      using namespace onika::cuda;
      auto &ces = ges.m_data; // Reference to cells containing interactions
                              // Parallel loop to process interactions within a wave
#         pragma omp for schedule(guided) nowait
      for (size_t it = 0; it < n1; it++)
      {
        auto item1 = data[it];
        // Check if interaction in wave has non-zero friction and moment
        if (item1.is_active()) // alway true if unclassify is called after compress
        {
          auto &celli = ces[item1.cell()];
          const unsigned int ni = vector_size(celli.m_data);
          PlaceholderInteraction * __restrict__ data_i_ptr = vector_data(celli.m_data);
          // Iterate through interactions in cell to find matching interaction
          bool find = false;
          for (size_t it2 = 0; it2 < ni ; it2++)
          {
            auto &item2 = (data_i_ptr[it2]).convert<IT>();
            if (item1 == item2)
            {
              item2.update(item1);
              find = true;
              break;
            }
          }

          if( find || (item1.type() >= 4 /** drivers */)) continue;

          // check if this interaction is included into the other cell
          auto& cellj = ces[item1.partner_cell()];
          const unsigned int nj = vector_size(cellj.m_data);
          PlaceholderInteraction * __restrict__ data_j_ptr = vector_data(cellj.m_data);

          for (size_t it2 = 0; it2 < nj; it2++)
          {
            auto &item2 = data_j_ptr[it2].convert<IT>();
            if (item1 == item2)
            {
              item2.update(item1);
              find = true;
              break;
            }
          }

          if(!find)
          {
            item1.print();
            color_log::error("unclassify", "One active interaction has not been updated");
          }
        }
      }
    }

  /**
   * @brief Restores friction and moment data for interactions from categorized waves to cell interactions.
   *
   * This function restores friction and moment data from categorized waves back to their corresponding
   * interactions in cell data (`ges.m_data`). It iterates through each wave, retrieves interactions
   * with non-zero friction and moment from the wave, and updates the corresponding interaction in the
   * cell data.
   *
   * @param ges Reference to the GridCellParticleInteraction object containing cell interactions.
   */
  inline void unclassify(Classifier& classifier, GridCellParticleInteraction &ges)
  {
    using namespace onika::cuda;
#     pragma omp parallel
    {
      for (int interacion_type = 0; interacion_type < InteractionTypeId::NTypes; interacion_type++)
      {
        if( interacion_type < InteractionTypeId::NTypesParticleParticle )
        {
          auto [data, n1] = classifier.get_info<ParticleParticle>(interacion_type);
          unclassify_core(ges, data, n1);
        }
        else if( interacion_type == InteractionTypeId::InnerBond )
        {
          auto [data, n1] = classifier.get_info<InnerBond>(interacion_type);
          unclassify_core(ges, data, n1);
        }
      }
    }
  }
} // namespace exaDEM
