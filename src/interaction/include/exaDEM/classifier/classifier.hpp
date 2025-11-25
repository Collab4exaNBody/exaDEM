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

#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/classifier/classifier_container.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/itools/buffer.hpp>
#include <exaDEM/classifier/interaction_wrapper.hpp>

namespace exaDEM
{
  /**
   * @brief Classifier for managing interactions categorized into different types.
   *
   * The Classifier struct manages interactions categorized into different types (up to 13 types).
   * It provides functionalities to store interactions in CUDA memory-managed vectors (`VectorT`).
   */
  struct Classifier
  {
    using WavePP = ClassifierContainer<InteractionType::ParticleParticle>;
    using WaveIB = ClassifierContainer<InteractionType::InnerBond>;
    static constexpr int typesParticles = 4; // Particle / Particle + Particle / Driver
    static constexpr int typesDirvers = 9; // Particle / Particle + Particle / Driver
    static constexpr int typesPP = typesParticles + typesDirvers; // 13 -- Particle / Particle + Particle / Driver
    static constexpr int typesIB = 1;   // Sticked Particles
    static constexpr int types = typesPP + typesIB;
    static constexpr int InnerBondTypeId = InteractionTypeId::InnerBond;

    // Members
    std::vector<WavePP> waves; ///< Storage for interactions categorized by type.
    std::vector<itools::interaction_buffers> buffers;     ///< Storage for analysis. Empty if there is no analysis
    WaveIB sticked_interaction; ///< Used for fragmentation

    /**
     * @brief Default constructor.
     *
     * Initializes the waves vector to hold interactions for each type.
     */
    Classifier()
    {
      ldbg << "Initialize Classifier" << std::endl;
      initialize();
    }

    /**
     * @brief Initializes the waves vector to hold interactions for each type.
     */
    void initialize()
    {
      waves.resize(typesPP)  ;
      buffers.resize(types);
    }

    /**
     * @brief Clears all stored interactions in the waves vector.
     */
    void reset_waves()
    {
      for (auto &wave : waves)
      {
        wave.clear();
      }
      sticked_interaction.clear();
    }

    /**
     * @brief Retrieves the CUDA memory-managed vector of interactions for a specific type.
     *
     * @param id Type identifier for the interaction wave.
     * @return Reference to the CUDA memory-managed vector storing interactions of the specified type.
     */
    template<InteractionType IT> 
      auto& get_data(size_t id) 
      {
        if constexpr (IT == ParticleParticle)
        {
          if (id < types)
          {
            return waves[id];
          }
        }
        if constexpr (IT == InnerBond)
        {
          if (id == InnerBondTypeId)
          {
            return sticked_interaction;
          }
        }
        color_log::error("Classifier::get_wave", "Invalid id in get_wave()");
        std::exit(EXIT_FAILURE);
      }

    InteractionWrapper<InteractionType::InnerBond> get_sticked_interaction_wrapper()
    {
      return get_data<InnerBond>(InnerBondTypeId);
    }

    template<InteractionType IT> 
      const auto& get_data(size_t id) const 
      {
        if constexpr (IT == ParticleParticle)
        {
          if (id < types)
          {
            return waves[id];
          }
        }
        if constexpr (IT == InnerBond)
        {
          if (id == InnerBondTypeId)
          {
            return sticked_interaction;
          }
        }
        color_log::error("Classifier::get_wave", "Invalid id in get_wave()");
        std::exit(EXIT_FAILURE);
      }

    size_t get_size(size_t id)
    {
      if( id < typesPP ) return waves[id].size();
      else if (id == InnerBondTypeId) return sticked_interaction.size();
      color_log::error("Classifier::get_size", "Invalid id in get_size()");
      std::exit(EXIT_FAILURE);
    }


    /**
     * @brief Retrieves the pointer and size of the data stored in the CUDA memory-managed vector for a specific type.
     *
     * @param id Type identifier for the interaction wave.
     * @return Pair containing the pointer to the interaction data and the size of the data.
     */

    template<InteractionType IT> 
      auto get_info(size_t id)
      {
        if constexpr (IT == ParticleParticle)
        {
          if (id < types)
          {
            const unsigned int data_size = waves[id].size();
            WavePP& data = waves[id];
            return  std::pair<WavePP&, size_t> {data, data_size};
          }
        }
        if constexpr (IT == InnerBond)
        {
          if (id == InnerBondTypeId)
          {
            const unsigned int data_size = sticked_interaction.size();
            return std::pair<WaveIB&, size_t>{sticked_interaction, data_size};
          }
        }

        color_log::error("Classifier::get_info", "Invalid id in get_info()");
        std::exit(EXIT_FAILURE);
      }

    template<InteractionType IT> 
      auto get_info(size_t id) const
      {
        if constexpr (IT == ParticleParticle)
        {
          if (id < types)
          {
            const unsigned int data_size = waves[id].size();
            WavePP& data = waves[id];
            return  std::pair<const WavePP&, size_t> {data, data_size};
          }
        }
        if constexpr (IT == InnerBond)
        {
          if (id == InnerBondTypeId)
          {
            const unsigned int data_size = sticked_interaction.size();
            return std::pair<const WaveIB&, size_t>{sticked_interaction, data_size};
          }
        }
        color_log::error("Classifier::get_info", "Invalid id in get_info()");
        std::exit(EXIT_FAILURE);
      }

    std::tuple<double *, Vec3d *, Vec3d *, Vec3d *> buffer_p(int id)
    {
      assert(id < types);
      auto &analysis = buffers[id];
      // fit size if needed
      size_t size = get_size(id);
      analysis.resize(size);
      double *const __restrict__ dnp = onika::cuda::vector_data(analysis.dn);
      Vec3d *const __restrict__ cpp = onika::cuda::vector_data(analysis.cp);
      Vec3d *const __restrict__ fnp = onika::cuda::vector_data(analysis.fn);
      Vec3d *const __restrict__ ftp = onika::cuda::vector_data(analysis.ft);
      return {dnp, cpp, fnp, ftp};
    }

    /**
     * @brief Returns the number of interaction types managed by the classifier.
     *
     * @return Number of interaction types.
     */
    size_t number_of_waves()
    {
      assert(types == typesPP + typesIB);
      return types;
    }

    size_t number_of_waves() const
    {
      assert(types == typesPP + typesIB);
      return types;
    }
  };
} // namespace exaDEM
