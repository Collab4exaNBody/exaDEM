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

#include <cassert>

#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/classifier/classifier_container.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/itools/buffer.hpp>
#include <exaDEM/classifier/interaction_wrapper.hpp>

namespace exaDEM {
/**
 * @brief Classifier for managing interactions categorized into different types.
 *
 * The Classifier struct manages interactions categorized into different types (up to 13 types).
 * It provides functionalities to store interactions in CUDA memory-managed vectors (`VectorT`).
 */
struct Classifier {
  using WavePP = ClassifierContainer<InteractionType::ParticleParticle>;
  using WavePD = ClassifierContainer<InteractionType::ParticleDriver>;
  using WaveIB = ClassifierContainer<InteractionType::InnerBond>;

  // Members
  std::vector<WavePP> m_particles;                     ///< Storage for interactions categorized by type.
  std::vector<WavePD> m_drivers;                       ///< Storage for interactions categorized by type.
  std::vector<WaveIB> m_innerbonds;                   ///< Used for fragmentation
  std::vector<itools::interaction_buffers> buffers;  ///< Storage for analysis. Empty if there is no analysis

  /**
   * @brief Default constructor.
   *
   * Initializes the waves vector to hold interactions for each type.
   */
  Classifier() {
    ldbg << "Initialize Classifier" << std::endl;
    initialize();
  }

  /**
   * @brief Initializes the waves vector to hold interactions for each type.
   */
  void initialize() {
    m_particles.resize(InteractionTypeId::NTypesPP);
    m_drivers.resize(InteractionTypeId::NTypesParticleDriver);
    m_innerbonds.resize(InteractionTypeId::NTypesInnerBond);
    buffers.resize(InteractionTypeId::NTypes);
    size_t typeId = 0;
    for(; typeId <= get_last_id<InteractionType::ParticleParticle>() ; typeId++) {
      int typed_id = get_typed_idx<InteractionType::ParticleParticle>(typeId);
      m_particles[typed_id].type = typeId;
    }
    for(; typeId <= get_last_id<InteractionType::ParticleDriver>() ; typeId++) {
      int typed_id = get_typed_idx<InteractionType::ParticleDriver>(typeId);
      // lout << "typeID " << typeId << " typed id " << typed_id << std::endl;
      m_drivers[typed_id].type = typeId;
    }
    for(; typeId <= get_last_id<InteractionType::InnerBond>() ; typeId++) {
      int typed_id = get_typed_idx<InteractionType::InnerBond>(typeId);
      m_innerbonds[typed_id].type = typeId;
    }
  }

  /**
   * @brief Clears all stored interactions in the waves vector.
   */
  void reset_containers() {
    for (auto& container : m_particles) {
      container.clear();
    }
    for (auto& container : m_drivers) {
      container.clear();
    }
    for (auto& container : m_innerbonds) {
      container.clear();
    }
  }

  template<InteractionType IT>
  auto& get_container() {
    if constexpr (IT == InteractionType::ParticleParticle) {
      return m_particles;
    } else if constexpr (IT == InteractionType::ParticleDriver) {
      return m_drivers;
    } else if constexpr (IT == InteractionType::InnerBond) {
      return m_innerbonds;
    }
  }

  template<InteractionType IT>
  const auto& get_container() const {
    if constexpr (IT == InteractionType::ParticleParticle) {
      return m_particles;
    } else if constexpr (IT == InteractionType::ParticleDriver) {
      return m_drivers;
    } else if constexpr (IT == InteractionType::InnerBond) {
      return m_innerbonds;
    }
  }

  /**
   * @brief Retrieves the CUDA memory-managed vector of interactions for a specific type.
   *
   * @param id Type identifier for the interaction wave.
   * @return Reference to the CUDA memory-managed vector storing interactions of the specified type.
   */
  template <InteractionType IT>
  auto& get_data(size_t id) {
    int typed_id = get_typed_idx<IT>(id);
    auto& data = get_container<IT>();
    if (typed_id >= int(data.size())) {
      color_log::error("Classifier::get_data", "Invalid id in get_wave()");
    }
    return data[typed_id];
  }

  template <InteractionType IT>
  const auto& get_data(size_t id) const {
    int typed_id = get_typed_idx<IT>(id);
    auto& data = get_container<IT>();
    if (typed_id >= int(data.size())) {
      color_log::error("Classifier::get_data", "Invalid id in get_wave()");
    }
    return data[typed_id];
  }

  InteractionWrapper<InteractionType::InnerBond> get_sticked_interaction_wrapper() {
    WaveIB& ib = get_data<InnerBond>(InteractionTypeId::FirstIdInnerBond);
    assert(ib.size() == 1);
    return InteractionWrapper<InteractionType::InnerBond>(ib);  // WARNING here
  }

  size_t get_size(size_t id) {
    ClassifierContainerSizeFunc func;
    CDispatcher::dispatch(id, *this, func);
    return func.value;
  }

  /**
   * @brief Retrieves the pointer and size of the data stored in the CUDA memory-managed vector for a specific type.
   *
   * @param id Type identifier for the interaction wave.
   * @return Pair containing the pointer to the interaction data and the size of the data.
   */

  template <InteractionType IT>
  std::pair<ClassifierContainer<IT>&, size_t> get_info(size_t id) {
    int typed_id = get_typed_idx<IT>(id);
    auto& data = get_container<IT>();
    if (typed_id >= int(data.size())) {
      color_log::error("Classifier::get_info", "Invalid Type id");
    }
    const unsigned int data_size = data[typed_id].size();
    return std::pair<ClassifierContainer<IT>&, size_t>{data[typed_id], data_size};
  }

  template <InteractionType IT>
  std::pair<const ClassifierContainer<IT>&, size_t> get_info(size_t id) const {
    int typed_id = get_typed_idx<IT>(id);
    auto& data = get_container<IT>();
    if (typed_id >= int(data.size())) {
      color_log::error("Classifier::get_info", "Invalid Type id");
    }
    const unsigned int data_size = data[typed_id].size();
    return std::pair<const ClassifierContainer<IT>&, size_t>{data[typed_id], data_size};
  }

  std::tuple<double*, Vec3d*, Vec3d*, Vec3d*> buffer_p(int id) {
    assert(id < types);
    auto& analysis = buffers[id];
    // fit size if needed
    size_t size = get_size(id);
    analysis.resize(size);
    double* const __restrict__ dnp = onika::cuda::vector_data(analysis.dn);
    Vec3d* const __restrict__ cpp = onika::cuda::vector_data(analysis.cp);
    Vec3d* const __restrict__ fnp = onika::cuda::vector_data(analysis.fn);
    Vec3d* const __restrict__ ftp = onika::cuda::vector_data(analysis.ft);
    return {dnp, cpp, fnp, ftp};
  }

  /**
   * @brief Returns the number of interaction types managed by the classifier.
   *
   * @return Number of interaction types.
   */
  size_t number_of_waves() {
    assert(types == InteractionTypeId::NTypes);
    return m_particles.size() + m_drivers.size() + m_innerbonds.size();
  }

  size_t number_of_waves() const {
    assert(types == InteractionTypeId::NTypes);
    return m_particles.size() + m_drivers.size() + m_innerbonds.size();
  }

  // debug
  void display() {
    for (auto& container : m_particles) {
      container.display();
    }
    for (auto& container : m_drivers) {
      container.display();
    }
    for (auto& container : m_innerbonds) {
      container.display();
    }
  }
};

struct ForAllInteractionFunc {
  template<InteractionType IT, typename Func, typename... Args>
  void operator()(ClassifierContainer<IT>& container,
                  Func& func, Args&&... args) {
    for_all_interactions(container, func, std::forward<Args>(args)...);
  }
};

template<typename Func, typename... Args>
void for_all_interactions(Classifier& classifier,
                          Func& func, Args&&... args) {
  ForAllInteractionFunc functor;
  for (size_t typeID = 0 ; typeID < classifier.number_of_waves() ; typeID++) {
    CDispatcher::dispatch(typeID, classifier,
                          functor, func,std::forward<Args>(args)...);
  }
}
}  // namespace exaDEM

#include <exaDEM/classifier/interaction_wrapper_accessor.hpp>
