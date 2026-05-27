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
#include <exaDEM/classifier/classifier_container.hpp>
#include <exaDEM/classifier/contact_state.hpp>
#include <exaDEM/classifier/interaction_wrapper.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/interaction.hpp>

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
  std::vector<WavePP> m_particles;            ///< Storage for interactions categorized by type.
  std::vector<WavePD> m_drivers;              ///< Storage for interactions categorized by type.
  std::vector<WaveIB> m_innerbonds;           ///< Used for fragmentation
  std::vector<ContactState> m_contact_state;  ///< Storage for contact state data.

  /**
   * @brief Default constructor.
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
    m_contact_state.resize(InteractionTypeId::NTypes);
    size_t typeId = 0;
    for (; typeId <= get_last_id<InteractionType::ParticleParticle>(); typeId++) {
      int typed_id = get_typed_idx<InteractionType::ParticleParticle>(typeId);
      m_particles[typed_id].type = typeId;
    }
    for (; typeId <= get_last_id<InteractionType::ParticleDriver>(); typeId++) {
      int typed_id = get_typed_idx<InteractionType::ParticleDriver>(typeId);
      // lout << "typeID " << typeId << " typed id " << typed_id << std::endl;
      m_drivers[typed_id].type = typeId;
    }
    for (; typeId <= get_last_id<InteractionType::InnerBond>(); typeId++) {
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

  /** @brief Retrieves the const vector of interactions for a specific type.
   * @return Const reference to the vector storing interactions of the specified type.
   */
  template <InteractionType IT>
  auto& get_container() {
    if constexpr (IT == InteractionType::ParticleParticle) {
      return m_particles;
    } else if constexpr (IT == InteractionType::ParticleDriver) {
      return m_drivers;
    } else if constexpr (IT == InteractionType::InnerBond) {
      return m_innerbonds;
    }
  }

  /** @brief Retrieves the const vector of interactions for a specific type.
   * @return Const reference to the vector storing interactions of the specified type.
   */
  template <InteractionType IT>
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
   * @brief Retrieves the vector of interactions for a specific type.
   * @param id Type identifier for the interaction wave.
   * @return Reference to the vector storing interactions of the specified type.
   */
  template <InteractionType IT>
  auto& get_data(size_t typeID) {
    int typed_id = get_typed_idx<IT>(typeID);
    auto& data = get_container<IT>();
    if (typed_id >= int(data.size())) {
      std::string msg = "Invalid id in get_data:\n";
      msg += "Type ID: " + std::to_string(typeID) + "\n";
      msg += "Typed ID: " + std::to_string(typed_id) + "\n";
      msg += "InteractionType: " + get_name<IT>();
      color_log::error("Classifier::get_data", msg);
    }
    return data[typed_id];
  }

  /** @brief Retrieves the data for a specific interaction type (const version).
   * @param typeID Type identifier for the interaction type.
   * @return Const reference to the data for the specified interaction type.
   */
  template <InteractionType IT>
  const auto& get_data(size_t typeID) const {
    int typed_id = get_typed_idx<IT>(typeID);
    auto& data = get_container<IT>();
    if (typed_id >= int(data.size())) {
      std::string msg = "Invalid id in get_data:\n";
      msg += "Type ID: " + std::to_string(typeID) + "\n";
      msg += "Typed ID: " + std::to_string(typed_id) + "\n";
      msg += "InteractionType: " + get_name<IT>();
      color_log::error("Classifier::get_data", msg);
    }
    return data[typed_id];
  }

  /** @brief Retrieves a wrapper for the sticked interaction of a specific type.
   * @return Wrapper for the sticked interaction.
   */
  InteractionWrapper<InteractionType::InnerBond> get_sticked_interaction_wrapper() {
    WaveIB& ib = get_data<InnerBond>(InteractionTypeId::FirstIdInnerBond);
    assert(m_innerbonds.size() == InteractionTypeId::NTypesInnerBond);
    return InteractionWrapper<InteractionType::InnerBond>(ib);  // WARNING here
  }

  /** @brief Retrieves the size of the container for a specific interaction type.
   * @param id Type identifier for the interaction type.
   * @return Size of the container for a specific interaction types.
   */
  size_t get_size(size_t id) {
    ClassifierContainerSizeFunc func;
    CDispatcher::dispatch(id, *this, func);
    return func.value;
  }

  /** @brief Resizes the container for a specific interaction type.
   * @param typeID Type identifier for the interaction type.
   * @param size New size for the container.
   */
  void resize(int typeID, size_t size) {
    auto resizer = [](auto& container, size_t s) -> void { container.resize(s); };
    ClassifierContainerApplyFunc func = {resizer};
    CDispatcher::dispatch(typeID, *this, func, size);
  }

  /** @brief Copies interactions of a specific type to a vector.
   * @param typeID Type identifier for the interaction type.
   * @param start Starting index for the copy operation.
   * @param size Number of interactions to copy.
   * @param vec Vector to store the copied interactions.
   */
  void copy(int typeID, size_t start, size_t size, std::vector<PlaceholderInteraction>& vec) {
    auto copier = [typeID](auto& container, size_t st, size_t si, std::vector<PlaceholderInteraction>& v) {
      container.copy(st, si, v, typeID);
    };
    ClassifierContainerApplyFunc func = {copier};
    CDispatcher::dispatch(typeID, *this, func, start, size, vec);
  }

  /**
   * @brief Retrieves the pointer and size of the data stored in the CUDA memory-managed vector for a specific type.
   * @param id Type identifier for the interaction wave.
   * @return Pair containing the pointer to the interaction data and the size of the data.
   */
  template <InteractionType IT>
  std::pair<ClassifierContainer<IT>&, size_t> get_info(size_t typeID) {
    int typed_id = get_typed_idx<IT>(typeID);
    auto& data = get_container<IT>();
    if (typed_id >= int(data.size())) {
      std::string msg = "Invalid Type id: " + std::to_string(typeID);
      color_log::error("Classifier::get_info", msg);
    }
    const unsigned int data_size = data[typed_id].size();
    return std::pair<ClassifierContainer<IT>&, size_t>{data[typed_id], data_size};
  }

  /** @brief Retrieves the pointer and size of the data stored vector for a specific type (const version).
   * @param id Type identifier for the interaction type.
   * @return Pair containing the pointer to the interaction data and the size of the data.
   */
  template <InteractionType IT>
  std::pair<const ClassifierContainer<IT>&, size_t> get_info(size_t typeID) const {
    int typed_id = get_typed_idx<IT>(typeID);
    auto& data = get_container<IT>();
    if (typed_id >= int(data.size())) {
      std::string msg = "Invalid Type id: " + std::to_string(typeID);
      color_log::error("Classifier::get_info", msg);
    }
    const unsigned int data_size = data[typed_id].size();
    return std::pair<const ClassifierContainer<IT>&, size_t>{data[typed_id], data_size};
  }

  /** @brief Retrieves the contact state buffers for a specific interaction type.
   * @param id Type identifier for the interaction wave.
   * @return Tuple containing pointers to the contact state buffers: overlap (dn), contact points, normal forces, and
   * tangential forces.
   */
  std::tuple<double*, Vec3d*, Vec3d*, Vec3d*> contact_state(int id) {
    assert(id < InteractionTypeId::NTypes);
    auto& state = m_contact_state[id];
    // fit size if needed
    size_t size = get_size(id);
    state.resize(size);
    double* const __restrict__ dnp = onika::cuda::vector_data(state.dn);
    Vec3d* const __restrict__ cpp = onika::cuda::vector_data(state.cp);
    Vec3d* const __restrict__ fnp = onika::cuda::vector_data(state.fn);
    Vec3d* const __restrict__ ftp = onika::cuda::vector_data(state.ft);
    return {dnp, cpp, fnp, ftp};
  }

  /**
   * @brief Returns the number of interaction types managed by the classifier.
   *
   * @return Number of interaction types.
   */
  size_t number_of_waves() { return m_particles.size() + m_drivers.size() + m_innerbonds.size(); }
  size_t number_of_waves() const { return m_particles.size() + m_drivers.size() + m_innerbonds.size(); }

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

/** @brief Function object for applying a function to all interactions of a specific interaction type. */
struct ForAllInteractionFunc {
  template <InteractionType IT, typename Func, typename... Args>
  void operator()(ClassifierContainer<IT>& container, Func& func, Args&&... args) {
    for_all_interactions(container, func, std::forward<Args>(args)...);
  }
};

/** @brief Applies a function to all interactions of a specific interaction type.
 * @param classifier The classifier containing the interactions.
 * @param func The function to apply to each interaction.
 * @param args Additional arguments to pass to the function.
 */
template <typename Func, typename... Args>
void for_all_interactions(Classifier& classifier, Func& func, Args&&... args) {
  ForAllInteractionFunc functor;
  for (size_t typeID = 0; typeID < classifier.number_of_waves(); typeID++) {
    CDispatcher::dispatch(typeID, classifier, functor, func, std::forward<Args>(args)...);
  }
}
}  // namespace exaDEM

#include <exaDEM/classifier/interaction_wrapper_accessor.hpp>
