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
// #include <ostream>
#include <onika/cuda/stl_adaptors.h>
#include <onika/math/basic_types.h>
#include <onika/math/basic_types_stream.h>

#include <exaDEM/color_log.hpp>
#include <exaDEM/interaction/interaction_decoration.hpp>
#include <exaDEM/interaction/placeholder_interaction.hpp>
#include <exaDEM/interface/rupture_criterion.hpp>

#define EXADEM_INTERACTION_COMMON_FIELDS(X) \
  X(uint64_t, id_i, 0, NEW_TAG)             \
  X(uint64_t, id_j, 1, NEW_TAG)             \
  X(uint32_t, cell_i, 2, NEW_TAG)           \
  X(uint32_t, cell_j, 3, NEW_TAG)           \
  X(uint16_t, p_i, 4, NEW_TAG)              \
  X(uint16_t, p_j, 5, NEW_TAG)              \
  X(uint32_t, sub_i, 6, NEW_TAG)            \
  X(uint32_t, sub_j, 7, NEW_TAG)            \
  X(uint8_t, swap, 8, NEW_TAG)              \
  X(uint8_t, ghost, 9, NEW_TAG)

EXADEM_INTERACTION_FIELD_TAGS(EXADEM_INTERACTION_COMMON_FIELDS)

namespace exaDEM {
using exanb::Vec3d;

ONIKA_HOST_DEVICE_FUNC inline InteractionPair build_interaction_pair(
    const uint64_t* __restrict__ id_i, const uint32_t* __restrict__ cell_i, const uint16_t* __restrict__ p_i,
    const uint32_t* __restrict__ sub_i, const uint64_t* __restrict__ id_j, const uint32_t* __restrict__ cell_j,
    const uint16_t* __restrict__ p_j, const uint32_t* __restrict__ sub_j, const uint8_t* __restrict__ swap,
    const uint8_t* __restrict__ ghost, uint64_t id, uint16_t type) {
  return InteractionPair{{id_i[id], cell_i[id], p_i[id], sub_i[id]},
                         {id_j[id], cell_j[id], p_j[id], sub_j[id]},
                         type,
                         swap[id],
                         ghost[id]};
}

ONIKA_HOST_DEVICE_FUNC inline void store_interaction_pair(uint64_t* __restrict__ id_i, uint32_t* __restrict__ cell_i,
                                                          uint16_t* __restrict__ p_i, uint32_t* __restrict__ sub_i,
                                                          uint64_t* __restrict__ id_j, uint32_t* __restrict__ cell_j,
                                                          uint16_t* __restrict__ p_j, uint32_t* __restrict__ sub_j,
                                                          uint8_t* __restrict__ swap, uint8_t* __restrict__ ghost,
                                                          uint64_t id, const InteractionPair& pair) {
  id_i[id] = pair.pi_.id_;
  cell_i[id] = pair.pi_.cell_;
  p_i[id] = pair.pi_.p_;
  sub_i[id] = pair.pi_.sub_;
  id_j[id] = pair.pj_.id_;
  cell_j[id] = pair.pj_.cell_;
  p_j[id] = pair.pj_.p_;
  sub_j[id] = pair.pj_.sub_;
  swap[id] = pair.swap_;
  ghost[id] = pair.ghost_;
}

/**
 * @brief Structure representing the Structure of Arrays data structure for the interactions in a Discrete Element
 * Method (DEM) simulation.
 */
template <InteractionType IT>
struct ClassifierContainer;

// Explicit specialization: backs ParticleParticle, based on Interaction (friction, moment).
template <>
struct ClassifierContainer<InteractionType::ParticleParticle> {
  template <typename T>
  using VectorT = onika::memory::CudaMMVector<T>;

  // One VectorT<T> per field, accessed via operator[](attr::NAME) just like on Interaction.
  EXADEM_INTERACTION_VECTOR_TUPLE_TYPE(EXADEM_INTERACTION_FIELDS) fm_ = {};
  EXADEM_INTERACTION_VECTOR_FIELD_ACCESSOR(EXADEM_INTERACTION_FIELDS)  // friction, moment

  EXADEM_INTERACTION_VECTOR_TUPLE_TYPE(EXADEM_INTERACTION_COMMON_FIELDS) bk_ = {};
  EXADEM_INTERACTION_VECTOR_FIELD_ACCESSOR_BK(EXADEM_INTERACTION_COMMON_FIELDS)  // id_i, id_j, cell_i,
                                                                                 // cell_j, p_i, p_j,
                                                                                 // sub_i, sub_j, swap, ghost

  uint16_t type_; /**< Type of the interaction (e.g., contact type). */

  template <typename Func>
  void apply_on_fields(Func& func) {
    cpu_apply_on_flat_tuple(func, bk_);
    cpu_apply_on_flat_tuple(func, fm_);
  }

  struct ClearFunctor {
    template <typename T>
    inline void operator()(T& vec) {
      vec.clear();
    }
  };
  /**
   *@briefs CLears all the lists.
   */
  void clear() {
    ClearFunctor func;
    apply_on_fields(func);
  }

  struct ResizeFunctor {
    const size_t size_;
    template <typename T>
    inline void operator()(T& vec) {
      vec.resize(size_);
    }
  };

  /**
   * briefs Resize all the lists.
   */
  void resize(const size_t size) {
    ResizeFunctor func = {size};
    apply_on_fields(func);
  }

  /**
   * briefs Returns the number of interactions.
   */
  ONIKA_HOST_DEVICE_FUNC inline size_t size() const { return onika::cuda::vector_size((*this)[attr::id_i]); }
  ONIKA_HOST_DEVICE_FUNC inline size_t size() { return onika::cuda::vector_size((*this)[attr::id_i]); }

  ONIKA_HOST_DEVICE_FUNC void set(size_t idx, exaDEM::PlaceholderInteraction& interaction) {
    auto& I = interaction.as<Interaction>();
    StoreAtIndexFunctor store_func{idx};
    zip_apply_on_flat_tuple(store_func, fm_, I.fm_);

    assert(interaction.pair_.type_ == type_);
    using namespace onika::cuda;
    store_interaction_pair(vector_data((*this)[attr::id_i]), vector_data((*this)[attr::cell_i]),
                           vector_data((*this)[attr::p_i]), vector_data((*this)[attr::sub_i]),
                           vector_data((*this)[attr::id_j]), vector_data((*this)[attr::cell_j]),
                           vector_data((*this)[attr::p_j]), vector_data((*this)[attr::sub_j]),
                           vector_data((*this)[attr::swap]), vector_data((*this)[attr::ghost]), idx, interaction.pair_);
  }

  /**
   *@briefs Fills the lists.
   */
  void insert(std::vector<exaDEM::PlaceholderInteraction>& tmp, int w) {
    const size_t new_elements = tmp.size();
    const size_t old_size = this->size();
    this->resize(old_size + new_elements);

    if (w != type_) {
      color_log::mpi_error("Classifier::insert", "Wrong interaction type Id: " + std::to_string(w) +
                                                     ". It should be: " + std::to_string(type_));
    }

    for (size_t i = 0; i < new_elements; i++) {
      const size_t idx = old_size + i;
      auto& interaction = tmp[i];
      set(idx, interaction);
    }
  }

  void copy(size_t start, size_t size, std::vector<exaDEM::PlaceholderInteraction>& tmp, int w) {
    if (tmp.size() != size) {
      color_log::mpi_error("Classifier::copy", "When resizing wave: " + std::to_string(w));
    } else if (w != type_) {
      color_log::mpi_error("Classifier::copy", "Wrong interaction type Id: " + std::to_string(w) +
                                                   ". It should be: " + std::to_string(type_));
    }

    for (size_t i = 0; i < size; i++) {
      const size_t idx = start + i;
      auto& interaction = tmp[i];
      set(idx, interaction);
    }
  }

  /**
   *@briefs Return the interaction for a given list.
   */
  ONIKA_HOST_DEVICE_FUNC auto operator[](uint64_t id) {
    using namespace onika::cuda;
    InteractionPair ip = build_interaction_pair(
        vector_data((*this)[attr::id_i]), vector_data((*this)[attr::cell_i]), vector_data((*this)[attr::p_i]),
        vector_data((*this)[attr::sub_i]), vector_data((*this)[attr::id_j]), vector_data((*this)[attr::cell_j]),
        vector_data((*this)[attr::p_j]), vector_data((*this)[attr::sub_j]), vector_data((*this)[attr::swap]),
        vector_data((*this)[attr::ghost]), id, type_);

    exaDEM::Interaction res{ip};
    LoadAtIndexFunctor load_func{id};
    zip_apply_on_flat_tuple(load_func, res.fm_, fm_);
    return res;
  }

  /**
   *@briefs Updates the friction and moment of a given interaction.
   */
  ONIKA_HOST_DEVICE_FUNC void update(size_t id, PlaceholderInteraction& item) {
    Interaction& I = reinterpret_cast<Interaction&>(item);
    StoreAtIndexFunctor store_func{id};
    zip_apply_on_flat_tuple(store_func, fm_, I.fm_);
  }

  // debug
  void display() {
    onika::lout << "ClassifierContainer type is: " << type_;
    onika::lout << " and it contains " << this->size() << " interactions." << std::endl;
  }

  /**
   * @brief Non-owning, span-based, GPU-callable view over this container, built via view().
   */
  struct View {
    static constexpr InteractionType kInteractionType = InteractionType::ParticleParticle;

    template <typename T>
    using VectorT = onika::cuda::span<T>;

    EXADEM_INTERACTION_VECTOR_TUPLE_TYPE(EXADEM_INTERACTION_FIELDS) fm_ = {};
    EXADEM_INTERACTION_VECTOR_FIELD_ACCESSOR(EXADEM_INTERACTION_FIELDS)  // friction, moment

    EXADEM_INTERACTION_VECTOR_TUPLE_TYPE(EXADEM_INTERACTION_COMMON_FIELDS) bk_ = {};
    EXADEM_INTERACTION_VECTOR_FIELD_ACCESSOR_BK(EXADEM_INTERACTION_COMMON_FIELDS)  // id_i, id_j, cell_i,
                                                                                   // cell_j, p_i, p_j,
                                                                                   // sub_i, sub_j, swap, ghost

    uint16_t m_type = InteractionTypeId::Undefined;

    ONIKA_HOST_DEVICE_FUNC inline auto operator()(const uint64_t idx) const {
      InteractionPair ip = pair(idx);
      Interaction res{ip};
      LoadAtIndexSpanFunctor load_func{idx};
      zip_apply_on_flat_tuple(load_func, res.fm_, fm_);
      return res;
    }

    ONIKA_HOST_DEVICE_FUNC
    inline void set(const uint64_t idx, exaDEM::PlaceholderInteraction& item) {
      assert(m_type == item.pair_.type_);
      store_interaction_pair((*this)[attr::id_i].data(), (*this)[attr::cell_i].data(), (*this)[attr::p_i].data(),
                             (*this)[attr::sub_i].data(), (*this)[attr::id_j].data(), (*this)[attr::cell_j].data(),
                             (*this)[attr::p_j].data(), (*this)[attr::sub_j].data(), (*this)[attr::swap].data(),
                             (*this)[attr::ghost].data(), idx, item.pair_);
    }

    ONIKA_HOST_DEVICE_FUNC
    inline InteractionPair pair(const uint64_t i) const {
      return build_interaction_pair((*this)[attr::id_i].data(), (*this)[attr::cell_i].data(), (*this)[attr::p_i].data(),
                                    (*this)[attr::sub_i].data(), (*this)[attr::id_j].data(),
                                    (*this)[attr::cell_j].data(), (*this)[attr::p_j].data(),
                                    (*this)[attr::sub_j].data(), (*this)[attr::swap].data(),
                                    (*this)[attr::ghost].data(), i, m_type);
    }

    ONIKA_HOST_DEVICE_FUNC
    inline bool same(const uint64_t idx, const exaDEM::PlaceholderInteraction& item) const {
      return item.pair_ == pair(idx);
    }

    ONIKA_HOST_DEVICE_FUNC
    inline void update(const uint64_t idx, const exaDEM::Interaction& item) {
      StoreAtIndexSpanFunctor store_func{idx};
      zip_apply_on_flat_tuple(store_func, this->fm_, item.fm_);
    }

    ONIKA_HOST_DEVICE_FUNC
    inline void update(const uint64_t idx, const exaDEM::PlaceholderInteraction& item) {
      update(idx, item.as<Interaction>());
    }
  };

  /**
   * @brief Builds a View over this container, i.e. one onika::cuda::span<T> per field
   * (via make_span), so it can be passed to GPU kernels/parallel_for functors.
   */
  inline View view() {
    View v;
    ToSpanFunctor to_span_func;
    zip_apply_on_flat_tuple(to_span_func, v.fm_, fm_);
    zip_apply_on_flat_tuple(to_span_func, v.bk_, bk_);
    v.m_type = type_;
    return v;
  }
};

// Explicit specialization: backs ParticleDriver, based on Interaction (friction, moment).
// Identical in content to the ParticleParticle specialization above (both use Interaction);
// kept as a fully separate specialization rather than shared/generic code, as requested.
template <>
struct ClassifierContainer<InteractionType::ParticleDriver> {
  template <typename T>
  using VectorT = onika::memory::CudaMMVector<T>;

  // One VectorT<T> per field, accessed via operator[](attr::NAME) just like on Interaction.
  EXADEM_INTERACTION_VECTOR_TUPLE_TYPE(EXADEM_INTERACTION_FIELDS) fm_ = {};
  EXADEM_INTERACTION_VECTOR_FIELD_ACCESSOR(EXADEM_INTERACTION_FIELDS)  // friction, moment

  EXADEM_INTERACTION_VECTOR_TUPLE_TYPE(EXADEM_INTERACTION_COMMON_FIELDS) bk_ = {};
  EXADEM_INTERACTION_VECTOR_FIELD_ACCESSOR_BK(EXADEM_INTERACTION_COMMON_FIELDS)  // id_i, id_j, cell_i,
                                                                                 // cell_j, p_i, p_j,
                                                                                 // sub_i, sub_j, swap, ghost

  uint16_t type_; /**< Type of the interaction (e.g., contact type). */

  template <typename Func>
  void apply_on_fields(Func& func) {
    cpu_apply_on_flat_tuple(func, bk_);
    cpu_apply_on_flat_tuple(func, fm_);
  }

  struct ClearFunctor {
    template <typename T>
    inline void operator()(T& vec) {
      vec.clear();
    }
  };
  /**
   *@briefs CLears all the lists.
   */
  void clear() {
    ClearFunctor func;
    apply_on_fields(func);
  }

  struct ResizeFunctor {
    const size_t size_;
    template <typename T>
    inline void operator()(T& vec) {
      vec.resize(size_);
    }
  };

  /**
   * briefs Resize all the lists.
   */
  void resize(const size_t size) {
    ResizeFunctor func = {size};
    apply_on_fields(func);
  }

  /**
   * briefs Returns the number of interactions.
   */
  ONIKA_HOST_DEVICE_FUNC inline size_t size() const { return onika::cuda::vector_size((*this)[attr::id_i]); }
  ONIKA_HOST_DEVICE_FUNC inline size_t size() { return onika::cuda::vector_size((*this)[attr::id_i]); }

  ONIKA_HOST_DEVICE_FUNC void set(size_t idx, exaDEM::PlaceholderInteraction& interaction) {
    auto& I = interaction.as<Interaction>();
    StoreAtIndexFunctor store_func{idx};
    zip_apply_on_flat_tuple(store_func, fm_, I.fm_);

    assert(interaction.pair_.type_ == type_);
    using namespace onika::cuda;
    store_interaction_pair(vector_data((*this)[attr::id_i]), vector_data((*this)[attr::cell_i]),
                           vector_data((*this)[attr::p_i]), vector_data((*this)[attr::sub_i]),
                           vector_data((*this)[attr::id_j]), vector_data((*this)[attr::cell_j]),
                           vector_data((*this)[attr::p_j]), vector_data((*this)[attr::sub_j]),
                           vector_data((*this)[attr::swap]), vector_data((*this)[attr::ghost]), idx, interaction.pair_);
  }

  /**
   *@briefs Fills the lists.
   */
  void insert(std::vector<exaDEM::PlaceholderInteraction>& tmp, int w) {
    const size_t new_elements = tmp.size();
    const size_t old_size = this->size();
    this->resize(old_size + new_elements);

    if (w != type_) {
      color_log::mpi_error("Classifier::insert", "Wrong interaction type Id: " + std::to_string(w) +
                                                     ". It should be: " + std::to_string(type_));
    }

    for (size_t i = 0; i < new_elements; i++) {
      const size_t idx = old_size + i;
      auto& interaction = tmp[i];
      set(idx, interaction);
    }
  }

  void copy(size_t start, size_t size, std::vector<exaDEM::PlaceholderInteraction>& tmp, int w) {
    if (tmp.size() != size) {
      color_log::mpi_error("Classifier::copy", "When resizing wave: " + std::to_string(w));
    } else if (w != type_) {
      color_log::mpi_error("Classifier::copy", "Wrong interaction type Id: " + std::to_string(w) +
                                                   ". It should be: " + std::to_string(type_));
    }

    for (size_t i = 0; i < size; i++) {
      const size_t idx = start + i;
      auto& interaction = tmp[i];
      set(idx, interaction);
    }
  }

  /**
   *@briefs Return the interaction for a given list.
   */
  ONIKA_HOST_DEVICE_FUNC auto operator[](uint64_t id) {
    using namespace onika::cuda;
    InteractionPair ip = build_interaction_pair(
        vector_data((*this)[attr::id_i]), vector_data((*this)[attr::cell_i]), vector_data((*this)[attr::p_i]),
        vector_data((*this)[attr::sub_i]), vector_data((*this)[attr::id_j]), vector_data((*this)[attr::cell_j]),
        vector_data((*this)[attr::p_j]), vector_data((*this)[attr::sub_j]), vector_data((*this)[attr::swap]),
        vector_data((*this)[attr::ghost]), id, type_);

    exaDEM::Interaction res{ip};
    LoadAtIndexFunctor load_func{id};
    zip_apply_on_flat_tuple(load_func, res.fm_, fm_);
    return res;
  }

  /**
   *@briefs Updates the friction and moment of a given interaction.
   */
  ONIKA_HOST_DEVICE_FUNC void update(size_t id, PlaceholderInteraction& item) {
    Interaction& I = reinterpret_cast<Interaction&>(item);
    StoreAtIndexFunctor store_func{id};
    zip_apply_on_flat_tuple(store_func, fm_, I.fm_);
  }

  // debug
  void display() {
    onika::lout << "ClassifierContainer type is: " << type_;
    onika::lout << " and it contains " << this->size() << " interactions." << std::endl;
  }

  /**
   * @brief Non-owning, span-based, GPU-callable view over this container, built via view().
   */
  struct View {
    static constexpr InteractionType kInteractionType = InteractionType::ParticleDriver;

    template <typename T>
    using VectorT = onika::cuda::span<T>;

    EXADEM_INTERACTION_VECTOR_TUPLE_TYPE(EXADEM_INTERACTION_FIELDS) fm_ = {};
    EXADEM_INTERACTION_VECTOR_FIELD_ACCESSOR(EXADEM_INTERACTION_FIELDS)  // friction, moment

    EXADEM_INTERACTION_VECTOR_TUPLE_TYPE(EXADEM_INTERACTION_COMMON_FIELDS) bk_ = {};
    EXADEM_INTERACTION_VECTOR_FIELD_ACCESSOR_BK(EXADEM_INTERACTION_COMMON_FIELDS)  // id_i, id_j, cell_i,
                                                                                   // cell_j, p_i, p_j,
                                                                                   // sub_i, sub_j, swap, ghost

    uint16_t m_type = InteractionTypeId::Undefined;

    ONIKA_HOST_DEVICE_FUNC inline auto operator()(const uint64_t idx) const {
      InteractionPair ip = pair(idx);
      Interaction res{ip};
      LoadAtIndexSpanFunctor load_func{idx};
      zip_apply_on_flat_tuple(load_func, res.fm_, fm_);
      return res;
    }

    ONIKA_HOST_DEVICE_FUNC
    inline void set(const uint64_t idx, exaDEM::PlaceholderInteraction& item) {
      assert(m_type == item.pair_.type_);
      store_interaction_pair((*this)[attr::id_i].data(), (*this)[attr::cell_i].data(), (*this)[attr::p_i].data(),
                             (*this)[attr::sub_i].data(), (*this)[attr::id_j].data(), (*this)[attr::cell_j].data(),
                             (*this)[attr::p_j].data(), (*this)[attr::sub_j].data(), (*this)[attr::swap].data(),
                             (*this)[attr::ghost].data(), idx, item.pair_);
    }

    ONIKA_HOST_DEVICE_FUNC
    inline InteractionPair pair(const uint64_t i) const {
      return build_interaction_pair((*this)[attr::id_i].data(), (*this)[attr::cell_i].data(), (*this)[attr::p_i].data(),
                                    (*this)[attr::sub_i].data(), (*this)[attr::id_j].data(),
                                    (*this)[attr::cell_j].data(), (*this)[attr::p_j].data(),
                                    (*this)[attr::sub_j].data(), (*this)[attr::swap].data(),
                                    (*this)[attr::ghost].data(), i, m_type);
    }

    ONIKA_HOST_DEVICE_FUNC
    inline bool same(const uint64_t idx, const exaDEM::PlaceholderInteraction& item) const {
      return item.pair_ == pair(idx);
    }

    ONIKA_HOST_DEVICE_FUNC
    inline void update(const uint64_t idx, const exaDEM::Interaction& item) {
      StoreAtIndexSpanFunctor store_func{idx};
      zip_apply_on_flat_tuple(store_func, this->fm_, item.fm_);
    }

    ONIKA_HOST_DEVICE_FUNC
    inline void update(const uint64_t idx, const exaDEM::PlaceholderInteraction& item) {
      update(idx, item.as<Interaction>());
    }
  };

  /**
   * @brief Builds a View over this container, i.e. one onika::cuda::span<T> per field
   * (via make_span), so it can be passed to GPU kernels/parallel_for functors.
   */
  inline View view() {
    View v;
    ToSpanFunctor to_span_func;
    zip_apply_on_flat_tuple(to_span_func, v.fm_, fm_);
    zip_apply_on_flat_tuple(to_span_func, v.bk_, bk_);
    v.m_type = type_;
    return v;
  }
};

// Explicit specialization: backs InnerBond, based on InnerBondInteraction (friction, en, tds,
// et, dn0, weight, criterion, unbroken).
template <>
struct ClassifierContainer<InteractionType::InnerBond> {
  template <typename T>
  using VectorT = onika::memory::CudaMMVector<T>;

  // One VectorT<T> per field, accessed via operator[](attr::NAME) just like on
  // InnerBondInteraction.
  EXADEM_INTERACTION_VECTOR_TUPLE_TYPE(EXADEM_INNER_BOND_FIELDS) fm_ = {};
  EXADEM_INTERACTION_VECTOR_FIELD_ACCESSOR(EXADEM_INNER_BOND_FIELDS)  // friction, en, tds, et, dn0,
                                                                      // weight, criterion, unbroken

  EXADEM_INTERACTION_VECTOR_TUPLE_TYPE(EXADEM_INTERACTION_COMMON_FIELDS) bk_ = {};
  EXADEM_INTERACTION_VECTOR_FIELD_ACCESSOR_BK(EXADEM_INTERACTION_COMMON_FIELDS)  // id_i, id_j, cell_i,
                                                                                 // cell_j, p_i, p_j,
                                                                                 // sub_i, sub_j, swap, ghost

  uint16_t type_; /**< Type of the interaction (e.g., contact type). */

  template <typename Func>
  void apply_on_fields(Func& func) {
    cpu_apply_on_flat_tuple(func, bk_);
    cpu_apply_on_flat_tuple(func, fm_);
  }

  struct ClearFunctor {
    template <typename T>
    inline void operator()(T& vec) {
      vec.clear();
    }
  };
  /**
   *@briefs CLears all the lists.
   */
  void clear() {
    ClearFunctor func;
    apply_on_fields(func);
  }

  struct ResizeFunctor {
    const size_t size_;
    template <typename T>
    inline void operator()(T& vec) {
      vec.resize(size_);
    }
  };

  /**
   * briefs Resize all the lists.
   */
  void resize(const size_t size) {
    ResizeFunctor func = {size};
    apply_on_fields(func);
  }

  /**
   * briefs Returns the number of interactions.
   */
  ONIKA_HOST_DEVICE_FUNC inline size_t size() const { return onika::cuda::vector_size((*this)[attr::id_i]); }
  ONIKA_HOST_DEVICE_FUNC inline size_t size() { return onika::cuda::vector_size((*this)[attr::id_i]); }

  ONIKA_HOST_DEVICE_FUNC void set(size_t idx, exaDEM::PlaceholderInteraction& interaction) {
    auto& I = interaction.as<InnerBondInteraction>();
    StoreAtIndexFunctor store_func{idx};
    zip_apply_on_flat_tuple(store_func, fm_, I.fm_);

    assert(interaction.pair_.type_ == type_);
    using namespace onika::cuda;
    store_interaction_pair(vector_data((*this)[attr::id_i]), vector_data((*this)[attr::cell_i]),
                           vector_data((*this)[attr::p_i]), vector_data((*this)[attr::sub_i]),
                           vector_data((*this)[attr::id_j]), vector_data((*this)[attr::cell_j]),
                           vector_data((*this)[attr::p_j]), vector_data((*this)[attr::sub_j]),
                           vector_data((*this)[attr::swap]), vector_data((*this)[attr::ghost]), idx, interaction.pair_);
  }

  /**
   *@briefs Fills the lists.
   */
  void insert(std::vector<exaDEM::PlaceholderInteraction>& tmp, int w) {
    const size_t new_elements = tmp.size();
    const size_t old_size = this->size();
    this->resize(old_size + new_elements);

    if (w != type_) {
      color_log::mpi_error("Classifier::insert", "Wrong interaction type Id: " + std::to_string(w) +
                                                     ". It should be: " + std::to_string(type_));
    }

    for (size_t i = 0; i < new_elements; i++) {
      const size_t idx = old_size + i;
      auto& interaction = tmp[i];
      set(idx, interaction);
    }
  }

  void copy(size_t start, size_t size, std::vector<exaDEM::PlaceholderInteraction>& tmp, int w) {
    if (tmp.size() != size) {
      color_log::mpi_error("Classifier::copy", "When resizing wave: " + std::to_string(w));
    } else if (w != type_) {
      color_log::mpi_error("Classifier::copy", "Wrong interaction type Id: " + std::to_string(w) +
                                                   ". It should be: " + std::to_string(type_));
    }

    for (size_t i = 0; i < size; i++) {
      const size_t idx = start + i;
      auto& interaction = tmp[i];
      set(idx, interaction);
    }
  }

  /**
   *@briefs Return the interaction for a given list.
   */
  ONIKA_HOST_DEVICE_FUNC auto operator[](uint64_t id) {
    using namespace onika::cuda;
    InteractionPair ip = build_interaction_pair(
        vector_data((*this)[attr::id_i]), vector_data((*this)[attr::cell_i]), vector_data((*this)[attr::p_i]),
        vector_data((*this)[attr::sub_i]), vector_data((*this)[attr::id_j]), vector_data((*this)[attr::cell_j]),
        vector_data((*this)[attr::p_j]), vector_data((*this)[attr::sub_j]), vector_data((*this)[attr::swap]),
        vector_data((*this)[attr::ghost]), id, type_);

    exaDEM::InnerBondInteraction res{ip};
    LoadAtIndexFunctor load_func{id};
    zip_apply_on_flat_tuple(load_func, res.fm_, fm_);
    return res;
  }

  /**
   * @briefs Updates the fields of a given interaction.
   */
  ONIKA_HOST_DEVICE_FUNC void update(size_t id, PlaceholderInteraction& item) {
    InnerBondInteraction& I = reinterpret_cast<InnerBondInteraction&>(item);
    StoreAtIndexFunctor store_func{id};
    zip_apply_on_flat_tuple(store_func, fm_, I.fm_);
  }

  // debug
  void display() {
    onika::lout << "ClassifierContainer type is: " << type_;
    onika::lout << " and it contains " << this->size() << " interactions." << std::endl;
  }

  /**
   * @brief Non-owning, span-based, GPU-callable view over this container, built via view().
   */
  struct View {
    static constexpr InteractionType kInteractionType = InteractionType::InnerBond;

    template <typename T>
    using VectorT = onika::cuda::span<T>;

    EXADEM_INTERACTION_VECTOR_TUPLE_TYPE(EXADEM_INNER_BOND_FIELDS) fm_ = {};
    EXADEM_INTERACTION_VECTOR_FIELD_ACCESSOR(EXADEM_INNER_BOND_FIELDS)  // friction, en, tds, et, dn0,
                                                                        // weight, criterion, unbroken

    EXADEM_INTERACTION_VECTOR_TUPLE_TYPE(EXADEM_INTERACTION_COMMON_FIELDS) bk_ = {};
    EXADEM_INTERACTION_VECTOR_FIELD_ACCESSOR_BK(EXADEM_INTERACTION_COMMON_FIELDS)  // id_i, id_j, cell_i,
                                                                                   // cell_j, p_i, p_j,
                                                                                   // sub_i, sub_j, swap, ghost

    uint16_t m_type = InteractionTypeId::Undefined;

    ONIKA_HOST_DEVICE_FUNC inline auto operator()(const uint64_t idx) const {
      InteractionPair ip = pair(idx);
      InnerBondInteraction res{ip};
      LoadAtIndexSpanFunctor load_func{idx};
      zip_apply_on_flat_tuple(load_func, res.fm_, fm_);
      return res;
    }

    ONIKA_HOST_DEVICE_FUNC
    inline void set(const uint64_t idx, exaDEM::PlaceholderInteraction& item) {
      assert(m_type == item.pair_.type_);
      store_interaction_pair((*this)[attr::id_i].data(), (*this)[attr::cell_i].data(), (*this)[attr::p_i].data(),
                             (*this)[attr::sub_i].data(), (*this)[attr::id_j].data(), (*this)[attr::cell_j].data(),
                             (*this)[attr::p_j].data(), (*this)[attr::sub_j].data(), (*this)[attr::swap].data(),
                             (*this)[attr::ghost].data(), idx, item.pair_);
    }

    ONIKA_HOST_DEVICE_FUNC
    inline InteractionPair pair(const uint64_t i) const {
      return build_interaction_pair((*this)[attr::id_i].data(), (*this)[attr::cell_i].data(), (*this)[attr::p_i].data(),
                                    (*this)[attr::sub_i].data(), (*this)[attr::id_j].data(),
                                    (*this)[attr::cell_j].data(), (*this)[attr::p_j].data(),
                                    (*this)[attr::sub_j].data(), (*this)[attr::swap].data(),
                                    (*this)[attr::ghost].data(), i, m_type);
    }

    ONIKA_HOST_DEVICE_FUNC
    inline bool same(const uint64_t idx, const exaDEM::PlaceholderInteraction& item) const {
      return item.pair_ == pair(idx);
    }

    ONIKA_HOST_DEVICE_FUNC
    inline void update(const uint64_t idx, const exaDEM::InnerBondInteraction& item) {
      StoreAtIndexSpanFunctor store_func{idx};
      zip_apply_on_flat_tuple(store_func, this->fm_, item.fm_);
    }

    ONIKA_HOST_DEVICE_FUNC
    inline void update(const uint64_t idx, const exaDEM::PlaceholderInteraction& item) {
      update(idx, item.as<InnerBondInteraction>());
    }
  };

  /**
   * @brief Builds a View over this container, i.e. one onika::cuda::span<T> per field
   * (via make_span), so it can be passed to GPU kernels/parallel_for functors.
   */
  inline View view() {
    View v;
    ToSpanFunctor to_span_func;
    zip_apply_on_flat_tuple(to_span_func, v.fm_, fm_);
    zip_apply_on_flat_tuple(to_span_func, v.bk_, bk_);
    v.m_type = type_;
    return v;
  }
};

template <InteractionType IT, typename Func, typename... Args>
void for_all_interactions(ClassifierContainer<IT>& container, Func& func, Args&&... args) {
  size_t size = container.size();
  for (size_t i = 0; i < size; i++) {
    auto I = container[i];  // I is built, not a ref
    func(I, std::forward<Args>(args)...);
  }
}

template <InteractionType... Types>
struct ClassifierContainerDispatcher {
  template <typename ClassifierT, typename Func, typename... Args>
  static inline void dispatch(int type_, ClassifierT& iwa, Func& func, Args&&... args) {
    ((get_typed(type_) == static_cast<int>(Types)
          ? (func.template operator()<Types>(iwa.template get_data<Types>(type_), std::forward<Args>(args)...), 0)
          : 0),
     ...);
  }
};
using CDispatcher = ClassifierContainerDispatcher<InteractionType::ParticleParticle, InteractionType::ParticleDriver,
                                                  InteractionType::InnerBond>;

// bunch of functions used by the dispatcher
template <typename Apply>
struct ClassifierContainerApplyFunc {
  Apply apply_;
  template <InteractionType IT, typename... Args>
  void operator()(ClassifierContainer<IT>& container, Args&&... args) {
    apply_(container, std::forward<Args>(args)...);
  }
};
struct ClassifierContainerSizeFunc {
  size_t value_ = 0;
  template <InteractionType IT>
  void operator()(ClassifierContainer<IT>& container) {
    value_ = container.size();
  }
};

struct ClassifierContainerResizerFunc {
  template <InteractionType IT>
  void operator()(ClassifierContainer<IT>& container, size_t new_size) {
    container.resize(new_size);
  }
};

struct ClassifierContainerCopierFunc {
  template <InteractionType IT>
  void operator()(ClassifierContainer<IT>& container, std::vector<exaDEM::PlaceholderInteraction>& vec,
                  const size_t start, const size_t size, const int typeID) {
    // std::cout << get_name<IT>() << std::endl;
    container.copy(start, size, vec, typeID);
  }
};

}  // namespace exaDEM
