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

#include <onika/cuda/stl_adaptors.h>

#include <exaDEM/classifier/classifier_container.hpp>
#include <exaDEM/interaction/interaction.hpp>
#include <type_traits>

namespace exaDEM {

struct InterationPairWrapper {
  template <typename T>
  using VectorT = onika::cuda::span<T>;

  EXADEM_INTERACTION_VECTOR_TUPLE_TYPE(EXADEM_INTERACTION_COMMON_FIELDS) bk_ = {};
  EXADEM_INTERACTION_VECTOR_FIELD_ACCESSOR_BK(EXADEM_INTERACTION_COMMON_FIELDS)  // id_i, id_j, cell_i,
                                                                                 // cell_j, p_i, p_j,
                                                                                 // sub_i, sub_j, swap, ghost

  uint16_t m_type;

  template <typename InteractionContainerT>
  void wrap(InteractionContainerT& container) {
    ToSpanFunctor to_span_func;
    zip_apply_on_flat_tuple(to_span_func, bk_, container.bk_);
    m_type = container.type_;
  }

  ONIKA_HOST_DEVICE_FUNC
  inline InteractionPair operator()(size_t i) {
    return InteractionPair{ParticleSubLocation{(*this)[attr::id_i][i], (*this)[attr::cell_i][i], (*this)[attr::p_i][i],
                                               (*this)[attr::sub_i][i]},
                           ParticleSubLocation{(*this)[attr::id_j][i], (*this)[attr::cell_j][i], (*this)[attr::p_j][i],
                                               (*this)[attr::sub_j][i]},
                           m_type, (*this)[attr::swap][i], (*this)[attr::ghost][i]};
  }
};

template <InteractionType IT>
struct InteractionWrapper;

template <>
struct InteractionWrapper<InteractionType::ParticleParticle> {
  template <typename T>
  using VectorT = onika::cuda::span<T>;

  // Members are declared here, to see attributes, please go to interaction.hpp
  EXADEM_INTERACTION_VECTOR_TUPLE_TYPE(EXADEM_INTERACTION_FIELDS) fm_ = {};
  EXADEM_INTERACTION_VECTOR_FIELD_ACCESSOR(EXADEM_INTERACTION_FIELDS)  // friction, moment

  EXADEM_INTERACTION_VECTOR_TUPLE_TYPE(EXADEM_INTERACTION_COMMON_FIELDS) bk_ = {};
  EXADEM_INTERACTION_VECTOR_FIELD_ACCESSOR_BK(EXADEM_INTERACTION_COMMON_FIELDS)  // id_i, id_j, cell_i,
                                                                                 // cell_j, p_i, p_j,
                                                                                 // sub_i, sub_j, swap, ghost

  uint16_t m_type = InteractionTypeId::Undefined;

  InteractionWrapper() {}

  InteractionWrapper(ClassifierContainer<InteractionType::ParticleParticle>& data) {
    ToSpanFunctor to_span_func;
    zip_apply_on_flat_tuple(to_span_func, fm_, data.fm_);
    zip_apply_on_flat_tuple(to_span_func, bk_, data.bk_);
    m_type = data.type_;
  }

  ONIKA_HOST_DEVICE_FUNC inline auto operator()(const uint64_t idx) const {
    InteractionPair ip = {
        {(*this)[attr::id_i][idx], (*this)[attr::cell_i][idx], (*this)[attr::p_i][idx], (*this)[attr::sub_i][idx]},
        {(*this)[attr::id_j][idx], (*this)[attr::cell_j][idx], (*this)[attr::p_j][idx], (*this)[attr::sub_j][idx]},
        m_type,
        (*this)[attr::swap][idx],
        (*this)[attr::ghost][idx]};
    Interaction res{ip};
    LoadAtIndexSpanFunctor load_func{idx};
    zip_apply_on_flat_tuple(load_func, res.fm_, fm_);
    return res;
  }

  ONIKA_HOST_DEVICE_FUNC
  inline void set(const uint64_t idx, exaDEM::PlaceholderInteraction& item) {
    assert(m_type == item.pair_.type_);
    (*this)[attr::id_i][idx] = item.pair_.pi_.id_;
    (*this)[attr::id_j][idx] = item.pair_.pj_.id_;
    (*this)[attr::cell_i][idx] = item.pair_.pi_.cell_;
    (*this)[attr::cell_j][idx] = item.pair_.pj_.cell_;
    (*this)[attr::p_i][idx] = item.pair_.pi_.p_;
    (*this)[attr::p_j][idx] = item.pair_.pj_.p_;
    (*this)[attr::sub_i][idx] = item.pair_.pi_.sub_;
    (*this)[attr::sub_j][idx] = item.pair_.pj_.sub_;
    (*this)[attr::swap][idx] = item.pair_.swap_;
    (*this)[attr::ghost][idx] = item.pair_.ghost_;
  }

  ONIKA_HOST_DEVICE_FUNC
  inline InteractionPair pair(const uint64_t i) const {
    return InteractionPair{ParticleSubLocation{(*this)[attr::id_i][i], (*this)[attr::cell_i][i], (*this)[attr::p_i][i],
                                               (*this)[attr::sub_i][i]},
                           ParticleSubLocation{(*this)[attr::id_j][i], (*this)[attr::cell_j][i], (*this)[attr::p_j][i],
                                               (*this)[attr::sub_j][i]},
                           m_type, (*this)[attr::swap][i], (*this)[attr::ghost][i]};
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

template <>
struct InteractionWrapper<InteractionType::ParticleDriver> {
  template <typename T>
  using VectorT = onika::cuda::span<T>;

  // Members are declared here, to see attributes, please go to interaction.hpp
  EXADEM_INTERACTION_VECTOR_TUPLE_TYPE(EXADEM_INTERACTION_FIELDS) fm_ = {};
  EXADEM_INTERACTION_VECTOR_FIELD_ACCESSOR(EXADEM_INTERACTION_FIELDS)  // friction, moment

  EXADEM_INTERACTION_VECTOR_TUPLE_TYPE(EXADEM_INTERACTION_COMMON_FIELDS) bk_ = {};
  EXADEM_INTERACTION_VECTOR_FIELD_ACCESSOR_BK(EXADEM_INTERACTION_COMMON_FIELDS)  // pair interaction attributes

  uint16_t m_type = InteractionTypeId::Undefined;

  InteractionWrapper() {}

  InteractionWrapper(ClassifierContainer<InteractionType::ParticleDriver>& data) {
    ToSpanFunctor to_span_func;
    zip_apply_on_flat_tuple(to_span_func, fm_, data.fm_);
    zip_apply_on_flat_tuple(to_span_func, bk_, data.bk_);
    m_type = data.type_;
  }

  ONIKA_HOST_DEVICE_FUNC inline auto operator()(const uint64_t idx) const {
    InteractionPair ip = {
        {(*this)[attr::id_i][idx], (*this)[attr::cell_i][idx], (*this)[attr::p_i][idx], (*this)[attr::sub_i][idx]},
        {(*this)[attr::id_j][idx], (*this)[attr::cell_j][idx], (*this)[attr::p_j][idx], (*this)[attr::sub_j][idx]},
        m_type,
        (*this)[attr::swap][idx],
        (*this)[attr::ghost][idx]};
    Interaction res{ip};
    LoadAtIndexSpanFunctor load_func{idx};
    zip_apply_on_flat_tuple(load_func, res.fm_, fm_);
    return res;
  }

  ONIKA_HOST_DEVICE_FUNC
  inline void set(const uint64_t idx, exaDEM::PlaceholderInteraction& item) {
    assert(m_type == item.pair_.type_);
    (*this)[attr::id_i][idx] = item.pair_.pi_.id_;
    (*this)[attr::id_j][idx] = item.pair_.pj_.id_;
    (*this)[attr::cell_i][idx] = item.pair_.pi_.cell_;
    (*this)[attr::cell_j][idx] = item.pair_.pj_.cell_;
    (*this)[attr::p_i][idx] = item.pair_.pi_.p_;
    (*this)[attr::p_j][idx] = item.pair_.pj_.p_;
    (*this)[attr::sub_i][idx] = item.pair_.pi_.sub_;
    (*this)[attr::sub_j][idx] = item.pair_.pj_.sub_;
    (*this)[attr::swap][idx] = item.pair_.swap_;
    (*this)[attr::ghost][idx] = item.pair_.ghost_;
  }

  ONIKA_HOST_DEVICE_FUNC
  inline InteractionPair pair(const uint64_t i) const {
    return InteractionPair{ParticleSubLocation{(*this)[attr::id_i][i], (*this)[attr::cell_i][i], (*this)[attr::p_i][i],
                                               (*this)[attr::sub_i][i]},
                           ParticleSubLocation{(*this)[attr::id_j][i], (*this)[attr::cell_j][i], (*this)[attr::p_j][i],
                                               (*this)[attr::sub_j][i]},
                           m_type, (*this)[attr::swap][i], (*this)[attr::ghost][i]};
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

template <>
struct InteractionWrapper<InteractionType::InnerBond> {
  template <typename T>
  using VectorT = onika::cuda::span<T>;

  // Members are declared here, to see attributes, please go to innerbond_interaction.hpp
  EXADEM_INTERACTION_VECTOR_TUPLE_TYPE(EXADEM_INNER_BOND_FIELDS) fm_ = {};
  EXADEM_INTERACTION_VECTOR_FIELD_ACCESSOR(EXADEM_INNER_BOND_FIELDS)  // friction, en, tds, et, dn0, weight,
                                                                      // criterion, unbroken

  EXADEM_INTERACTION_VECTOR_TUPLE_TYPE(EXADEM_INTERACTION_COMMON_FIELDS) bk_ = {};
  EXADEM_INTERACTION_VECTOR_FIELD_ACCESSOR_BK(EXADEM_INTERACTION_COMMON_FIELDS)  // pairt interaction attributes

  uint16_t m_type = InteractionTypeId::Undefined;

  InteractionWrapper() {}

  InteractionWrapper(ClassifierContainer<InteractionType::InnerBond>& data) {
    ToSpanFunctor to_span_func;
    zip_apply_on_flat_tuple(to_span_func, fm_, data.fm_);
    zip_apply_on_flat_tuple(to_span_func, bk_, data.bk_);
    m_type = data.type_;
  }

  ONIKA_HOST_DEVICE_FUNC inline auto operator()(const uint64_t idx) const {
    InteractionPair ip = {
        {(*this)[attr::id_i][idx], (*this)[attr::cell_i][idx], (*this)[attr::p_i][idx], (*this)[attr::sub_i][idx]},
        {(*this)[attr::id_j][idx], (*this)[attr::cell_j][idx], (*this)[attr::p_j][idx], (*this)[attr::sub_j][idx]},
        m_type,
        (*this)[attr::swap][idx],
        (*this)[attr::ghost][idx]};
    InnerBondInteraction res{ip};
    LoadAtIndexSpanFunctor load_func{idx};
    zip_apply_on_flat_tuple(load_func, res.fm_, fm_);
    return res;
  }

  ONIKA_HOST_DEVICE_FUNC
  inline void set(const uint64_t idx, exaDEM::PlaceholderInteraction& item) {
    assert(m_type == item.pair_.type_);
    (*this)[attr::id_i][idx] = item.pair_.pi_.id_;
    (*this)[attr::id_j][idx] = item.pair_.pj_.id_;
    (*this)[attr::cell_i][idx] = item.pair_.pi_.cell_;
    (*this)[attr::cell_j][idx] = item.pair_.pj_.cell_;
    (*this)[attr::p_i][idx] = item.pair_.pi_.p_;
    (*this)[attr::p_j][idx] = item.pair_.pj_.p_;
    (*this)[attr::sub_i][idx] = item.pair_.pi_.sub_;
    (*this)[attr::sub_j][idx] = item.pair_.pj_.sub_;
    (*this)[attr::swap][idx] = item.pair_.swap_;
    (*this)[attr::ghost][idx] = item.pair_.ghost_;
  }

  ONIKA_HOST_DEVICE_FUNC
  inline InteractionPair pair(const uint64_t i) const {
    return InteractionPair{ParticleSubLocation{(*this)[attr::id_i][i], (*this)[attr::cell_i][i], (*this)[attr::p_i][i],
                                               (*this)[attr::sub_i][i]},
                           ParticleSubLocation{(*this)[attr::id_j][i], (*this)[attr::cell_j][i], (*this)[attr::p_j][i],
                                               (*this)[attr::sub_j][i]},
                           m_type, (*this)[attr::swap][i], (*this)[attr::ghost][i]};
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

InteractionWrapper(ClassifierContainer<InteractionType::ParticleParticle>&)
    -> InteractionWrapper<InteractionType::ParticleParticle>;
InteractionWrapper(ClassifierContainer<InteractionType::ParticleDriver>&)
    -> InteractionWrapper<InteractionType::ParticleDriver>;
InteractionWrapper(ClassifierContainer<InteractionType::InnerBond>&) -> InteractionWrapper<InteractionType::InnerBond>;
}  // namespace exaDEM
