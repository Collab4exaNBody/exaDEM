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
#include <exaDEM/interaction/placeholder_interaction.hpp>
#include <exaDEM/interface/rupture_criterion.hpp>

namespace exaDEM {
using exanb::Vec3d;
/**
 * @brief Structure representing the Structure of Arrays data structure for the interactions in a Discrete Element
 * Method (DEM) simulation.
 */

template <InteractionType IT>
struct ClassifierContainer {
  template <typename T>
  using VectorT = onika::memory::CudaMMVector<T>;

  VectorT<double> ft_x_; /**< List of the x coordinate for the friction.  */
  VectorT<double> ft_y_; /**< List of the y coordinate for the friction.  */
  VectorT<double> ft_z_; /**< List of the z coordinate for the friction.  */

  VectorT<double> mom_x_; /**< List of the x coordinate for the moment.  */
  VectorT<double> mom_y_; /**< List of the y coordinate for the moment.  */
  VectorT<double> mom_z_; /**< List of the z coordinate for the moment.  */

  VectorT<double> en_;                 /**< List of the en.  */
  VectorT<Vec3d> tds_;                 /**< List of cumulative tangential displacement.  */
  VectorT<double> et_;                 /**< List of the et.  */
  VectorT<double> dn0_;                /**< List of the dn0.  */
  VectorT<double> weight_;             /**< List of the weight.  */
  VectorT<RuptureCriteria> criterion_; /**< List of the rupture criteria.  */

  VectorT<uint8_t> unbroken_; /**< List of the sticked interactions are unbroken.  */

  VectorT<uint64_t> id_i_; /**< List of the ids of the first particle involved in the interaction.  */
  VectorT<uint64_t> id_j_; /**< List of the ids of the second particle involved in the interaction.  */

  VectorT<uint32_t> cell_i_; /**< List of the indexes of the cell for the first particle involved in the interaction. */
  VectorT<uint32_t>
      cell_j_; /**< List of the indexes of the cell for the second particle involved in the interaction. */

  VectorT<uint16_t> p_i_; /**< List of the indexes of the particle within its cell for the first particle involved in
                            the interaction. */
  VectorT<uint16_t> p_j_; /**< List of the indexes of the particle within its cell for the second particle involved in
                            the interaction.  */

  VectorT<uint32_t> sub_i_; /**< List of the sub-particle indexes for the first particle involved in the interaction. */
  VectorT<uint32_t> sub_j_; /**< List of the sub-particle indexes for the first particle involved in the interaction. */

  uint16_t type_;          /**< Type of the interaction (e.g., contact type). */
  VectorT<uint8_t> swap_;  /**< List of .  */
  VectorT<uint8_t> ghost_; /**< List of .  */

  template <typename Func, typename Field>
  void apply_on_field(Func& func, Field& field) {
    func(field);
  }

  template <typename Func, typename Field, typename... Fields>
  void apply_on_fields(Func& func, Field& field, Fields&... fields) {
    apply_on_field(func, field);
    if constexpr (sizeof...(fields) > 0) apply_on_fields(func, fields...);
  }

  template <typename Func>
  void apply_on_fields(Func& func) {
    apply_on_fields(func, id_i_, id_j_, cell_i_, cell_j_, p_i_, p_j_, sub_i_, sub_j_, swap_, ghost_);
    if constexpr (IT == InteractionType::ParticleParticle || IT == InteractionType::ParticleDriver) {
      apply_on_fields(func, ft_x_, ft_y_, ft_z_, mom_x_, mom_y_, mom_z_);
    }
    if constexpr (IT == InteractionType::InnerBond) {
      apply_on_fields(func, ft_x_, ft_y_, ft_z_, en_, tds_, et_, dn0_, weight_, criterion_, unbroken_);
    }
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
  ONIKA_HOST_DEVICE_FUNC inline size_t size() const { return onika::cuda::vector_size(id_i_); }
  ONIKA_HOST_DEVICE_FUNC inline size_t size() { return onika::cuda::vector_size(id_i_); }

  // Some accessors
  ONIKA_HOST_DEVICE_FUNC inline uint64_t particle_id_i(size_t idx) const {
#ifdef ONIKA_CUDA_VERSION
    auto* __restrict__ ptr = onika::cuda::vector_data(id_i_);
    return ptr[idx];
#else
    return id_i_[idx];
#endif
  }

  ONIKA_HOST_DEVICE_FUNC inline uint64_t particle_id_j(size_t idx) const {
#ifdef ONIKA_CUDA_VERSION
    auto* __restrict__ ptr = onika::cuda::vector_data(id_j_);
    return ptr[idx];
#else
    return id_j_[idx];
#endif
  }

  template <typename T>
  ONIKA_HOST_DEVICE_FUNC void setter(VectorT<T>& vec, size_t idx, const T& value) {
#ifdef ONIKA_CUDA_VERSION
    auto* __restrict__ ptr = onika::cuda::vector_data(vec);
    ptr[idx] = value;
#else
    vec[idx] = value;
#endif
  }

  ONIKA_HOST_DEVICE_FUNC void set(size_t idx, exaDEM::PlaceholderInteraction& interaction) {
    if constexpr (IT == InteractionType::ParticleParticle || IT == InteractionType::ParticleDriver) {
      auto& I = interaction.as<Interaction>();
      setter(ft_x_, idx, I.friction_.x);
      setter(ft_y_, idx, I.friction_.y);
      setter(ft_z_, idx, I.friction_.z);

      setter(mom_x_, idx, I.moment_.x);
      setter(mom_y_, idx, I.moment_.y);
      setter(mom_z_, idx, I.moment_.z);
    }

    if constexpr (IT == InteractionType::InnerBond) {
      auto& I = interaction.as<InnerBondInteraction>();
      setter(ft_x_, idx, I.friction_.x);
      setter(ft_y_, idx, I.friction_.y);
      setter(ft_z_, idx, I.friction_.z);

      setter(en_, idx, I.en_);
      setter(tds_, idx, I.tds_);
      setter(et_, idx, I.et_);
      setter(dn0_, idx, I.dn0_);
      setter(weight_, idx, I.weight_);
      setter(criterion_, idx, I.criterion_);
      setter(unbroken_, idx, I.unbroken_);
    }

    auto& [pi, pj, _type, _swap, _ghost] = interaction.pair_;

    assert(_type == type_);

    setter(id_i_, idx, pi.id_);
    setter(id_j_, idx, pj.id_);

    setter(cell_i_, idx, pi.cell_);
    setter(cell_j_, idx, pj.cell_);

    setter(p_i_, idx, pi.p_);
    setter(p_j_, idx, pj.p_);

    setter(sub_i_, idx, pi.sub_);
    setter(sub_j_, idx, pj.sub_);

    setter(swap_, idx, _swap);
    setter(ghost_, idx, _ghost);
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
    InteractionPair ip = {
        // pi
        {vector_data(id_i_)[id], vector_data(cell_i_)[id], vector_data(p_i_)[id], vector_data(sub_i_)[id]},
        // pj
        {vector_data(id_j_)[id], vector_data(cell_j_)[id], vector_data(p_j_)[id], vector_data(sub_j_)[id]},
        // type_, swap_, ghost_
        type_,
        vector_data(swap_)[id],
        vector_data(ghost_)[id]};

    if constexpr (IT == InteractionType::ParticleParticle || IT == InteractionType::ParticleDriver) {
      exaDEM::Interaction res{ip,
                              {vector_data(ft_x_)[id], vector_data(ft_y_)[id], vector_data(ft_z_)[id]},
                              {vector_data(mom_x_)[id], vector_data(mom_y_)[id], vector_data(mom_z_)[id]}};
      return res;
    } else if constexpr (IT == InteractionType::InnerBond) {
      exaDEM::InnerBondInteraction res{ip,
                                       {vector_data(ft_x_)[id], vector_data(ft_y_)[id], vector_data(ft_z_)[id]},
                                       vector_data(en_)[id],
                                       vector_data(tds_)[id],
                                       vector_data(et_)[id],
                                       vector_data(dn0_)[id],
                                       vector_data(weight_)[id],
                                       vector_data(criterion_)[id],
                                       vector_data(unbroken_)[id]};
      return res;
    }
  }

  /**
   *@briefs Updates the friction and moment of a given interaction.
   */
  ONIKA_HOST_DEVICE_FUNC void update(size_t id, PlaceholderInteraction& item) {
    using namespace onika::cuda;

    if constexpr (IT == InteractionType::ParticleParticle || IT == InteractionType::ParticleDriver) {
      Interaction& I = reinterpret_cast<Interaction&>(item);
      vector_data(ft_x_)[id] = I.friction_.x;
      vector_data(ft_y_)[id] = I.friction_.y;
      vector_data(ft_z_)[id] = I.friction_.z;

      vector_data(mom_x_)[id] = I.moment_.x;
      vector_data(mom_y_)[id] = I.moment_.y;
      vector_data(mom_z_)[id] = I.moment_.z;
    }

    if constexpr (IT == InteractionType::InnerBond) {
      InnerBondInteraction& I = reinterpret_cast<InnerBondInteraction&>(item);
      vector_data(ft_x_)[id] = I.friction_.x;
      vector_data(ft_y_)[id] = I.friction_.y;
      vector_data(ft_z_)[id] = I.friction_.z;

      vector_data(en_)[id] = I.en_;
      vector_data(tds_)[id] = I.tds_;
      vector_data(et_)[id] = I.et_;
      vector_data(dn0_)[id] = I.dn0_;
      vector_data(weight_)[id] = I.weight_;
      vector_data(criterion_)[id] = I.criterion_;
      vector_data(unbroken_)[id] = I.unbroken_;
    }
  }

  inline Vec3d load_ft(size_t id) { return {ft_x_[id], ft_y_[id], ft_z_[id]}; }

  void store_ft(Vec3d&& value, size_t id) {
    ft_x_[id] = value.x;
    ft_y_[id] = value.y;
    ft_z_[id] = value.z;
  }

  // debug
  void display() {
    onika::lout << "ClassifierContainer type is: " << type_;
    onika::lout << " and it contains " << this->size() << " interactions." << std::endl;
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
