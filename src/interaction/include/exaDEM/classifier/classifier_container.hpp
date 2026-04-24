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
#include <onika/math/basic_types.h>
#include <onika/math/basic_types_stream.h>
#include <onika/cuda/stl_adaptors.h>
#include <exaDEM/color_log.hpp>
#include <exaDEM/interaction/placeholder_interaction.hpp>

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

  VectorT<double> ft_x; /**< List of the x coordinate for the friction.  */
  VectorT<double> ft_y; /**< List of the y coordinate for the friction.  */
  VectorT<double> ft_z; /**< List of the z coordinate for the friction.  */

  VectorT<double> mom_x; /**< List of the x coordinate for the moment.  */
  VectorT<double> mom_y; /**< List of the y coordinate for the moment.  */
  VectorT<double> mom_z; /**< List of the z coordinate for the moment.  */

  VectorT<double> en;        /**< List of the en.  */
  VectorT<Vec3d> tds;        /**< List of cumulative tangential displacement.  */
  VectorT<double> et;        /**< List of the et.  */
  VectorT<double> dn0;       /**< List of the dn0.  */
  VectorT<double> weight;       /**< List of the weight.  */
  VectorT<double> criterion; /**< List of the criterion.  */
  VectorT<uint8_t> unbroken; /**< List of the sticked interactions are unbroken.  */

  VectorT<uint64_t> id_i; /**< List of the ids of the first particle involved in the interaction.  */
  VectorT<uint64_t> id_j; /**< List of the ids of the second particle involved in the interaction.  */

  VectorT<uint32_t> cell_i; /**< List of the indexes of the cell for the first particle involved in the interaction.  */
  VectorT<uint32_t> cell_j; /**< List of the indexes of the cell for the second particle involved in the interaction. */

  VectorT<uint16_t> p_i; /**< List of the indexes of the particle within its cell for the first particle involved in the
                            interaction. */
  VectorT<uint16_t> p_j; /**< List of the indexes of the particle within its cell for the second particle involved in
                            the interaction.  */

  VectorT<uint32_t> sub_i; /**< List of the sub-particle indexes for the first particle involved in the interaction.  */
  VectorT<uint32_t> sub_j; /**< List of the sub-particle indexes for the first particle involved in the interaction.  */

  uint16_t type;          /**< Type of the interaction (e.g., contact type). */
  VectorT<uint8_t> swap;  /**< List of .  */
  VectorT<uint8_t> ghost; /**< List of .  */

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
    apply_on_fields(func, id_i, id_j, cell_i, cell_j, p_i, p_j, sub_i, sub_j, swap, ghost);
    if constexpr (IT == InteractionType::ParticleParticle || IT == InteractionType::ParticleDriver) {
      apply_on_fields(func, ft_x, ft_y, ft_z, mom_x, mom_y, mom_z);
    }
    if constexpr (IT == InteractionType::InnerBond) {
      apply_on_fields(func, ft_x, ft_y, ft_z, en, tds, et, dn0, weight, criterion, unbroken);
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
    const size_t size;
    template <typename T>
    inline void operator()(T& vec) {
      vec.resize(size);
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
  ONIKA_HOST_DEVICE_FUNC inline size_t size() const {
    return onika::cuda::vector_size(id_i);
  }
  ONIKA_HOST_DEVICE_FUNC inline size_t size() {
    return onika::cuda::vector_size(id_i);
  }

  // Some accessors
  ONIKA_HOST_DEVICE_FUNC inline uint64_t particle_id_i(size_t idx) const {
#ifdef ONIKA_CUDA_VERSION
    auto* __restrict__ ptr = onika::cuda::vector_data(id_i);
    return ptr[idx];
#else
    return id_i[idx];
#endif
  }

  ONIKA_HOST_DEVICE_FUNC inline uint64_t particle_id_j(size_t idx) const {
#ifdef ONIKA_CUDA_VERSION
    auto* __restrict__ ptr = onika::cuda::vector_data(id_j);
    return ptr[idx];
#else
    return id_j[idx];
#endif
  }

  template<typename T>
  ONIKA_HOST_DEVICE_FUNC
  void setter(VectorT<T>& vec, size_t idx, const T& value) {
#ifdef ONIKA_CUDA_VERSION
    auto* __restrict__ ptr = onika::cuda::vector_data(vec);
    ptr[idx] = value;
#else
    vec[idx] = value;
#endif
  }

  ONIKA_HOST_DEVICE_FUNC void set(
      size_t idx,
      exaDEM::PlaceholderInteraction& interaction) {
    if constexpr (IT == InteractionType::ParticleParticle || IT == InteractionType::ParticleDriver) {
      auto& I = interaction.as<Interaction>();
      setter(ft_x, idx, I.friction.x);
      setter(ft_y, idx, I.friction.y);
      setter(ft_z, idx, I.friction.z);

      setter(mom_x, idx, I.moment.x);
      setter(mom_y, idx, I.moment.y);
      setter(mom_z, idx, I.moment.z);
    }

    if constexpr (IT == InteractionType::InnerBond) {
      auto& I = interaction.as<InnerBondInteraction>();
      setter(ft_x, idx, I.friction.x);
      setter(ft_y, idx, I.friction.y);
      setter(ft_z, idx, I.friction.z);

      setter(en, idx, I.en);
      setter(tds, idx, I.tds);
      setter(et, idx, I.et);
      setter(dn0, idx, I.dn0);
      setter(weight, idx, I.weight);
      setter(criterion, idx, I.criterion);
      setter(unbroken, idx, I.unbroken);
    }

    auto& [pi, pj, _type, _swap, _ghost] = interaction.pair;

    assert(_type == type);

    setter(id_i, idx, pi.id);
    setter(id_j, idx, pj.id);

    setter(cell_i, idx, pi.cell);
    setter(cell_j, idx, pj.cell);

    setter(p_i, idx, pi.p);
    setter(p_j, idx, pj.p);

    setter(sub_i, idx, pi.sub);
    setter(sub_j, idx, pj.sub);

    setter(swap, idx, _swap);
    setter(ghost, idx, _ghost);
  }

  /**
   *@briefs Fills the lists.
   */
  void insert(std::vector<exaDEM::PlaceholderInteraction>& tmp, int w) {
    const size_t new_elements = tmp.size();
    const size_t old_size = this->size();
    this->resize(old_size + new_elements);

    if (w != type) {
      color_log::mpi_error("Classifier::insert",
                           "Wrong interaction type Id: " + std::to_string(w) +
                           ". It should be: " + std::to_string(type));
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
    } else if (w != type) {
      color_log::mpi_error("Classifier::copy",
			 "Wrong interaction type Id: " + std::to_string(w) +
                           ". It should be: " + std::to_string(type));
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
      {vector_data(id_i)[id], vector_data(cell_i)[id], vector_data(p_i)[id], vector_data(sub_i)[id]},
      // pj
      {vector_data(id_j)[id], vector_data(cell_j)[id], vector_data(p_j)[id], vector_data(sub_j)[id]},
      // type, swap, ghost
      type,
      vector_data(swap)[id],
      vector_data(ghost)[id]};

    if constexpr (IT == InteractionType::ParticleParticle || IT == InteractionType::ParticleDriver) {
      exaDEM::Interaction res{ip,
        {vector_data(ft_x)[id], vector_data(ft_y)[id], vector_data(ft_z)[id]},
        {vector_data(mom_x)[id], vector_data(mom_y)[id], vector_data(mom_z)[id]}};
      return res;
    } else if constexpr (IT == InteractionType::InnerBond) {
      exaDEM::InnerBondInteraction res{ip,
        {vector_data(ft_x)[id], vector_data(ft_y)[id], vector_data(ft_z)[id]},
        vector_data(en)[id],
        vector_data(tds)[id],
        vector_data(et)[id],
        vector_data(dn0)[id],
        vector_data(weight)[id],
        vector_data(criterion)[id],
        vector_data(unbroken)[id]};
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
      vector_data(ft_x)[id] = I.friction.x;
      vector_data(ft_y)[id] = I.friction.y;
      vector_data(ft_z)[id] = I.friction.z;

      vector_data(mom_x)[id] = I.moment.x;
      vector_data(mom_y)[id] = I.moment.y;
      vector_data(mom_z)[id] = I.moment.z;
    }

    if constexpr (IT == InteractionType::InnerBond) {
      InnerBondInteraction& I = reinterpret_cast<InnerBondInteraction&>(item);
      vector_data(ft_x)[id] = I.friction.x;
      vector_data(ft_y)[id] = I.friction.y;
      vector_data(ft_z)[id] = I.friction.z;

      vector_data(en)[id] = I.en;
      vector_data(tds)[id] = I.tds;
      vector_data(et)[id] = I.et;
      vector_data(dn0)[id] = I.dn0;
      vector_data(weight)[id] = I.weight;
      vector_data(criterion)[id] = I.criterion;
      vector_data(unbroken)[id] = I.unbroken;
    }
  }

  inline Vec3d load_ft(size_t id) {
    return {ft_x[id], ft_y[id], ft_z[id]};
  }

  void store_ft(Vec3d&& value, size_t id) {
    ft_x[id] = value.x;
    ft_y[id] = value.y;
    ft_z[id] = value.z;
  }

  // debug
  void display() {
    onika::lout << "ClassifierContainer type is: " << type;
    onika::lout << " and it contains " << this->size() << " interactions." << std::endl;
  }
};


template<InteractionType IT, typename Func, typename... Args>
void for_all_interactions(ClassifierContainer<IT>& container, Func& func, Args&&... args) {
  size_t size = container.size();
  for(size_t i = 0 ; i<size ; i++) {
    auto I = container[i];  // I is built, not a ref
    func(I, std::forward<Args>(args)...);
  }
}

template<InteractionType... Types>
struct ClassifierContainerDispatcher
{
  template<typename ClassifierT, typename Func, typename... Args>
    static inline void dispatch(
	int type,
	ClassifierT& iwa,
	Func& func,
	Args&&... args)
    {
      ((get_typed(type) == static_cast<int>(Types)
	? (func.template operator()<Types>(
	    iwa.template get_data<Types>(type),
	    std::forward<Args>(args)...), 0)
	: 0), ...);
    }
};
using CDispatcher = ClassifierContainerDispatcher<InteractionType::ParticleParticle, InteractionType::ParticleDriver, InteractionType::InnerBond>;

// bunch of functions used by the dispatcher
template<typename Apply>
struct ClassifierContainerApplyFunc {
  Apply apply;
  template<InteractionType IT, typename... Args>
  void operator()(ClassifierContainer<IT>& container,
                  Args&&... args) {
    apply(container, std::forward<Args>(args)...);
  } 
};
struct ClassifierContainerSizeFunc {
  size_t value = 0;
  template<InteractionType IT>
    void operator()(ClassifierContainer<IT>& container) {
      value = container.size();
    }
};

struct ClassifierContainerResizerFunc {
  template<InteractionType IT>
  void operator()(ClassifierContainer<IT>& container,
                  size_t new_size) {
    container.resize(new_size);
  } 
};

struct ClassifierContainerCopierFunc {
  template<InteractionType IT>
    void operator()(ClassifierContainer<IT>& container,
        std::vector<exaDEM::PlaceholderInteraction>& vec,
	const size_t start,
	const size_t size,
        const int typeID) {
      // std::cout << get_name<IT>() << std::endl;
      container.copy(start, size, vec, typeID);
    } 
};

}  // namespace exaDEM
