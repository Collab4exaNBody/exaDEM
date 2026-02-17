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

namespace exaDEM {
struct InterationPairWrapper {
  // particle id
  uint64_t* id_i;
  uint64_t* id_j;
  // cell id
  uint32_t* cell_i;
  uint32_t* cell_j;
  // position into the cell
  uint16_t* p_i;
  uint16_t* p_j;
  // sub id
  uint32_t* sub_i;
  uint32_t* sub_j;
  uint16_t m_type;
  uint8_t* m_swap;
  uint8_t* m_ghost;

  template <typename InteractionContainerT>
  void wrap(InteractionContainerT& container) {
    using namespace onika::cuda;
    id_i = container.id_i.data();
    id_j = container.id_j.data();

    cell_i = container.cell_i.data();
    cell_j = container.cell_j.data();

    p_i = container.p_i.data();
    p_j = container.p_j.data();

    sub_i = container.sub_i.data();
    sub_j = container.sub_j.data();

    m_type = container.type;
    m_swap = container.swap.data();
    m_ghost = container.ghost.data();
  }

  ONIKA_HOST_DEVICE_FUNC
  inline InteractionPair operator()(size_t i) {
    return InteractionPair{ParticleSubLocation{id_i[i], cell_i[i], p_i[i], sub_i[i]},
                           ParticleSubLocation{id_j[i], cell_j[i], p_j[i], sub_j[i]}, m_type, m_swap[i], m_ghost[i]};
  }
};

template <InteractionType IT>
struct InteractionWrapper {
  // forces
  double* ft_x;
  double* ft_y;
  double* ft_z;
  // moment
  double* mom_x;
  double* mom_y;
  double* mom_z;
  // Fragmentation
  double* en;
  Vec3d* tds;
  double* et;
  double* dn0;
  double* weight;
  double* criterion;
  uint8_t* unbroken;

  // particle id
  uint64_t* id_i;
  uint64_t* id_j;
  // cell id
  uint32_t* cell_i;
  uint32_t* cell_j;
  // position into the cell
  uint16_t* p_i;
  uint16_t* p_j;
  // sub id
  uint32_t* sub_i;
  uint32_t* sub_j;
  uint16_t m_type;
  uint8_t* m_swap;
  uint8_t* m_ghost;

  InteractionWrapper(ClassifierContainer<IT>& data) {
    using namespace onika::cuda;
    if constexpr (IT == InteractionType::ParticleParticle || IT == InteractionType::ParticleDriver) {
      ft_x = vector_data(data.ft_x);
      ft_y = vector_data(data.ft_y);
      ft_z = vector_data(data.ft_z);

      mom_x = vector_data(data.mom_x);
      mom_y = vector_data(data.mom_y);
      mom_z = vector_data(data.mom_z);
    }

    if constexpr (IT == InteractionType::InnerBond) {
      ft_x = vector_data(data.ft_x);
      ft_y = vector_data(data.ft_y);
      ft_z = vector_data(data.ft_z);

      en = vector_data(data.en);
      tds = vector_data(data.tds);
      et = vector_data(data.et);
      dn0 = vector_data(data.dn0);
      weight = vector_data(data.weight);
      criterion = vector_data(data.criterion);
      unbroken = vector_data(data.unbroken);
    }

    id_i = vector_data(data.id_i);
    id_j = vector_data(data.id_j);

    cell_i = vector_data(data.cell_i);
    cell_j = vector_data(data.cell_j);

    p_i = vector_data(data.p_i);
    p_j = vector_data(data.p_j);

    sub_i = vector_data(data.sub_i);
    sub_j = vector_data(data.sub_j);

    m_type = data.type;
    m_swap = vector_data(data.swap);
    m_ghost = vector_data(data.ghost);
  }

  ONIKA_HOST_DEVICE_FUNC inline auto operator()(const uint64_t idx) const {
    InteractionPair ip = {{id_i[idx], cell_i[idx], p_i[idx], sub_i[idx]},
                          {id_j[idx], cell_j[idx], p_j[idx], sub_j[idx]},
                          m_type,
                          m_swap[idx],
                          m_ghost[idx]};

    if constexpr (IT == ParticleParticle) {
      return Interaction{ip, {ft_x[idx], ft_y[idx], ft_z[idx]}, {mom_x[idx], mom_y[idx], mom_z[idx]}};
    } else if constexpr (IT == InnerBond) {
      return InnerBondInteraction{
          ip, {ft_x[idx], ft_y[idx], ft_z[idx]}, en[idx], tds[idx], et[idx], dn0[idx], weight[idx], criterion[idx], unbroken[idx]};
    } else {
      // static_assert(always_false<T>::value, "Unsupported interaction type");
    }
  }

  ONIKA_HOST_DEVICE_FUNC inline double& En(const uint64_t idx) const {
    return en[idx];
  }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d& Tds(const uint64_t idx) const {
    return tds[idx];
  }

  ONIKA_HOST_DEVICE_FUNC inline double& Et(const uint64_t idx) const {
    return et[idx];
  }

  ONIKA_HOST_DEVICE_FUNC inline double& Criterion(const uint64_t idx) const {
    return criterion[idx];
  }

  ONIKA_HOST_DEVICE_FUNC inline void broke(const uint64_t idx) const {
    unbroken[idx] = false;
  }

  ONIKA_HOST_DEVICE_FUNC
  inline void update(const uint64_t idx, exaDEM::Interaction& item) const {
    ft_x[idx] = item.friction.x;
    ft_y[idx] = item.friction.y;
    ft_z[idx] = item.friction.z;

    mom_x[idx] = item.moment.x;
    mom_y[idx] = item.moment.y;
    mom_z[idx] = item.moment.z;
  }

  ONIKA_HOST_DEVICE_FUNC
  inline void update(const uint64_t idx, exaDEM::InnerBondInteraction& item) const {
    ft_x[idx] = item.friction.x;
    ft_y[idx] = item.friction.y;
    ft_z[idx] = item.friction.z;
    en[idx] = item.en;
    tds[idx] = item.tds;
    et[idx] = item.et;
    dn0[idx] = item.dn0;
    weight[idx] = item.weight;
    criterion[idx] = item.criterion;
    unbroken[idx] = item.unbroken;
  }
};
}  // namespace exaDEM
