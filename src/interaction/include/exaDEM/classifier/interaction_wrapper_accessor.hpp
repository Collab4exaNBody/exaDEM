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

namespace exaDEM {
struct InteractionWrapperAccessor {
  InteractionWrapper<ParticleParticle>* particleparticle;
  InteractionWrapper<ParticleDriver>* particledriver;
  InteractionWrapper<InnerBond>* innerbond;

  template<InteractionType IT>
  ONIKA_HOST_DEVICE_FUNC auto& get_typed_accessor(int idx) const {
    if constexpr (IT == InteractionType::ParticleParticle) {
      return particleparticle[get_typed_idx<IT>(idx)];
    } else if constexpr (IT == InteractionType::ParticleDriver) {
      return particledriver[get_typed_idx<IT>(idx)];
    } else if constexpr(IT == InteractionType::InnerBond) {
      return innerbond[get_typed_idx<IT>(idx)];
    }
  } 
};

struct InteractionWrapperStorage {
  template <typename T>
  using VectorT = onika::memory::CudaMMVector<T>;
  VectorT<InteractionWrapper<ParticleParticle>> particleparticle;
  VectorT<InteractionWrapper<ParticleDriver>> particledriver;
  VectorT<InteractionWrapper<InnerBond>> innerbond;

  InteractionWrapperStorage(Classifier& classifier) {
    particleparticle.resize(InteractionTypeId::NTypesPP);
    for (size_t i = InteractionTypeId::FirstIdParticle ; i <= InteractionTypeId::LastIdParticle ; i++) {
      auto& c = classifier.get_data<InteractionType::ParticleParticle>(i);
      particleparticle[i] = InteractionWrapper(c);
    }
    particledriver.resize(InteractionTypeId::NTypesParticleDriver);
    for (size_t i = InteractionTypeId::FirstIdDriver ; i <= InteractionTypeId::LastIdDriver ; i++) {
      auto& c = classifier.get_data<InteractionType::ParticleDriver>(i);
      particledriver[i - InteractionTypeId::FirstIdDriver] = InteractionWrapper<InteractionType::ParticleDriver>(c);
    }
    innerbond.resize(InteractionTypeId::NTypesStickecParticles);
    for (size_t i = InteractionTypeId::FirstIdInnerBond ; i <= InteractionTypeId::LastIdInnerBond ; i++) {
      auto& c = classifier.get_data<InteractionType::InnerBond>(i);
      innerbond[i - InteractionTypeId::FirstIdInnerBond] = InteractionWrapper(c);
    }
  }

  InteractionWrapperAccessor accessor() {
    InteractionWrapperAccessor res;
    res.particleparticle = particleparticle.data();
    res.particledriver = particledriver.data();
    res.innerbond = innerbond.data();
    return res;
  }
 private:
  InteractionWrapperStorage() {}
};

template<InteractionType... Types>
struct InteractionDispatcher
{
  template<typename Func, typename... Args>
  ONIKA_HOST_DEVICE_FUNC
  static inline void dispatch(
      uint16_t type,
      const InteractionWrapperAccessor& iwa,
      const Func& func,
      Args&&... args)
  {
    ((get_typed(type) == static_cast<int>(Types)
      ? (func.template operator()<Types>(
              iwa.template get_typed_accessor<Types>(type),
      std::forward<Args>(args)...), 0)
        : 0), ...);
  }
};

using IDispatcher = InteractionDispatcher<InteractionType::ParticleParticle, InteractionType::ParticleDriver, InteractionType::InnerBond>;
}  // namespace exaDEM
