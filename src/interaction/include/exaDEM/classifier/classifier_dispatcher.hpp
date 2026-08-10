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

namespace exaDEM {
struct ClassifierViewAccessor {
  ClassifierContainer<ParticleParticle>::View* particleparticle_;
  ClassifierContainer<ParticleDriver>::View* particledriver_;
  ClassifierContainer<InnerBond>::View* innerbond_;

  template <InteractionType IT>
  ONIKA_HOST_DEVICE_FUNC auto& get_typed_accessor(int idx) const {
    if constexpr (IT == InteractionType::ParticleParticle) {
      return particleparticle_[get_typed_idx<IT>(idx)];
    } else if constexpr (IT == InteractionType::ParticleDriver) {
      return particledriver_[get_typed_idx<IT>(idx)];
    } else if constexpr (IT == InteractionType::InnerBond) {
      return innerbond_[get_typed_idx<IT>(idx)];
    }
  }
};

struct ClassifierViewStorage {
  template <typename T>
  using VectorT = onika::memory::CudaMMVector<T>;
  VectorT<ClassifierContainer<ParticleParticle>::View> particleparticle_;
  VectorT<ClassifierContainer<ParticleDriver>::View> particledriver_;
  VectorT<ClassifierContainer<InnerBond>::View> innerbond_;

  ClassifierViewStorage(Classifier& classifier) {
    particleparticle_.resize(InteractionTypeId::NTypesPP);
    for (size_t i = InteractionTypeId::FirstIdParticle; i <= InteractionTypeId::LastIdParticle; i++) {
      auto& c = classifier.get_data<InteractionType::ParticleParticle>(i);
      particleparticle_[i] = c.view();
    }
    particledriver_.resize(InteractionTypeId::NTypesParticleDriver);
    for (size_t i = InteractionTypeId::FirstIdDriver; i <= InteractionTypeId::LastIdDriver; i++) {
      auto& c = classifier.get_data<InteractionType::ParticleDriver>(i);
      particledriver_[i - InteractionTypeId::FirstIdDriver] = c.view();
    }
    innerbond_.resize(InteractionTypeId::NTypesStickecParticles);
    for (size_t i = InteractionTypeId::FirstIdInnerBond; i <= InteractionTypeId::LastIdInnerBond; i++) {
      auto& c = classifier.get_data<InteractionType::InnerBond>(i);
      innerbond_[i - InteractionTypeId::FirstIdInnerBond] = c.view();
    }
  }

  ClassifierViewAccessor accessor() {
    ClassifierViewAccessor res;
    res.particleparticle_ = particleparticle_.data();
    res.particledriver_ = particledriver_.data();
    res.innerbond_ = innerbond_.data();
    return res;
  }

 private:
  ClassifierViewStorage() {}
};

template <InteractionType... Types>
struct ClassifierDispatcher {
  template <typename Func, typename... Args>
  ONIKA_HOST_DEVICE_FUNC static inline void dispatch(uint16_t type, const ClassifierViewAccessor& iva, const Func& func,
                                                     Args&&... args) {
    ((get_typed(type) == static_cast<int>(Types)
          ? (func.template operator()<Types>(iva.template get_typed_accessor<Types>(type), std::forward<Args>(args)...),
             0)
          : 0),
     ...);
  }
};

using IDispatcher = ClassifierDispatcher<InteractionType::ParticleParticle, InteractionType::ParticleDriver,
                                         InteractionType::InnerBond>;
}  // namespace exaDEM
