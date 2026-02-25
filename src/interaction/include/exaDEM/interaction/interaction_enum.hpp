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
enum InteractionType {
  ParticleParticle,
  ParticleDriver,
  InnerBond
};

struct InteractionTypeId {
  // classic
  static constexpr int NTypesParticleParticle = 13;
  static constexpr int VertexVertex = 0;
  static constexpr int VertexEdge = 1;
  static constexpr int VertexFace = 2;
  static constexpr int EdgeEdge = 3;
  static constexpr int FirstIdParticle = VertexVertex;
  static constexpr int LastIdParticle = EdgeEdge;
  static constexpr int NTypesPP = LastIdParticle - FirstIdParticle + 1;

  // drivers
  static constexpr int VertexCylinder = 4;
  static constexpr int VertexSurface = 5;
  static constexpr int VertexBall = 6;
  static constexpr int FirstIdDriver = VertexCylinder;  
  static constexpr int LastIdDriver = 12;
  static constexpr int NTypesParticleDriver = LastIdDriver - FirstIdDriver + 1;
  // fragmentation
  static constexpr int NTypesStickecParticles = 1;
  static constexpr int InnerBond = 13;
  static constexpr int FirstIdInnerBond = InnerBond;
  static constexpr int LastIdInnerBond = InnerBond;
  static constexpr int NTypesInnerBond = LastIdInnerBond - FirstIdInnerBond + 1;
  static constexpr int NTypes = NTypesPP + NTypesParticleDriver + NTypesInnerBond;
  // control initialization
  static constexpr int Undefined = 666;
};

ONIKA_HOST_DEVICE_FUNC inline
int get_type_idx(int idx) {
  static_assert(InteractionTypeId::LastIdParticle < InteractionTypeId::FirstIdDriver);
  static_assert(InteractionTypeId::LastIdDriver < InteractionTypeId::FirstIdInnerBond);
  if (idx <= InteractionTypeId::LastIdParticle) {
    return idx;
  } else if (idx <= InteractionTypeId::LastIdDriver) {
    return (idx - InteractionTypeId::FirstIdDriver);
  } else if (idx <= InteractionTypeId::LastIdInnerBond) {
    return (idx - InteractionTypeId::FirstIdInnerBond);
  }
  return InteractionTypeId::Undefined;
} 

template<InteractionType IT>
ONIKA_HOST_DEVICE_FUNC inline
int get_typed_idx(int idx) {
  if constexpr (IT == InteractionType::ParticleParticle) {
    return idx;
  } else if constexpr (IT == InteractionType::ParticleDriver) {
    return (idx - InteractionTypeId::FirstIdDriver);
  } else if constexpr(IT == InteractionType::InnerBond) {
    return (idx - InteractionTypeId::FirstIdInnerBond);
  }
} 

ONIKA_HOST_DEVICE_FUNC inline
int get_typed(int idx) {
  if (idx >=InteractionTypeId::FirstIdParticle &&
      idx <= InteractionTypeId::LastIdParticle) {
    return 0;
  } else if (idx >= InteractionTypeId::FirstIdDriver &&
      idx <= InteractionTypeId::LastIdDriver) {
    return 1;
  } else if (idx >= InteractionTypeId::FirstIdInnerBond &&
      idx <= InteractionTypeId::LastIdInnerBond) {
    return 2;
  }
  return 3;
}

template<InteractionType> constexpr int get_first_id();

template<> constexpr int get_first_id<InteractionType::ParticleParticle>() {
return InteractionTypeId::FirstIdParticle;
}

template<> constexpr int get_first_id<InteractionType::ParticleDriver>() {
return InteractionTypeId::FirstIdDriver;
}
template<> constexpr int get_first_id<InteractionType::InnerBond>() {
return InteractionTypeId::FirstIdInnerBond;
}

template<InteractionType> constexpr int get_last_id();

template<> constexpr int get_last_id<InteractionType::ParticleParticle>() {
return InteractionTypeId::LastIdParticle;
}
template<> constexpr int get_last_id<InteractionType::ParticleDriver>() {
return InteractionTypeId::LastIdDriver;
}
template<> constexpr int get_last_id<InteractionType::InnerBond>() {
return InteractionTypeId::LastIdInnerBond;
}

template<int Type>
constexpr InteractionType ConvertToIntertactionType() {
  if constexpr (Type >= get_first_id<InteractionType::ParticleParticle>() &&
                Type <= get_last_id<InteractionType::ParticleParticle>()) {
    return InteractionType::ParticleParticle;
  } else if constexpr (Type >= get_first_id<InteractionType::ParticleDriver>() &&
                       Type <= get_last_id<InteractionType::ParticleDriver>()) {
    return InteractionType::ParticleDriver;
  } else if constexpr (Type >= get_first_id<InteractionType::InnerBond>() &&
                       Type <= get_last_id<InteractionType::InnerBond>()) {
    return InteractionType::InnerBond;
  }
}

template<InteractionType IT> inline std::string get_name();
template<> inline
std::string get_name<InteractionType::ParticleParticle>() {
  return "InteractionType::ParticleParticle";
}
template<> inline
std::string get_name<InteractionType::ParticleDriver>() {
  return "InteractionType::ParticleDriver";
}
template<> inline
std::string get_name<InteractionType::InnerBond>() {
  return "InteractionType::InnerBond";
}
}  // namespace exaDEM
