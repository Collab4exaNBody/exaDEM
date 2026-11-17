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
#include <algorithm>
#include <cstring>
#include <vector>

#include <onika/math/basic_types_operators.h>

#include <exaDEM/interaction/inner_bond_interaction.hpp>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/interaction_enum.hpp>
#include <exaDEM/interaction/interaction_pair.hpp>
#include <exaDEM/interaction/placeholder_interaction.hpp>

#include "unit_test.hpp"

using exaDEM::InnerBondInteraction;
using exaDEM::Interaction;
using exaDEM::InteractionPair;
using exaDEM::InteractionType;
using exaDEM::InteractionTypeId;
using exaDEM::ParticleSubLocation;
using exaDEM::PlaceholderInteraction;
using exanb::Vec3d;
namespace attr = exaDEM::attr;

namespace {
constexpr double tol = 1e-14;

InteractionPair make_pair(uint64_t id_i, uint64_t id_j, uint16_t type, uint8_t swap = 0,
                          uint8_t ghost = InteractionPair::NotGhost, uint32_t sub_i = 0, uint32_t sub_j = 0) {
  InteractionPair p;
  p.pi_ = {id_i, /*cell*/ 10, /*p*/ 1, sub_i};
  p.pj_ = {id_j, /*cell*/ 20, /*p*/ 2, sub_j};
  p.type_ = type;
  p.swap_ = swap;
  p.ghost_ = ghost;
  return p;
}

PlaceholderInteraction make_placeholder(const InteractionPair& pair) {
  PlaceholderInteraction p;
  p.pair_ = pair;
  return p;
}
}  // namespace

// ============================================================
// === interaction_enum.hpp
// ============================================================

UNIT_TEST(enum_ids_are_contiguous) {
  static_assert(InteractionTypeId::FirstIdParticle == 0);
  static_assert(InteractionTypeId::LastIdParticle + 1 == InteractionTypeId::FirstIdDriver);
  static_assert(InteractionTypeId::LastIdDriver + 1 == InteractionTypeId::FirstIdInnerBond);
  static_assert(InteractionTypeId::NTypesPP == 4);
  static_assert(InteractionTypeId::NTypesParticleDriver == 9);
  static_assert(InteractionTypeId::NTypesInnerBond == 1);
  static_assert(InteractionTypeId::NTypes == InteractionTypeId::LastIdInnerBond + 1);
  static_assert(InteractionTypeId::NTypesParticleParticle == InteractionTypeId::FirstIdInnerBond);
  CHECK(InteractionTypeId::NTypes == 14);
}

UNIT_TEST(enum_get_typed_classifies_every_id) {
  for (int id = InteractionTypeId::FirstIdParticle; id <= InteractionTypeId::LastIdParticle; id++) {
    CHECK(exaDEM::get_typed(id) == static_cast<int>(InteractionType::ParticleParticle));
  }
  for (int id = InteractionTypeId::FirstIdDriver; id <= InteractionTypeId::LastIdDriver; id++) {
    CHECK(exaDEM::get_typed(id) == static_cast<int>(InteractionType::ParticleDriver));
  }
  CHECK(exaDEM::get_typed(InteractionTypeId::InnerBondId) == static_cast<int>(InteractionType::InnerBond));
  CHECK(exaDEM::get_typed(InteractionTypeId::Undefined) == 3);
  CHECK(exaDEM::get_typed(-1) == 3);
}

UNIT_TEST(enum_get_type_idx_is_local_to_its_family) {
  CHECK(exaDEM::get_type_idx(InteractionTypeId::VertexVertex) == 0);
  CHECK(exaDEM::get_type_idx(InteractionTypeId::EdgeEdge) == 3);
  CHECK(exaDEM::get_type_idx(InteractionTypeId::VertexCylinder) == 0);
  CHECK(exaDEM::get_type_idx(InteractionTypeId::FaceRshapeDriverVertex) == 8);
  CHECK(exaDEM::get_type_idx(InteractionTypeId::InnerBondId) == 0);
  CHECK(exaDEM::get_type_idx(InteractionTypeId::Undefined) == InteractionTypeId::Undefined);

  CHECK(exaDEM::get_typed_idx<InteractionType::ParticleParticle>(InteractionTypeId::VertexFace) == 2);
  CHECK(exaDEM::get_typed_idx<InteractionType::ParticleDriver>(InteractionTypeId::VertexSurface) == 1);
  CHECK(exaDEM::get_typed_idx<InteractionType::InnerBond>(InteractionTypeId::InnerBondId) == 0);
  for (int id = 0; id < InteractionTypeId::NTypes; id++) {
    int expected = 0;
    switch (exaDEM::get_typed(id)) {
      case 0:
        expected = exaDEM::get_typed_idx<InteractionType::ParticleParticle>(id);
        break;
      case 1:
        expected = exaDEM::get_typed_idx<InteractionType::ParticleDriver>(id);
        break;
      case 2:
        expected = exaDEM::get_typed_idx<InteractionType::InnerBond>(id);
        break;
    }
    CHECK(exaDEM::get_type_idx(id) == expected);
  }
}

UNIT_TEST(enum_compile_time_conversion) {
  static_assert(exaDEM::ConvertToIntertactionType<InteractionTypeId::VertexVertex>() ==
                InteractionType::ParticleParticle);
  static_assert(exaDEM::ConvertToIntertactionType<InteractionTypeId::EdgeEdge>() == InteractionType::ParticleParticle);
  static_assert(exaDEM::ConvertToIntertactionType<InteractionTypeId::VertexCylinder>() ==
                InteractionType::ParticleDriver);
  static_assert(exaDEM::ConvertToIntertactionType<InteractionTypeId::FaceRshapeDriverVertex>() ==
                InteractionType::ParticleDriver);
  static_assert(exaDEM::ConvertToIntertactionType<InteractionTypeId::InnerBondId>() == InteractionType::InnerBond);
  CHECK(exaDEM::get_name<InteractionType::ParticleParticle>() == "InteractionType::ParticleParticle");
  CHECK(exaDEM::get_name<InteractionType::ParticleDriver>() == "InteractionType::ParticleDriver");
  CHECK(exaDEM::get_name<InteractionType::InnerBond>() == "InteractionType::InnerBond");
}

// ============================================================
// === InteractionPair
// ============================================================

UNIT_TEST(pair_default_is_undefined_and_inconsistent) {
  InteractionPair p;
  CHECK(p.type_ == InteractionTypeId::Undefined);
  CHECK(p.swap_ == 0);
  CHECK(p.ghost_ == InteractionPair::NotGhost);
  CHECK(!p.consistent<false>());
}

UNIT_TEST(pair_owner_partner_follow_swap) {
  InteractionPair p = make_pair(1, 2, InteractionTypeId::VertexFace);
  CHECK(p.owner().id_ == 1);
  CHECK(p.partner().id_ == 2);
  p.swap_ = 1;
  CHECK(p.owner().id_ == 2);
  CHECK(p.partner().id_ == 1);
  const InteractionPair& cp = p;
  CHECK(cp.owner().id_ == 2);
  CHECK(cp.partner().id_ == 1);
  // owner()/partner() are references into pi_/pj_
  p.owner().cell_ = 99;
  CHECK(p.pj_.cell_ == 99);
}

UNIT_TEST(pair_active_depends_on_ghost) {
  InteractionPair p = make_pair(1, 2, InteractionTypeId::VertexVertex);
  CHECK(p.active());
  p.ghost_ = InteractionPair::OwnerGhost;
  CHECK(p.active());
  p.ghost_ = InteractionPair::PartnerGhost;
  CHECK(!p.active());
}

UNIT_TEST(pair_consistent) {
  CHECK(make_pair(1, 2, InteractionTypeId::VertexVertex).consistent<false>());
  CHECK(make_pair(1, 2, InteractionTypeId::InnerBondId, 1, InteractionPair::PartnerGhost).consistent<false>());
  // same particle but different sub-elements is fine
  CHECK(make_pair(1, 1, InteractionTypeId::EdgeEdge, 0, 0, 3, 4).consistent<false>());

  InteractionPair bad_type = make_pair(1, 2, InteractionTypeId::NTypes);
  CHECK(!bad_type.consistent<false>());

  InteractionPair self = make_pair(1, 2, InteractionTypeId::VertexVertex);
  self.pj_ = self.pi_;
  CHECK(!self.consistent<false>());

  InteractionPair bad_ghost = make_pair(1, 2, InteractionTypeId::VertexVertex, 0, InteractionPair::PartnerGhost + 1);
  CHECK(!bad_ghost.consistent<false>());

  InteractionPair bad_swap = make_pair(1, 2, InteractionTypeId::VertexVertex, 2);
  CHECK(!bad_swap.consistent<false>());
}

UNIT_TEST(pair_equality_ignores_cell_p_and_ghost) {
  InteractionPair a = make_pair(1, 2, InteractionTypeId::VertexEdge, 0, 0, 3, 4);
  InteractionPair b = a;
  b.pi_.cell_ = 1234;
  b.pj_.p_ = 42;
  b.ghost_ = InteractionPair::PartnerGhost;
  CHECK(a == b);  // cells/positions change with migration, the identity of the contact does not

  InteractionPair c = a;
  c.pi_.sub_ = 5;
  CHECK(!(a == c));
  InteractionPair d = a;
  d.type_ = InteractionTypeId::VertexFace;
  CHECK(!(a == d));
  InteractionPair e = a;
  e.swap_ = 1;
  CHECK(!(a == e));
  InteractionPair f = a;
  f.pj_.id_ = 3;
  CHECK(!(a == f));
}

UNIT_TEST(pair_ordering_is_lexicographic) {
  using T = InteractionTypeId;
  // owner id > partner id > type > swap > owner sub > partner sub
  CHECK(make_pair(1, 9, T::EdgeEdge) < make_pair(2, 0, T::VertexVertex));
  CHECK(make_pair(1, 2, T::EdgeEdge) < make_pair(1, 3, T::VertexVertex));
  CHECK(make_pair(1, 2, T::VertexVertex) < make_pair(1, 2, T::VertexEdge));
  CHECK(make_pair(1, 2, T::VertexVertex, 0, 0, 5, 5) < make_pair(1, 2, T::VertexVertex, 0, 0, 6, 0));
  CHECK(make_pair(1, 2, T::VertexVertex, 0, 0, 5, 5) < make_pair(1, 2, T::VertexVertex, 0, 0, 5, 6));
  InteractionPair a = make_pair(1, 2, T::VertexVertex);
  CHECK(!(a < a));

  // ordering uses owner/partner: a swapped pair sorts by its owner (pj_)
  InteractionPair swapped = make_pair(5, 1, T::VertexVertex, 1);
  CHECK(swapped < make_pair(2, 0, T::VertexVertex));

  std::vector<InteractionPair> v = {make_pair(3, 1, T::VertexVertex), make_pair(1, 2, T::VertexFace),
                                    make_pair(1, 2, T::VertexVertex), make_pair(2, 7, T::EdgeEdge)};
  std::sort(v.begin(), v.end());
  CHECK(v[0] == make_pair(1, 2, T::VertexVertex));
  CHECK(v[1] == make_pair(1, 2, T::VertexFace));
  CHECK(v[2] == make_pair(2, 7, T::EdgeEdge));
  CHECK(v[3] == make_pair(3, 1, T::VertexVertex));
}

// ============================================================
// === Interaction
// ============================================================

UNIT_TEST(interaction_accessors) {
  Interaction I;
  I.pair_ = make_pair(7, 8, InteractionTypeId::VertexSurface, 1);
  CHECK(I.i().id_ == 7);
  CHECK(I.j().id_ == 8);
  CHECK(I.owner().id_ == 8);
  CHECK(I.partner().id_ == 7);
  CHECK(I.driver().id_ == 8);
  CHECK(I.driver_id() == 8);
  CHECK(I.type() == InteractionTypeId::VertexSurface);
  CHECK(I.cell() == 20);  // owner is pj_ (swap)
  CHECK(&I.pair_info() == &I.pair_);
  CHECK(!I.persistent());
  CHECK(!I.ignore_other_interactions());
}

UNIT_TEST(interaction_fields_are_independent) {
  Interaction I;
  I[attr::friction] = Vec3d{1, 2, 3};
  I[attr::moment] = Vec3d{4, 5, 6};
  I[attr::weight] = 0.25;
  const Interaction& cI = I;
  CHECK_VEC3_NEAR(cI[attr::friction], (Vec3d{1, 2, 3}), tol);
  CHECK_VEC3_NEAR(cI[attr::moment], (Vec3d{4, 5, 6}), tol);
  CHECK_NEAR(cI[attr::weight], 0.25, tol);
}

UNIT_TEST(interaction_reset) {
  Interaction I;
  I.pair_ = make_pair(1, 2, InteractionTypeId::VertexVertex);
  I[attr::friction] = Vec3d{1, 2, 3};
  I[attr::moment] = Vec3d{4, 5, 6};
  I[attr::weight] = 0.25;
  I.reset();
  CHECK_VEC3_NEAR(I[attr::friction], (Vec3d{0, 0, 0}), tol);
  CHECK_VEC3_NEAR(I[attr::moment], (Vec3d{0, 0, 0}), tol);
  CHECK_NEAR(I[attr::weight], 1.0, tol);                               // neutral scaling of kn/kt
  CHECK(I.pair_ == make_pair(1, 2, InteractionTypeId::VertexVertex));  // pair untouched
}

UNIT_TEST(interaction_active) {
  Interaction I;
  I.pair_ = make_pair(1, 2, InteractionTypeId::VertexVertex);
  I.reset();
  CHECK(!I.active());  // no history
  I[attr::friction] = Vec3d{1e-12, 0, 0};
  CHECK(I.active());
  I.reset();
  I[attr::moment] = Vec3d{0, 0, -1};
  CHECK(I.active());
  I.reset();
  I[attr::weight] = 0.5;
  CHECK(!I.active());  // weight is not history
  I[attr::friction] = Vec3d{1, 0, 0};
  I.pair_.ghost_ = InteractionPair::PartnerGhost;
  CHECK(!I.active());  // duplicated interaction is never active
}

UNIT_TEST(interaction_update_copies_fields_only) {
  Interaction a;
  a.pair_ = make_pair(1, 2, InteractionTypeId::VertexVertex);
  a.reset();
  Interaction b;
  b.pair_ = make_pair(3, 4, InteractionTypeId::EdgeEdge);
  b[attr::friction] = Vec3d{1, 2, 3};
  b[attr::moment] = Vec3d{4, 5, 6};
  b[attr::weight] = 0.5;
  a.update(b);
  CHECK(a.pair_ == make_pair(1, 2, InteractionTypeId::VertexVertex));
  CHECK_VEC3_NEAR(a[attr::friction], (Vec3d{1, 2, 3}), tol);
  CHECK_VEC3_NEAR(a[attr::moment], (Vec3d{4, 5, 6}), tol);
  CHECK_NEAR(a[attr::weight], 0.5, tol);
}

UNIT_TEST(interaction_comparison_uses_pair) {
  Interaction a, b;
  a.pair_ = make_pair(1, 2, InteractionTypeId::VertexVertex);
  b.pair_ = a.pair_;
  a[attr::friction] = Vec3d{1, 0, 0};
  CHECK(a == b);
  b.pair_.pj_.id_ = 3;
  CHECK(!(a == b));
  CHECK(a < b);
  CHECK(!(b < a));
}

// ============================================================
// === InnerBondInteraction
// ============================================================

UNIT_TEST(inner_bond_defaults) {
  InnerBondInteraction I;
  CHECK_VEC3_NEAR(I[attr::friction], (Vec3d{0, 0, 0}), tol);
  CHECK_NEAR(I[attr::en], 0.0, tol);
  CHECK_VEC3_NEAR(I[attr::tds], (Vec3d{0, 0, 0}), tol);
  CHECK_NEAR(I[attr::et], 0.0, tol);
  CHECK_NEAR(I[attr::dn0], 0.0, tol);
  CHECK_NEAR(I[attr::weight], 1.0, tol);
  CHECK(I[attr::criterion].mode_ == exaDEM::RuptureMode::None);
  CHECK(I[attr::unbroken]);
  CHECK(I.persistent());
  CHECK(I.ignore_other_interactions());
}

UNIT_TEST(inner_bond_reset_keeps_weight_and_criterion) {
  InnerBondInteraction I;
  I.pair_ = make_pair(1, 2, InteractionTypeId::InnerBondId);
  I[attr::friction] = Vec3d{1, 2, 3};
  I[attr::en] = 1.5;
  I[attr::tds] = Vec3d{4, 5, 6};
  I[attr::et] = 2.5;
  I[attr::dn0] = -0.1;
  I[attr::weight] = 0.2;
  I[attr::criterion].mode_ = exaDEM::RuptureMode::EnergyMixedMode;
  I[attr::criterion].energy_criterion() = 7.0;
  I[attr::unbroken] = false;
  I.reset();
  CHECK_VEC3_NEAR(I[attr::friction], (Vec3d{0, 0, 0}), tol);
  CHECK_NEAR(I[attr::en], 0.0, tol);
  CHECK_VEC3_NEAR(I[attr::tds], (Vec3d{0, 0, 0}), tol);
  CHECK_NEAR(I[attr::et], 0.0, tol);
  CHECK_NEAR(I[attr::dn0], 0.0, tol);
  CHECK(I[attr::unbroken]);
  CHECK_NEAR(I[attr::weight], 0.2, tol);
  CHECK(I[attr::criterion].mode_ == exaDEM::RuptureMode::EnergyMixedMode);
  CHECK_NEAR(I[attr::criterion].criteria_1_, 7.0, tol);
}

UNIT_TEST(inner_bond_persistent_and_active) {
  InnerBondInteraction I;
  I.pair_ = make_pair(1, 2, InteractionTypeId::InnerBondId);
  CHECK(I.active());  // a bond is active without any history
  CHECK(I.persistent());
  I[attr::unbroken] = false;
  CHECK(!I.persistent());
  CHECK(I.active());
  I.pair_.ghost_ = InteractionPair::PartnerGhost;
  CHECK(!I.active());
}

UNIT_TEST(inner_bond_update_copies_fields_only) {
  InnerBondInteraction a, b;
  a.pair_ = make_pair(1, 2, InteractionTypeId::InnerBondId);
  b.pair_ = make_pair(3, 4, InteractionTypeId::InnerBondId);
  b[attr::en] = 3.0;
  b[attr::dn0] = -0.5;
  b[attr::unbroken] = false;
  a.update(b);
  CHECK(a.pair_ == make_pair(1, 2, InteractionTypeId::InnerBondId));
  CHECK_NEAR(a[attr::en], 3.0, tol);
  CHECK_NEAR(a[attr::dn0], -0.5, tol);
  CHECK(!a[attr::unbroken]);
}

UNIT_TEST(inner_bond_broke_interaction) {
  InnerBondInteraction ib;
  ib.pair_ = make_pair(1, 2, InteractionTypeId::InnerBondId, 1, InteractionPair::OwnerGhost, 3, 4);
  ib[attr::friction] = Vec3d{1, 2, 3};
  ib[attr::en] = 9.0;
  ib[attr::unbroken] = false;
  Interaction I = exaDEM::broke_interaction(ib);
  CHECK(I.type() == InteractionTypeId::VertexVertex);
  CHECK(I.pair_.pi_ == ib.pair_.pi_);
  CHECK(I.pair_.pj_ == ib.pair_.pj_);
  CHECK(I.pair_.swap_ == 1);
  CHECK(I.pair_.ghost_ == InteractionPair::OwnerGhost);
  CHECK_VEC3_NEAR(I[attr::friction], (Vec3d{1, 2, 3}), tol);
  CHECK_VEC3_NEAR(I[attr::moment], (Vec3d{0, 0, 0}), tol);
}

UNIT_TEST(rupture_criterion_accessors) {
  exaDEM::RuptureCriteria c;
  CHECK(exaDEM::display(c.mode_) == "None");
  c.mode_ = exaDEM::RuptureMode::StressEnergySeparateMode;
  c.energy_criterion() = 1.0;
  c.stress_criterion() = 2.0;
  CHECK_NEAR(c.criteria_1_, 1.0, tol);
  CHECK_NEAR(c.criteria_2_, 2.0, tol);
  c.mode_ = exaDEM::RuptureMode::EnergySeparateMode;
  CHECK_NEAR(c.energy_normal_criterion(), 1.0, tol);
  CHECK_NEAR(c.energy_tangential_criterion(), 2.0, tol);
  CHECK(exaDEM::display(exaDEM::RuptureMode::EnergyMixedMode) == "EnergyMixedMode");
  CHECK(exaDEM::display(exaDEM::RuptureMode::EnergySeparateMode) == "EnergySeparateMode");
  CHECK(exaDEM::display(exaDEM::RuptureMode::StressEnergySeparateMode) == "StressEnergySeparateMode");
}

// ============================================================
// === PlaceholderInteraction
// ============================================================

UNIT_TEST(placeholder_layout) {
  static_assert(sizeof(PlaceholderInteraction) >= sizeof(Interaction));
  static_assert(sizeof(PlaceholderInteraction) >= sizeof(InnerBondInteraction));
  static_assert(alignof(PlaceholderInteraction) >= alignof(Interaction));
  static_assert(alignof(PlaceholderInteraction) >= alignof(InnerBondInteraction));
  static_assert(std::is_trivially_copyable_v<PlaceholderInteraction>);
  PlaceholderInteraction p;
  // pair_ is shared by every view of the placeholder
  CHECK(static_cast<void*>(&p.as<Interaction>().pair_) == static_cast<void*>(&p.pair_));
  CHECK(static_cast<void*>(&p.as<InnerBondInteraction>().pair_) == static_cast<void*>(&p.pair_));
}

UNIT_TEST(placeholder_default_data_is_zero) {
  PlaceholderInteraction p;
  CHECK(p.type() == InteractionTypeId::Undefined);
  bool all_zero = true;
  for (int i = 0; i < PlaceholderInteraction::PlaceholderInteractionSize; i++) {
    all_zero = all_zero && (p.data_[i] == 0);
  }
  CHECK(all_zero);
}

UNIT_TEST(placeholder_assign_from_interaction) {
  Interaction I;
  I.pair_ = make_pair(1, 2, InteractionTypeId::VertexEdge);
  I[attr::friction] = Vec3d{1, 2, 3};
  I[attr::moment] = Vec3d{4, 5, 6};
  I[attr::weight] = 0.5;
  PlaceholderInteraction p;
  p = I;
  CHECK(p.pair_ == I.pair_);
  CHECK(p.type() == InteractionTypeId::VertexEdge);
  const PlaceholderInteraction& cp = p;
  CHECK_VEC3_NEAR(cp.as<Interaction>()[attr::friction], (Vec3d{1, 2, 3}), tol);
  CHECK_VEC3_NEAR(cp.as<Interaction>()[attr::moment], (Vec3d{4, 5, 6}), tol);
  CHECK_NEAR(cp.as<Interaction>()[attr::weight], 0.5, tol);
  CHECK(&p.convert<InteractionType::ParticleParticle>() == &p.as<Interaction>());
  CHECK(&p.convert<InteractionType::ParticleDriver>() == &p.as<Interaction>());
}

UNIT_TEST(placeholder_assign_from_inner_bond) {
  InnerBondInteraction I;
  I.pair_ = make_pair(1, 2, InteractionTypeId::InnerBondId);
  I[attr::en] = 2.0;
  I[attr::dn0] = -0.3;
  I[attr::weight] = 0.25;
  I[attr::criterion].mode_ = exaDEM::RuptureMode::EnergySeparateMode;
  I[attr::unbroken] = false;
  PlaceholderInteraction p;
  p = I;
  CHECK(p.type() == InteractionTypeId::InnerBondId);
  auto& ib = p.convert<InteractionType::InnerBond>();
  CHECK_NEAR(ib[attr::en], 2.0, tol);
  CHECK_NEAR(ib[attr::dn0], -0.3, tol);
  CHECK_NEAR(ib[attr::weight], 0.25, tol);
  CHECK(ib[attr::criterion].mode_ == exaDEM::RuptureMode::EnergySeparateMode);
  CHECK(!ib[attr::unbroken]);
}

UNIT_TEST(placeholder_dispatches_on_type) {
  // particle-particle / particle-driver -> Interaction semantics
  for (uint16_t type : {uint16_t(InteractionTypeId::VertexVertex), uint16_t(InteractionTypeId::VertexCylinder),
                        uint16_t(InteractionTypeId::FaceRshapeDriverVertex)}) {
    PlaceholderInteraction p = make_placeholder(make_pair(1, 2, type));
    p.reset();
    CHECK(!p.active());
    CHECK(!p.persistent());
    CHECK(!p.ignore_other_interactions());
    CHECK_NEAR(p.as<Interaction>()[attr::weight], 1.0, tol);
    p.as<Interaction>()[attr::friction] = Vec3d{1, 0, 0};
    CHECK(p.active());
    CHECK(p.consistent<false>());
  }

  // inner bond -> InnerBondInteraction semantics
  PlaceholderInteraction p = make_placeholder(make_pair(1, 2, InteractionTypeId::InnerBondId));
  p.reset();
  CHECK(p.active());
  CHECK(p.persistent());
  CHECK(p.ignore_other_interactions());
  p.as<InnerBondInteraction>()[attr::unbroken] = false;
  CHECK(!p.persistent());
  const PlaceholderInteraction& cp = p;
  CHECK(!cp.persistent());
}

UNIT_TEST(placeholder_update_and_clear) {
  Interaction I;
  I.pair_ = make_pair(1, 2, InteractionTypeId::VertexVertex);
  I[attr::friction] = Vec3d{1, 2, 3};
  PlaceholderInteraction src;
  src = I;
  PlaceholderInteraction dst = make_placeholder(I.pair_);
  dst.update(src);
  CHECK_VEC3_NEAR(dst.as<Interaction>()[attr::friction], (Vec3d{1, 2, 3}), tol);
  CHECK(std::memcmp(dst.data_, src.data_, PlaceholderInteraction::PlaceholderInteractionSize) == 0);

  dst.clear_placeholder();
  CHECK_VEC3_NEAR(dst.as<Interaction>()[attr::friction], (Vec3d{0, 0, 0}), tol);
  CHECK(dst.pair_ == I.pair_);  // clearing the data keeps the pair
}

UNIT_TEST(placeholder_sort_by_owner_p) {
  PlaceholderInteraction a = make_placeholder(make_pair(9, 2, InteractionTypeId::VertexVertex));
  PlaceholderInteraction b = make_placeholder(make_pair(1, 2, InteractionTypeId::VertexVertex));
  a.owner().p_ = 0;
  b.owner().p_ = 1;
  CHECK(a.sort_by_owner_p(b));
  CHECK(!b.sort_by_owner_p(a));
  b.owner().p_ = 0;  // tie on p -> falls back to pair ordering
  CHECK(b.sort_by_owner_p(a));
  CHECK(!a.sort_by_owner_p(b));
}

UNIT_TEST(placeholder_update_from_history) {
  auto make = [](uint64_t i, uint64_t j, double fx) {
    Interaction I;
    I.pair_ = make_pair(i, j, InteractionTypeId::VertexVertex);
    I.reset();
    I[attr::friction] = Vec3d{fx, 0, 0};
    PlaceholderInteraction p;
    p = I;
    return p;
  };
  // history must be sorted
  std::vector<PlaceholderInteraction> history = {make(1, 2, 10), make(1, 3, 20), make(4, 5, 30)};
  std::sort(history.begin(), history.end());
  // new list: contacts detected at this step, without history
  std::vector<PlaceholderInteraction> current = {make(4, 5, 0), make(1, 2, 0), make(7, 8, 0), make(0, 1, 0)};
  exaDEM::update(current, history);
  CHECK_NEAR(current[0].as<Interaction>()[attr::friction].x, 30, tol);
  CHECK_NEAR(current[1].as<Interaction>()[attr::friction].x, 10, tol);
  CHECK_NEAR(current[2].as<Interaction>()[attr::friction].x, 0, tol);  // new contact
  CHECK_NEAR(current[3].as<Interaction>()[attr::friction].x, 0, tol);  // new contact, before history.begin()
}

UNIT_TEST(placeholder_filter_duplicates) {
  PlaceholderInteraction p = make_placeholder(make_pair(1, 2, InteractionTypeId::VertexVertex));
  CHECK(exaDEM::filter_duplicates(p));
  p.pair_.ghost_ = InteractionPair::OwnerGhost;
  CHECK(exaDEM::filter_duplicates(p));
  p.pair_.ghost_ = InteractionPair::PartnerGhost;
  CHECK(!exaDEM::filter_duplicates(p));
}

UNIT_TEST_MAIN()
