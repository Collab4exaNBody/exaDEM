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
#include <vector>

#include <onika/math/basic_types_operators.h>

#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/classifier/classifier_dispatcher.hpp>

#include "unit_test.hpp"

using exaDEM::Classifier;
using exaDEM::ClassifierContainer;
using exaDEM::InnerBondInteraction;
using exaDEM::Interaction;
using exaDEM::InteractionPair;
using exaDEM::InteractionType;
using exaDEM::InteractionTypeId;
using exaDEM::PlaceholderInteraction;
using exanb::Vec3d;
namespace attr = exaDEM::attr;

namespace {
constexpr double tol = 1e-14;

using ContainerPP = ClassifierContainer<InteractionType::ParticleParticle>;
using ContainerPD = ClassifierContainer<InteractionType::ParticleDriver>;
using ContainerIB = ClassifierContainer<InteractionType::InnerBond>;

// Builds a distinct, consistent pair for index k, every field of the pair being different.
InteractionPair make_pair(uint16_t type, uint64_t k) {
  InteractionPair p;
  p.pi_ = {100 + k, uint32_t(200 + k), uint16_t(300 + k), uint32_t(400 + k)};
  p.pj_ = {500 + k, uint32_t(600 + k), uint16_t(700 + k), uint32_t(800 + k)};
  p.type_ = type;
  p.swap_ = k % 2;
  p.ghost_ = k % 3;
  return p;
}

PlaceholderInteraction make_contact(uint16_t type, uint64_t k) {
  Interaction I;
  I.pair_ = make_pair(type, k);
  I.reset();
  I[attr::friction] = Vec3d{double(k), 1, 2};
  I[attr::moment] = Vec3d{3, double(k), 4};
  I[attr::weight] = 1.0 / double(k + 1);
  PlaceholderInteraction p;
  p = I;
  return p;
}

PlaceholderInteraction make_bond(uint64_t k) {
  InnerBondInteraction I;
  I.pair_ = make_pair(InteractionTypeId::InnerBondId, k);
  I[attr::friction] = Vec3d{double(k), 0, 0};
  I[attr::en] = 1.0 + k;
  I[attr::tds] = Vec3d{0, double(k), 0};
  I[attr::et] = 2.0 + k;
  I[attr::dn0] = -0.01 * k;
  I[attr::weight] = 0.5;
  I[attr::criterion].mode_ = exaDEM::RuptureMode::EnergySeparateMode;
  I[attr::criterion].criteria_1_ = 10.0 + k;
  I[attr::criterion].criteria_2_ = 20.0 + k;
  I[attr::unbroken] = (k % 2 == 0);
  PlaceholderInteraction p;
  p = I;
  return p;
}

std::vector<PlaceholderInteraction> make_contacts(uint16_t type, size_t n, uint64_t first = 0) {
  std::vector<PlaceholderInteraction> v;
  for (size_t k = 0; k < n; k++) {
    v.push_back(make_contact(type, first + k));
  }
  return v;
}

std::vector<PlaceholderInteraction> make_bonds(size_t n) {
  std::vector<PlaceholderInteraction> v;
  for (size_t k = 0; k < n; k++) {
    v.push_back(make_bond(k));
  }
  return v;
}

void check_pair(const InteractionPair& a, const InteractionPair& b) {
  CHECK(a.pi_ == b.pi_);
  CHECK(a.pj_ == b.pj_);
  CHECK(a.type_ == b.type_);
  CHECK(a.swap_ == b.swap_);
  CHECK(a.ghost_ == b.ghost_);
}

void check_contact(const Interaction& got, const PlaceholderInteraction& expected) {
  const Interaction& e = expected.as<Interaction>();
  check_pair(got.pair_, e.pair_);
  CHECK_VEC3_NEAR(got[attr::friction], e[attr::friction], tol);
  CHECK_VEC3_NEAR(got[attr::moment], e[attr::moment], tol);
  CHECK_NEAR(got[attr::weight], e[attr::weight], tol);
}

void check_bond(const InnerBondInteraction& got, const PlaceholderInteraction& expected) {
  const InnerBondInteraction& e = expected.as<InnerBondInteraction>();
  check_pair(got.pair_, e.pair_);
  CHECK_VEC3_NEAR(got[attr::friction], e[attr::friction], tol);
  CHECK_NEAR(got[attr::en], e[attr::en], tol);
  CHECK_VEC3_NEAR(got[attr::tds], e[attr::tds], tol);
  CHECK_NEAR(got[attr::et], e[attr::et], tol);
  CHECK_NEAR(got[attr::dn0], e[attr::dn0], tol);
  CHECK_NEAR(got[attr::weight], e[attr::weight], tol);
  CHECK(got[attr::criterion].mode_ == e[attr::criterion].mode_);
  CHECK_NEAR(got[attr::criterion].criteria_1_, e[attr::criterion].criteria_1_, tol);
  CHECK_NEAR(got[attr::criterion].criteria_2_, e[attr::criterion].criteria_2_, tol);
  CHECK(got[attr::unbroken] == e[attr::unbroken]);
}

// Every per-field vector of a container must keep the same size.
struct SizeCollector {
  std::vector<size_t> sizes;
  template <typename T>
  void operator()(T& vec) {
    sizes.push_back(vec.size());
  }
};

template <typename ContainerT>
bool all_fields_have_size(ContainerT& c, size_t n) {
  SizeCollector col;
  c.apply_on_fields(col);
  bool ok = !col.sizes.empty();
  for (size_t s : col.sizes) {
    ok = ok && (s == n);
  }
  return ok;
}
}  // namespace

// ============================================================
// === ClassifierContainer
// ============================================================

UNIT_TEST(container_pp_insert_and_read_back) {
  ContainerPP c;
  c.type_ = InteractionTypeId::VertexFace;
  CHECK(c.size() == 0);
  auto first = make_contacts(InteractionTypeId::VertexFace, 3);
  c.insert(first, InteractionTypeId::VertexFace);
  CHECK(c.size() == 3);
  auto second = make_contacts(InteractionTypeId::VertexFace, 2, 3);
  c.insert(second, InteractionTypeId::VertexFace);  // insert appends
  CHECK(c.size() == 5);
  CHECK(all_fields_have_size(c, 5));
  for (size_t k = 0; k < 3; k++) {
    check_contact(c[k], first[k]);
  }
  for (size_t k = 0; k < 2; k++) {
    check_contact(c[3 + k], second[k]);
  }
  CHECK(c[0].type() == InteractionTypeId::VertexFace);
}

UNIT_TEST(container_pd_insert_and_read_back) {
  ContainerPD c;
  c.type_ = InteractionTypeId::VertexSurface;
  auto v = make_contacts(InteractionTypeId::VertexSurface, 4);
  c.insert(v, InteractionTypeId::VertexSurface);
  CHECK(c.size() == 4);
  CHECK(all_fields_have_size(c, 4));
  for (size_t k = 0; k < v.size(); k++) {
    check_contact(c[k], v[k]);
    CHECK(c[k].driver_id() == v[k].pair_.pj_.id_);
  }
}

UNIT_TEST(container_ib_insert_and_read_back) {
  ContainerIB c;
  c.type_ = InteractionTypeId::InnerBondId;
  auto v = make_bonds(5);
  c.insert(v, InteractionTypeId::InnerBondId);
  CHECK(c.size() == 5);
  CHECK(all_fields_have_size(c, 5));
  for (size_t k = 0; k < v.size(); k++) {
    check_bond(c[k], v[k]);
  }
}

UNIT_TEST(container_insert_empty_is_noop) {
  ContainerPP c;
  c.type_ = InteractionTypeId::VertexVertex;
  std::vector<PlaceholderInteraction> empty;
  c.insert(empty, InteractionTypeId::VertexVertex);
  CHECK(c.size() == 0);
}

UNIT_TEST(container_resize_copy_clear) {
  ContainerPP c;
  c.type_ = InteractionTypeId::EdgeEdge;
  c.resize(6);
  CHECK(c.size() == 6);
  CHECK(all_fields_have_size(c, 6));
  auto head = make_contacts(InteractionTypeId::EdgeEdge, 2);
  auto tail = make_contacts(InteractionTypeId::EdgeEdge, 4, 2);
  c.copy(2, tail.size(), tail, InteractionTypeId::EdgeEdge);  // copy does not reorder nor resize
  c.copy(0, head.size(), head, InteractionTypeId::EdgeEdge);
  CHECK(c.size() == 6);
  for (size_t k = 0; k < 2; k++) {
    check_contact(c[k], head[k]);
  }
  for (size_t k = 0; k < 4; k++) {
    check_contact(c[2 + k], tail[k]);
  }
  c.clear();
  CHECK(c.size() == 0);
  CHECK(all_fields_have_size(c, 0));
}

UNIT_TEST(container_update_changes_fields_not_pair) {
  ContainerPP c;
  c.type_ = InteractionTypeId::VertexVertex;
  auto v = make_contacts(InteractionTypeId::VertexVertex, 3);
  c.insert(v, InteractionTypeId::VertexVertex);
  // the new item has a different pair: update() only touches friction/moment/weight
  PlaceholderInteraction other = make_contact(InteractionTypeId::VertexVertex, 42);
  c.update(1, other);
  Interaction got = c[1];
  check_pair(got.pair_, v[1].pair_);
  CHECK_VEC3_NEAR(got[attr::friction], (Vec3d{42, 1, 2}), tol);
  CHECK_VEC3_NEAR(got[attr::moment], (Vec3d{3, 42, 4}), tol);
  CHECK_NEAR(got[attr::weight], 1.0 / 43.0, tol);
  check_contact(c[0], v[0]);
  check_contact(c[2], v[2]);
}

UNIT_TEST(container_ib_update) {
  ContainerIB c;
  c.type_ = InteractionTypeId::InnerBondId;
  auto v = make_bonds(2);
  c.insert(v, InteractionTypeId::InnerBondId);
  PlaceholderInteraction item = v[0];
  item.as<InnerBondInteraction>()[attr::unbroken] = false;
  item.as<InnerBondInteraction>()[attr::en] = 123.0;
  c.update(0, item);
  CHECK(!c[0][attr::unbroken]);
  CHECK_NEAR(c[0][attr::en], 123.0, tol);
  check_bond(c[1], v[1]);
}

UNIT_TEST(container_for_all_interactions) {
  ContainerPP c;
  c.type_ = InteractionTypeId::VertexEdge;
  auto v = make_contacts(InteractionTypeId::VertexEdge, 4);
  c.insert(v, InteractionTypeId::VertexEdge);
  double sum = 0;
  size_t count = 0;
  auto func = [&count](Interaction& I, double& acc) {
    acc += I[attr::friction].x;
    count++;
  };
  exaDEM::for_all_interactions(c, func, sum);
  CHECK(count == 4);
  CHECK_NEAR(sum, 0 + 1 + 2 + 3, tol);
}

// ============================================================
// === ClassifierContainer::View
// ============================================================

UNIT_TEST(view_reads_container) {
  ContainerPP c;
  c.type_ = InteractionTypeId::VertexVertex;
  auto v = make_contacts(InteractionTypeId::VertexVertex, 3);
  c.insert(v, InteractionTypeId::VertexVertex);
  auto view = c.view();
  CHECK(view.m_type == InteractionTypeId::VertexVertex);
  CHECK(view[attr::id_i].size() == 3);
  CHECK(view[attr::friction].size() == 3);
  for (size_t k = 0; k < v.size(); k++) {
    check_contact(view(k), v[k]);
    check_pair(view.pair(k), v[k].pair_);
    CHECK(view.same(k, v[k]));
  }
  CHECK(!view.same(0, v[1]));
}

UNIT_TEST(view_writes_through_to_container) {
  ContainerPD c;
  c.type_ = InteractionTypeId::VertexBall;
  auto v = make_contacts(InteractionTypeId::VertexBall, 3);
  c.insert(v, InteractionTypeId::VertexBall);
  auto view = c.view();

  // update() writes the specialized fields only
  PlaceholderInteraction other = make_contact(InteractionTypeId::VertexBall, 7);
  view.update(1, other);
  Interaction got = c[1];
  check_pair(got.pair_, v[1].pair_);
  CHECK_VEC3_NEAR(got[attr::friction], (Vec3d{7, 1, 2}), tol);

  // set() writes the pair only
  view.set(2, other);
  got = c[2];
  check_pair(got.pair_, other.pair_);
  CHECK_VEC3_NEAR(got[attr::friction], v[2].as<Interaction>()[attr::friction], tol);
  CHECK(view.same(2, other));

  // update() with an Interaction
  Interaction I = other.as<Interaction>();
  I[attr::moment] = Vec3d{-1, -2, -3};
  view.update(0, I);
  CHECK_VEC3_NEAR(c[0][attr::moment], (Vec3d{-1, -2, -3}), tol);
}

UNIT_TEST(view_ib) {
  ContainerIB c;
  c.type_ = InteractionTypeId::InnerBondId;
  auto v = make_bonds(3);
  c.insert(v, InteractionTypeId::InnerBondId);
  auto view = c.view();
  for (size_t k = 0; k < v.size(); k++) {
    check_bond(view(k), v[k]);
  }
  InnerBondInteraction I = view(0);
  I[attr::unbroken] = false;
  I[attr::dn0] = -1.0;
  view.update(0, I);
  CHECK(!c[0][attr::unbroken]);
  CHECK_NEAR(c[0][attr::dn0], -1.0, tol);
}

// ============================================================
// === ContactState
// ============================================================

UNIT_TEST(contact_state_resize) {
  exaDEM::ContactState s;
  s.resize(4);
  CHECK(s.dn_.size() == 4);
  CHECK(s.cp_.size() == 4);
  CHECK(s.fn_.size() == 4);
  CHECK(s.ft_.size() == 4);
  s.resize(0);
  CHECK(s.dn_.size() == 0);
  CHECK(s.ft_.size() == 0);
}

// ============================================================
// === Classifier
// ============================================================

UNIT_TEST(classifier_initialization) {
  Classifier c;
  CHECK(c.particles_.size() == InteractionTypeId::NTypesPP);
  CHECK(c.drivers_.size() == InteractionTypeId::NTypesParticleDriver);
  CHECK(c.innerbonds_.size() == InteractionTypeId::NTypesInnerBond);
  CHECK(c.contact_state_.size() == InteractionTypeId::NTypes);
  CHECK(c.number_of_waves() == InteractionTypeId::NTypes);
  const Classifier& cc = c;
  CHECK(cc.number_of_waves() == InteractionTypeId::NTypes);

  for (int id = InteractionTypeId::FirstIdParticle; id <= InteractionTypeId::LastIdParticle; id++) {
    CHECK(c.get_data<InteractionType::ParticleParticle>(id).type_ == id);
    CHECK(c.get_size(id) == 0);
  }
  for (int id = InteractionTypeId::FirstIdDriver; id <= InteractionTypeId::LastIdDriver; id++) {
    CHECK(c.get_data<InteractionType::ParticleDriver>(id).type_ == id);
    CHECK(c.get_size(id) == 0);
  }
  CHECK(c.get_data<InteractionType::InnerBond>(InteractionTypeId::InnerBondId).type_ == InteractionTypeId::InnerBondId);
  CHECK(cc.get_data<InteractionType::ParticleDriver>(InteractionTypeId::VertexBall).type_ ==
        InteractionTypeId::VertexBall);
  CHECK(&c.get_container<InteractionType::ParticleParticle>() == &c.particles_);
  CHECK(&c.get_container<InteractionType::ParticleDriver>() == &c.drivers_);
  CHECK(&c.get_container<InteractionType::InnerBond>() == &c.innerbonds_);
}

UNIT_TEST(classifier_insert_per_type) {
  Classifier c;
  // put (id + 1) interactions in wave id
  for (int id = 0; id < InteractionTypeId::NTypes; id++) {
    const size_t n = id + 1;
    if (id == InteractionTypeId::InnerBondId) {
      auto v = make_bonds(n);
      c.get_data<InteractionType::InnerBond>(id).insert(v, id);
    } else if (exaDEM::get_typed(id) == 0) {
      auto v = make_contacts(id, n);
      c.get_data<InteractionType::ParticleParticle>(id).insert(v, id);
    } else {
      auto v = make_contacts(id, n);
      c.get_data<InteractionType::ParticleDriver>(id).insert(v, id);
    }
  }
  for (int id = 0; id < InteractionTypeId::NTypes; id++) {
    CHECK(c.get_size(id) == size_t(id + 1));
  }
  auto [pd, pd_size] = c.get_info<InteractionType::ParticleDriver>(InteractionTypeId::VertexCylinder);
  CHECK(pd_size == InteractionTypeId::VertexCylinder + 1);
  CHECK(&pd == &c.drivers_[0]);
  check_contact(pd[0], make_contact(InteractionTypeId::VertexCylinder, 0));
  const Classifier& cc = c;
  auto [ib, ib_size] = cc.get_info<InteractionType::InnerBond>(InteractionTypeId::InnerBondId);
  CHECK(ib_size == InteractionTypeId::InnerBondId + 1);
  CHECK(ib.type_ == InteractionTypeId::InnerBondId);

  c.reset_containers();
  for (int id = 0; id < InteractionTypeId::NTypes; id++) {
    CHECK(c.get_size(id) == 0);
  }
}

UNIT_TEST(classifier_resize_and_copy) {
  Classifier c;
  const int pp = InteractionTypeId::VertexEdge;
  const int pd = InteractionTypeId::EdgeRshapeDriverEdge;
  const int ib = InteractionTypeId::InnerBondId;
  c.resize(pp, 4);
  c.resize(pd, 2);
  c.resize(ib, 3);
  CHECK(c.get_size(pp) == 4);
  CHECK(c.get_size(pd) == 2);
  CHECK(c.get_size(ib) == 3);
  CHECK(c.get_size(InteractionTypeId::VertexVertex) == 0);  // other waves untouched

  auto vpp = make_contacts(pp, 2);
  c.copy(pp, 2, 2, vpp);
  auto& wpp = c.get_data<InteractionType::ParticleParticle>(pp);
  check_contact(wpp[2], vpp[0]);
  check_contact(wpp[3], vpp[1]);

  auto vpd = make_contacts(pd, 2);
  c.copy(pd, 0, 2, vpd);
  auto& wpd = c.get_data<InteractionType::ParticleDriver>(pd);
  check_contact(wpd[0], vpd[0]);
  check_contact(wpd[1], vpd[1]);

  auto vib = make_bonds(3);
  c.copy(ib, 0, 3, vib);
  auto& wib = c.get_data<InteractionType::InnerBond>(ib);
  for (size_t k = 0; k < 3; k++) {
    check_bond(wib[k], vib[k]);
  }
  check_bond(c.get_sticked_interaction_wrapper()(1), vib[1]);
}

UNIT_TEST(classifier_contact_state_fits_wave_size) {
  Classifier c;
  const int type = InteractionTypeId::VertexFace;
  auto v = make_contacts(type, 5);
  c.get_data<InteractionType::ParticleParticle>(type).insert(v, type);
  auto [dn, cp, fn, ft] = c.contact_state(type);
  CHECK(c.contact_state_[type].dn_.size() == 5);
  CHECK(c.contact_state_[type].ft_.size() == 5);
  CHECK(dn == c.contact_state_[type].dn_.data());
  CHECK(cp == c.contact_state_[type].cp_.data());
  CHECK(fn == c.contact_state_[type].fn_.data());
  CHECK(ft == c.contact_state_[type].ft_.data());
  CHECK(c.contact_state_[InteractionTypeId::VertexVertex].dn_.size() == 0);

  c.reset_containers();
  c.contact_state(type);
  CHECK(c.contact_state_[type].dn_.size() == 0);
}

UNIT_TEST(classifier_for_all_interactions) {
  Classifier c;
  auto v0 = make_contacts(InteractionTypeId::VertexVertex, 2);
  c.get_data<InteractionType::ParticleParticle>(InteractionTypeId::VertexVertex).insert(v0, 0);
  auto v5 = make_contacts(InteractionTypeId::VertexSurface, 3);
  c.get_data<InteractionType::ParticleDriver>(InteractionTypeId::VertexSurface)
      .insert(v5, InteractionTypeId::VertexSurface);
  auto vib = make_bonds(4);
  c.get_data<InteractionType::InnerBond>(InteractionTypeId::InnerBondId).insert(vib, InteractionTypeId::InnerBondId);

  std::vector<int> count(InteractionTypeId::NTypes, 0);
  auto func = [&count](auto& I) { count[I.type()]++; };
  exaDEM::for_all_interactions(c, func);
  for (int id = 0; id < InteractionTypeId::NTypes; id++) {
    const int expected = (id == InteractionTypeId::VertexVertex)    ? 2
                         : (id == InteractionTypeId::VertexSurface) ? 3
                         : (id == InteractionTypeId::InnerBondId)   ? 4
                                                                    : 0;
    CHECK(count[id] == expected);
  }
}

// ============================================================
// === ClassifierViewStorage / IDispatcher
// ============================================================

namespace {
struct RecordViewFunc {
  template <InteractionType IT, typename ViewT>
  void operator()(ViewT& view, uint16_t* type, size_t* size, int* family) const {
    static_assert(ViewT::kInteractionType == IT);
    *type = view.m_type;
    *size = view[attr::id_i].size();
    *family = static_cast<int>(IT);
  }
};
}  // namespace

UNIT_TEST(dispatcher_selects_the_right_view) {
  Classifier c;
  for (int id = 0; id < InteractionTypeId::NTypes; id++) {
    const size_t n = 2 * id + 1;
    if (id == InteractionTypeId::InnerBondId) {
      auto v = make_bonds(n);
      c.get_data<InteractionType::InnerBond>(id).insert(v, id);
    } else if (exaDEM::get_typed(id) == 0) {
      auto v = make_contacts(id, n);
      c.get_data<InteractionType::ParticleParticle>(id).insert(v, id);
    } else {
      auto v = make_contacts(id, n);
      c.get_data<InteractionType::ParticleDriver>(id).insert(v, id);
    }
  }
  exaDEM::ClassifierViewStorage storage(c);
  auto accessor = storage.accessor();
  const RecordViewFunc func;
  for (int id = 0; id < InteractionTypeId::NTypes; id++) {
    uint16_t type = InteractionTypeId::Undefined;
    size_t size = 0;
    int family = -1;
    exaDEM::IDispatcher::dispatch(id, accessor, func, &type, &size, &family);
    CHECK(type == id);
    CHECK(size == size_t(2 * id + 1));
    CHECK(family == exaDEM::get_typed(id));
  }

  // an undefined type is dispatched nowhere
  uint16_t type = 0;
  size_t size = 0;
  int family = -1;
  exaDEM::IDispatcher::dispatch(InteractionTypeId::Undefined, accessor, func, &type, &size, &family);
  CHECK(family == -1);

  // views from the accessor alias the classifier storage
  auto& view = accessor.get_typed_accessor<InteractionType::ParticleDriver>(InteractionTypeId::VertexBall);
  PlaceholderInteraction item = make_contact(InteractionTypeId::VertexBall, 99);
  view.update(0, item);
  CHECK_VEC3_NEAR(c.get_data<InteractionType::ParticleDriver>(InteractionTypeId::VertexBall)[0][attr::friction],
                  (Vec3d{99, 1, 2}), tol);
}

UNIT_TEST_MAIN()
