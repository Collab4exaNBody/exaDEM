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

/**
 * Unit tests of the building blocks of the nbh_polyhedron operator:
 *  - InteractionManager (history, persistent interactions, ignored pairs, storage layout),
 *  - add_polyhedron_pair_interactions (polyhedron/polyhedron detection, compared to a brute-force reference),
 *  - add_driver_interaction (polyhedron/driver detection).
 */

// Driver headers are not self-contained: they are included in the same order as in the operators.
#include <exanb/core/grid.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <onika/scg/operator.h>

#include <onika/math/basic_types_operators.h>
#include <onika/math/quaternion_operators.h>

#include <exaDEM/drivers.hpp>
#include <exaDEM/interaction/interaction_manager.hpp>
#include <exaDEM/polyhedron/nbh_polyhedron_driver.hpp>
#include <exaDEM/polyhedron/nbh_polyhedron_pair.hpp>
#include <exaDEM/shape.hpp>
#include <exaDEM/shape/basic_shape/cube.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <tuple>
#include <vector>

#include "unit_test.hpp"

using exaDEM::InnerBondInteraction;
using exaDEM::Interaction;
using exaDEM::InteractionManager;
using exaDEM::InteractionPair;
using exaDEM::InteractionTypeId;
using exaDEM::ParticleVertexView;
using exaDEM::PlaceholderInteraction;
using exaDEM::PolyhedronPairSide;
using exaDEM::shape;
using exaDEM::VertexField;
using exanb::Quaternion;
using exanb::Vec3d;
namespace attr = exaDEM::attr;

namespace {
constexpr double tol = 1e-14;
constexpr double cube_length = 1.0;
constexpr double cube_radius = 0.1;  // minkowski radius
constexpr double rVerlet = 0.1;

using Storage = exanb::CellExtraDynamicDataStorageT<PlaceholderInteraction>;

// ------------------------------------------------------------
// --- helpers: interactions
// ------------------------------------------------------------

PlaceholderInteraction make_contact(uint64_t id_i, uint64_t id_j, uint16_t p_i, uint32_t cell = 0,
                                    uint16_t type = InteractionTypeId::VertexVertex, uint32_t sub_i = 0,
                                    uint32_t sub_j = 0) {
  Interaction I;
  I.pair_.pi_ = {id_i, cell, p_i, sub_i};
  I.pair_.pj_ = {id_j, cell + 1, 0, sub_j};
  I.pair_.type_ = type;
  I.reset();
  PlaceholderInteraction p;
  p = I;
  return p;
}

PlaceholderInteraction make_bond(uint64_t id_i, uint64_t id_j, uint16_t p_i, uint32_t cell = 0, uint32_t sub_i = 0,
                                 bool unbroken = true) {
  InnerBondInteraction I;
  I.pair_.pi_ = {id_i, cell, p_i, sub_i};
  I.pair_.pj_ = {id_j, cell, 0, 0};
  I.pair_.type_ = InteractionTypeId::InnerBondId;
  I[attr::unbroken] = unbroken;
  PlaceholderInteraction p;
  p = I;
  return p;
}

// Stores `items` (already sorted by owner p) in a cell storage of n_particles particles.
Storage make_storage(size_t n_particles, std::vector<PlaceholderInteraction> items) {
  InteractionManager m;
  m.reset(n_particles);
  for (auto& it : items) {
    m.list_[it.owner().p_].push_back(it);
  }
  Storage s;
  s.initialize(n_particles);
  m.update_extra_storage<false>(s);
  return s;
}

// ------------------------------------------------------------
// --- helpers: polyhedra
// ------------------------------------------------------------

shape make_cube() { return exaDEM::basic_shape::create_cube("cube", cube_length, cube_radius); }

struct Particle {
  uint64_t id;
  Vec3d r;
  Quaternion q = {1, 0, 0, 0};
  double h = 1.0;
};

// Vertex positions of every particle, computed the same way as compute_vertices.
struct Scene {
  shape shp = make_cube();
  std::vector<Particle> particles;
  VertexField vertices;

  void build() {
    vertices.resize(particles.size(), shp.get_number_of_vertices());
    for (size_t p = 0; p < particles.size(); p++) {
      for (int v = 0; v < shp.get_number_of_vertices(); v++) {
        vertices.set(shp.get_vertex(v, particles[p].r, particles[p].h, particles[p].q), p, v);
      }
    }
  }

  PolyhedronPairSide side(size_t p, uint32_t cell = 0) {
    return {particles[p].id, cell,           uint16_t(p), particles[p].r,
            particles[p].q,  particles[p].h, &shp,        ParticleVertexView{p, vertices}};
  }
};

// Everything the detection writes in an interaction: identifies one detected contact.
using ContactKey =
    std::tuple<int /*type*/, int /*swap*/, uint64_t /*id_i*/, uint64_t /*id_j*/, uint32_t /*sub_i*/, uint32_t /*sub_j*/,
               uint32_t /*cell_i*/, uint32_t /*cell_j*/, int /*p_i*/, int /*p_j*/, int /*ghost*/>;

ContactKey key_of(const PlaceholderInteraction& item) {
  const auto& pi = item.pair_.pi_;
  const auto& pj = item.pair_.pj_;
  return {item.pair_.type_, item.pair_.swap_, pi.id_, pj.id_, pi.sub_,          pj.sub_,
          pi.cell_,         pj.cell_,         pi.p_,  pj.p_,  item.pair_.ghost_};
}

// Same callback as in nbh_polyhedron, but collecting the contacts instead of feeding an InteractionManager.
struct Collector {
  std::vector<PlaceholderInteraction> items;
  void operator()(PlaceholderInteraction& item, int sub_i, int sub_j) {
    item.pair_.pi_.sub_ = sub_i;
    item.pair_.pj_.sub_ = sub_j;
    item.reset();
    items.push_back(item);
  }
  std::multiset<ContactKey> keys() const {
    std::multiset<ContactKey> res;
    for (auto& it : items) {
      res.insert(key_of(it));
    }
    return res;
  }
  size_t count(int type, int swap) const {
    return std::count_if(items.begin(), items.end(),
                         [=](auto& it) { return it.pair_.type_ == type && it.pair_.swap_ == swap; });
  }
};

// Reference: the narrow-phase filters on every (sub_i, sub_j) couple, without any OBB pruning.
std::multiset<ContactKey> brute_force(PolyhedronPairSide& a, PolyhedronPairSide& b, int ghost) {
  std::multiset<ContactKey> res;
  const shape* sa = a.shp_;
  const shape* sb = b.shp_;
  auto add = [&](int type, int swap, PolyhedronPairSide& i, PolyhedronPairSide& j, int si, int sj) {
    res.insert({type, swap, i.id_, j.id_, uint32_t(si), uint32_t(sj), i.cell_, j.cell_, i.p_, j.p_, ghost});
  };
  for (int i = 0; i < sa->get_number_of_vertices(); i++) {
    for (int j = 0; j < sb->get_number_of_vertices(); j++) {
      if (exaDEM::filter_vertex_vertex(rVerlet, a.vertices_, a.h_, i, sa, b.vertices_, b.h_, j, sb)) {
        add(InteractionTypeId::VertexVertex, 0, a, b, i, j);
      }
    }
    for (int j = 0; j < sb->get_number_of_edges(); j++) {
      if (exaDEM::filter_vertex_edge(rVerlet, a.vertices_, a.h_, i, sa, b.vertices_, b.h_, j, sb)) {
        add(InteractionTypeId::VertexEdge, 0, a, b, i, j);
      }
    }
    for (int j = 0; j < sb->get_number_of_faces(); j++) {
      if (exaDEM::filter_vertex_face(rVerlet, a.vertices_, a.h_, i, sa, b.vertices_, b.h_, j, sb)) {
        add(InteractionTypeId::VertexFace, 0, a, b, i, j);
      }
    }
  }
  for (int i = 0; i < sa->get_number_of_edges(); i++) {
    for (int j = 0; j < sb->get_number_of_edges(); j++) {
      if (exaDEM::filter_edge_edge(rVerlet, a.vertices_, a.h_, i, sa, b.vertices_, b.h_, j, sb)) {
        add(InteractionTypeId::EdgeEdge, 0, a, b, i, j);
      }
    }
  }
  for (int j = 0; j < sb->get_number_of_vertices(); j++) {
    for (int i = 0; i < sa->get_number_of_edges(); i++) {
      if (exaDEM::filter_vertex_edge(rVerlet, b.vertices_, b.h_, j, sb, a.vertices_, a.h_, i, sa)) {
        add(InteractionTypeId::VertexEdge, 1, b, a, j, i);
      }
    }
    for (int i = 0; i < sa->get_number_of_faces(); i++) {
      if (exaDEM::filter_vertex_face(rVerlet, b.vertices_, b.h_, j, sb, a.vertices_, a.h_, i, sa)) {
        add(InteractionTypeId::VertexFace, 1, b, a, j, i);
      }
    }
  }
  return res;
}

// Two aligned cubes along x, with a gap `gap` between their (rounded) surfaces.
Scene face_to_face(double gap) {
  Scene s;
  s.particles = {{10, {0, 0, 0}}, {20, {cube_length + 2 * cube_radius + gap, 0, 0}}};
  s.build();
  return s;
}

Collector detect(Scene& s, int ghost = InteractionPair::NotGhost, bool* accepted = nullptr) {
  auto a = s.side(0, 3);
  auto b = s.side(1, 4);
  PlaceholderInteraction item;
  item.pair_.ghost_ = ghost;
  Collector c;
  bool res = exaDEM::add_polyhedron_pair_interactions(a, b, rVerlet, item, c);
  if (accepted) {
    *accepted = res;
  }
  return c;
}
}  // namespace

// ============================================================
// === InteractionManager
// ============================================================

UNIT_TEST(manager_add_item_routes_by_owner) {
  InteractionManager m;
  m.reset(3);
  CHECK(m.list_.size() == 3);
  CHECK(m.ignore_.size() == 3);
  PlaceholderInteraction a = make_contact(1, 2, /*p*/ 2);
  PlaceholderInteraction b = make_contact(1, 3, /*p*/ 0);
  b.pair_.swap_ = 1;  // owner is pj_ (p = 0)
  b.pair_.pi_.p_ = 2;
  m.add_item(a);
  m.add_item(b);
  CHECK(m.list_[2].size() == 1);
  CHECK(m.list_[0].size() == 1);
  CHECK(m.get_size() == 2);

  m.reset(2);  // reset clears and resizes
  CHECK(m.list_.size() == 2);
  CHECK(m.get_size() == 0);
}

UNIT_TEST(manager_update_extra_storage_layout) {
  InteractionManager m;
  m.reset(4);
  std::vector<PlaceholderInteraction> v = {make_contact(1, 2, 0), make_contact(1, 3, 0), make_contact(5, 6, 2),
                                           make_contact(7, 8, 3)};
  m.add(v);
  Storage s;
  s.initialize(4);
  m.update_extra_storage<false>(s);
  CHECK(s.m_data.size() == 4);
  CHECK(s.number_of_particles() == 4);
  CHECK(s.m_info[0].offset == 0 && s.m_info[0].size == 2);
  CHECK(s.m_info[1].offset == 2 && s.m_info[1].size == 0);
  CHECK(s.m_info[2].offset == 2 && s.m_info[2].size == 1);
  CHECK(s.m_info[3].offset == 3 && s.m_info[3].size == 1);
  CHECK(s.m_data[2].pair_.pi_.id_ == 5);
  CHECK(s.m_data[3].pair_.pi_.id_ == 7);
}

UNIT_TEST(manager_update_extra_storage_applies_history) {
  // history: two contacts with friction, sorted
  PlaceholderInteraction h1 = make_contact(1, 2, 0);
  h1.as<Interaction>()[attr::friction] = Vec3d{1, 2, 3};
  PlaceholderInteraction h2 = make_contact(4, 5, 1);
  h2.as<Interaction>()[attr::moment] = Vec3d{0, 0, 7};
  InteractionManager m;
  m.hist_ = {h1, h2};
  std::stable_sort(m.hist_.begin(), m.hist_.end());

  // new detection: the same pairs (cell/p may differ), without history, plus a new contact
  m.reset(2);
  std::vector<PlaceholderInteraction> v = {make_contact(1, 2, 0, /*cell*/ 9), make_contact(1, 9, 0),
                                           make_contact(4, 5, 1)};
  m.add(v);
  Storage s;
  s.initialize(2);
  m.update_extra_storage<true>(s);
  CHECK_VEC3_NEAR(s.m_data[0].as<Interaction>()[attr::friction], (Vec3d{1, 2, 3}), tol);
  CHECK_VEC3_NEAR(s.m_data[1].as<Interaction>()[attr::friction], (Vec3d{0, 0, 0}), tol);
  CHECK_VEC3_NEAR(s.m_data[2].as<Interaction>()[attr::moment], (Vec3d{0, 0, 7}), tol);
  CHECK(s.m_data[0].pair_.pi_.cell_ == 9);  // history only restores the specialized fields

  // without history nothing is restored
  m.reset(2);
  m.add(v);
  m.update_extra_storage<false>(s);
  CHECK_VEC3_NEAR(s.m_data[0].as<Interaction>()[attr::friction], (Vec3d{0, 0, 0}), tol);
}

UNIT_TEST(manager_extract_history) {
  PlaceholderInteraction inactive = make_contact(1, 2, 0);
  PlaceholderInteraction active = make_contact(1, 3, 0);
  active.as<Interaction>()[attr::friction] = Vec3d{1, 0, 0};
  PlaceholderInteraction ghost = make_contact(1, 4, 0);
  ghost.as<Interaction>()[attr::friction] = Vec3d{1, 0, 0};
  ghost.pair_.ghost_ = InteractionPair::PartnerGhost;
  PlaceholderInteraction bond = make_bond(1, 5, 0);
  std::vector<PlaceholderInteraction> data = {inactive, active, ghost, bond};

  std::vector<PlaceholderInteraction> hist = {make_contact(9, 9, 0)};  // cleared first
  exaDEM::extract_history(hist, data.data(), data.size());
  CHECK(hist.size() == 1);
  CHECK(hist[0] == active);
}

UNIT_TEST(manager_update_persistent_interactions) {
  // particle 0: one contact + two bonds (unsorted), particle 1: one broken bond
  Storage s = make_storage(2, {make_contact(1, 2, 0), make_bond(1, 7, 0, 0, 3), make_bond(1, 7, 0, 0, 1),
                               make_bond(4, 8, 1, 0, 0, /*unbroken*/ false)});
  InteractionManager m;
  m.current_cell_id_ = 0;
  m.current_cell_particles_ = 2;
  m.reset(2);
  exaDEM::update_persistent_interactions(m, s);
  CHECK(m.list_[0].size() == 2);  // only unbroken bonds are persistent
  CHECK(m.list_[1].size() == 0);
  CHECK(m.list_[0][0].pair_.pi_.sub_ == 1);  // sorted
  CHECK(m.list_[0][1].pair_.pi_.sub_ == 3);
}

UNIT_TEST(manager_ignore_bonded_partners) {
  InteractionManager m;
  m.reset(2);
  m.list_[0] = {make_bond(1, 7, 0, 0, 0), make_bond(1, 7, 0, 0, 1), make_bond(1, 8, 0, 0, 2)};
  m.list_[1] = {make_contact(4, 9, 1)};  // a regular contact never makes the partner ignored
  m.update_ignore_interaction();
  CHECK(m.ignore_[0].size() == 2);  // no duplicates
  CHECK(m.ignore_[1].empty());

  PlaceholderInteraction contact_with_bonded = make_contact(1, 7, 0);
  PlaceholderInteraction contact_with_other = make_contact(1, 9, 0);
  m.add_item(contact_with_bonded);
  m.add_item(contact_with_other);
  CHECK(m.list_[0].size() == 4);  // 3 bonds + contact_with_other
  CHECK(m.list_[0].back() == contact_with_other);
}

// ============================================================
// === add_polyhedron_pair_interactions
// ============================================================

UNIT_TEST(pair_far_apart_is_rejected) {
  Scene s = face_to_face(1.0);
  bool accepted = true;
  Collector c = detect(s, InteractionPair::NotGhost, &accepted);
  CHECK(!accepted);
  CHECK(c.items.empty());
}

UNIT_TEST(pair_face_to_face) {
  Scene s = face_to_face(0.5 * rVerlet);
  bool accepted = false;
  Collector c = detect(s, InteractionPair::NotGhost, &accepted);
  CHECK(accepted);
  // degenerate case (vertices of a on the boundary of the facing face of b): VV and VE contacts
  CHECK(c.count(InteractionTypeId::VertexVertex, 0) == 4);
  CHECK(c.count(InteractionTypeId::VertexEdge, 0) == 8);
  CHECK(c.count(InteractionTypeId::VertexEdge, 1) == 8);
  // VV and EE are only built from a to b
  CHECK(c.count(InteractionTypeId::VertexVertex, 1) == 0);
  CHECK(c.count(InteractionTypeId::EdgeEdge, 1) == 0);

  for (auto& it : c.items) {
    const auto& owner = it.owner();
    const auto& partner = it.partner();
    // the owner is always particle a (smallest id), whatever the direction
    CHECK(owner.id_ == 10 && owner.cell_ == 3 && owner.p_ == 0);
    CHECK(partner.id_ == 20 && partner.cell_ == 4 && partner.p_ == 1);
    CHECK(it.consistent<false>());
    // only the facing side is involved: vertices of a with x > 0, vertices of b with x < r_b
    const auto& sub_vertex_side = it.pair_.swap_ ? s.particles[1] : s.particles[0];
    if (it.type() != InteractionTypeId::EdgeEdge) {
      const Vec3d v = s.vertices(it.pair_.swap_ ? 1 : 0, it.pair_.pi_.sub_);
      CHECK((it.pair_.swap_ ? (v.x < sub_vertex_side.r.x) : (v.x > sub_vertex_side.r.x)));
    }
  }
  auto a = s.side(0, 3);
  auto b = s.side(1, 4);
  CHECK(c.keys() == brute_force(a, b, InteractionPair::NotGhost));
}

UNIT_TEST(pair_vertices_inside_face) {
  // b is twice as large as a: the 4 vertices of the +x face of a project strictly inside the -x face of b
  constexpr int face_minus_x = 4;  // {4, 7, 3, 0} in create_cube
  const double hb = 2.0;
  const double gap = 0.5 * rVerlet;
  Scene s;
  s.particles = {{10, {0, 0, 0}}, {20, {0.5 * cube_length + cube_radius + gap + hb * (cube_radius + 0.5), 0, 0}}};
  s.particles[1].h = hb;
  s.build();
  Collector c = detect(s);
  CHECK(c.count(InteractionTypeId::VertexFace, 0) == 4);
  std::set<uint32_t> vertices_of_a;
  for (auto& it : c.items) {
    if (it.type() == InteractionTypeId::VertexFace && !it.pair_.swap_) {
      CHECK(it.pair_.pj_.sub_ == face_minus_x);
      vertices_of_a.insert(it.pair_.pi_.sub_);
    }
  }
  CHECK((vertices_of_a == std::set<uint32_t>{1, 2, 5, 6}));  // +x vertices of create_cube
  // the vertices of b are far from a: no b -> a interaction, and no vertex/edge pair is close enough
  CHECK(c.count(InteractionTypeId::VertexFace, 1) == 0);
  CHECK(c.count(InteractionTypeId::VertexEdge, 1) == 0);
  CHECK(c.count(InteractionTypeId::VertexVertex, 0) == 0);
  auto a = s.side(0, 3);
  auto b = s.side(1, 4);
  CHECK(c.keys() == brute_force(a, b, InteractionPair::NotGhost));
}

UNIT_TEST(pair_detection_distance_threshold) {
  // potential contacts are kept while the gap between the rounded surfaces is below rVerlet
  Scene close = face_to_face(0.9 * rVerlet);
  CHECK(!detect(close).items.empty());
  Scene far = face_to_face(1.1 * rVerlet);
  CHECK(detect(far).items.empty());
}

UNIT_TEST(pair_corner_to_corner) {
  // b is moved along the diagonal: only the vertex 6 of a (+,+,+) and the vertex 0 of b (-,-,-) are close
  Scene s;
  const double d = (cube_length + (2 * cube_radius + 0.5 * rVerlet) / std::sqrt(3.0));
  s.particles = {{10, {0, 0, 0}}, {20, {d, d, d}}};
  s.build();
  Collector c = detect(s);
  CHECK(c.count(InteractionTypeId::VertexVertex, 0) == 1);
  for (auto& it : c.items) {
    if (it.type() == InteractionTypeId::VertexVertex) {
      CHECK(it.pair_.pi_.sub_ == 6);
      CHECK(it.pair_.pj_.sub_ == 0);
    }
  }
  CHECK(c.count(InteractionTypeId::VertexFace, 0) == 0);
  CHECK(c.count(InteractionTypeId::VertexFace, 1) == 0);
}

UNIT_TEST(pair_ghost_and_reset) {
  Scene s = face_to_face(0.5 * rVerlet);
  Collector c = detect(s, InteractionPair::OwnerGhost);
  CHECK(!c.items.empty());
  for (auto& it : c.items) {
    CHECK(it.pair_.ghost_ == InteractionPair::OwnerGhost);  // not modified by the detection
    CHECK_NEAR(it.as<Interaction>()[attr::weight], 1.0, tol);
    CHECK(!it.active());  // new contacts have no history
  }
}

UNIT_TEST(pair_obb_pruning_matches_brute_force) {
  // random orientations, homotheties and close positions: the OBB broad phase must never drop a contact
  std::mt19937 gen(12345);
  std::uniform_real_distribution<double> unif(-1.0, 1.0);
  std::uniform_real_distribution<double> homothety(0.5, 1.5);
  int tested = 0, with_contacts = 0;
  for (int trial = 0; trial < 300; trial++) {
    Scene s;
    auto random_quat = [&]() {
      Quaternion q = {unif(gen), unif(gen), unif(gen), unif(gen)};
      const double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
      return Quaternion{q.w / n, q.x / n, q.y / n, q.z / n};
    };
    const double ha = homothety(gen), hb = homothety(gen);
    Vec3d dir = {unif(gen), unif(gen), unif(gen)};
    dir = dir / exanb::norm(dir);
    const double dist = (0.5 + 0.4 * (unif(gen) + 1.0)) * (ha + hb) * cube_length;
    s.particles = {{1, {0, 0, 0}, random_quat(), ha}, {2, dist * dir, random_quat(), hb}};
    s.build();

    auto a = s.side(0, 0);
    auto b = s.side(1, 1);
    PlaceholderInteraction item;
    item.pair_.ghost_ = InteractionPair::NotGhost;
    Collector c;
    exaDEM::add_polyhedron_pair_interactions(a, b, rVerlet, item, c);
    const auto expected = brute_force(a, b, InteractionPair::NotGhost);
    CHECK(c.keys() == expected);
    if (c.keys() != expected) {
      std::cerr << "trial " << trial << ": " << c.items.size() << " contacts vs " << expected.size() << " (brute force)"
                << std::endl;
    }
    tested++;
    with_contacts += !expected.empty();
  }
  // the random scenes must cover both cases
  CHECK(with_contacts > 30);
  CHECK(tested - with_contacts > 30);
}

// ============================================================
// === add_driver_interaction
// ============================================================

UNIT_TEST(driver_surface_detects_bottom_vertices) {
  // plane z = 0, cube just above it: the 4 bottom vertices are potential contacts
  exaDEM::Surface surf;
  surf.fields_.offset_ = 0;
  surf.fields_.normal_ = {0, 0, 1};
  surf.fields_.center_ = {0, 0, 0};
  surf.motion_type_ = exaDEM::STATIONARY;
  exaDEM::Driver_params motion_params;
  surf.initialize(motion_params);

  std::vector<shape> shps = {make_cube()};
  const double z_close = 0.5 * cube_length + cube_radius + 0.5 * rVerlet;
  const double z_far = 0.5 * cube_length + cube_radius + 2.0 * rVerlet;
  std::vector<Vec3d> pos = {{0, 0, z_close}, {5, 5, z_far}};
  std::vector<uint64_t> ids = {42, 43};
  std::vector<exanb::ParticleTypeInt> types = {0, 0};
  std::vector<double> h = {1.0, 1.0};
  VertexField vertices;
  vertices.resize(2, shps[0].get_number_of_vertices());
  for (size_t p = 0; p < 2; p++) {
    for (int v = 0; v < shps[0].get_number_of_vertices(); v++) {
      vertices.set(shps[0].get_vertex(v, pos[p], h[p], Quaternion{1, 0, 0, 0}), p, v);
    }
  }

  PlaceholderInteraction item;
  item.pair_.type_ = InteractionTypeId::VertexSurface;
  item.pair_.ghost_ = InteractionPair::NotGhost;
  item.pair_.swap_ = 0;
  item.i().cell_ = 5;
  item.driver() = {/*driver idx*/ 0, uint32_t(-1), uint16_t(-1), 0};
  Collector c;
  exaDEM::add_driver_interaction(surf, c, item, 2, rVerlet, types.data(), ids.data(), vertices, h.data(), shps.data());
  CHECK(c.items.size() == 4);
  for (auto& it : c.items) {
    CHECK(it.type() == InteractionTypeId::VertexSurface);
    CHECK(it.pair_.pi_.id_ == 42);
    CHECK(it.pair_.pi_.p_ == 0);
    CHECK(it.pair_.pi_.cell_ == 5);
    CHECK(it.pair_.pj_.id_ == 0);                       // driver index
    CHECK(it.pair_.pj_.sub_ == uint32_t(-1));           // DRIVER_VERTEX_SUB_IDX
    CHECK(vertices(0, it.pair_.pi_.sub_).z < z_close);  // bottom face only (the far cube gives nothing)
  }
}

UNIT_TEST_MAIN()
