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
#include <stdexcept>  // contact_force.hpp uses std::logic_error
#include <utility>

#include <onika/math/basic_types_operators.h>

#include <exaDEM/forcefield/contact_force.hpp>

#include "unit_test.hpp"

using exaDEM::ContactParams;
using exanb::Vec3d;

namespace {
constexpr double tol = 1e-10;

// Two spheres of radius 1 and mass 1: i at the origin, j at (1.9, 0, 0).
// Same convention as detection_vertex_vertex_core: n goes from j to i, dn < 0 means overlap.
struct HeadOnContact {
  Vec3d pos_i = {0, 0, 0};
  Vec3d pos_j = {1.9, 0, 0};
  Vec3d vel_i = {0, 0, 0};
  Vec3d vel_j = {0, 0, 0};
  Vec3d vrot_i = {0, 0, 0};
  Vec3d vrot_j = {0, 0, 0};
  double dn = -0.1;
  Vec3d n = {-1, 0, 0};
  Vec3d contact_position = {0.95, 0, 0};  // pos_i - n * (r_i + dn / 2)
  double meff = 0.5;
  double reff = 0.5;
  double dt = 1e-3;

  // outputs
  Vec3d ft = {0, 0, 0};  // tangential force history
  Vec3d f_i = {0, 0, 0};
  Vec3d mom_i = {0, 0, 0};

  // the same contact seen from particle j
  HeadOnContact swapped() const {
    HeadOnContact s = *this;
    std::swap(s.pos_i, s.pos_j);
    std::swap(s.vel_i, s.vel_j);
    std::swap(s.vrot_i, s.vrot_j);
    s.n = -1.0 * n;
    s.ft = {0, 0, 0};
    s.f_i = {0, 0, 0};
    s.mom_i = {0, 0, 0};
    return s;
  }

  template <CohesiveLawType CohesiveLaw = CohesiveLawType::None>
  void compute(const ContactParams& cp) {
    exaDEM::contact_force_core<ContactLawType::Hooke, CohesiveLaw>(
        dn, n, dt, cp, meff, reff, ft, contact_position, pos_i, vel_i, f_i, mom_i, vrot_i, pos_j, vel_j, vrot_j);
  }
};

ContactParams make_params(double kn, double kt = 0, double mu = 0, double damp_rate = 0) {
  ContactParams cp;
  cp.kn_ = kn;
  cp.kt_ = kt;
  cp.mu_ = mu;
  cp.damp_rate_ = damp_rate;
  return cp;
}
}  // namespace

// ============================================================
// === Hooke (no cohesion)
// ============================================================

UNIT_TEST(hooke_static_overlap_is_repulsive) {
  HeadOnContact c;
  c.compute(make_params(1000.0));
  CHECK_VEC3_NEAR(c.f_i, (Vec3d{-100, 0, 0}), tol);  // kn * |dn|, pushes i away from j
  CHECK_VEC3_NEAR(c.ft, (Vec3d{0, 0, 0}), tol);
  CHECK_VEC3_NEAR(c.mom_i, (Vec3d{0, 0, 0}), tol);
}

UNIT_TEST(hooke_force_overwrites_f_i) {
  // f_i is an output (not accumulated): the caller adds it to the particle forces
  HeadOnContact c;
  c.f_i = {1e6, 1e6, 1e6};
  c.compute(make_params(1000.0));
  CHECK_VEC3_NEAR(c.f_i, (Vec3d{-100, 0, 0}), tol);
}

UNIT_TEST(hooke_damping) {
  const double damp = 0.5 * 2.0 * std::sqrt(1000.0 * 0.5);
  HeadOnContact approaching;
  approaching.vel_i = {1, 0, 0};
  approaching.compute(make_params(1000.0, 0, 0, 0.5));
  CHECK_NEAR(approaching.f_i.x, -(100 + damp), tol);  // damping increases the repulsion

  HeadOnContact separating;
  separating.vel_i = {-1, 0, 0};
  separating.compute(make_params(1000.0, 0, 0, 0.5));
  CHECK_NEAR(separating.f_i.x, -(100 - damp), tol);  // and decreases it while separating

  HeadOnContact fast_separating;
  fast_separating.vel_i = {-10, 0, 0};
  fast_separating.compute(make_params(1000.0, 0, 0, 0.5));
  CHECK_VEC3_NEAR(fast_separating.f_i, (Vec3d{0, 0, 0}), tol);  // never attractive
}

UNIT_TEST(hooke_friction_follows_relative_velocity) {
  // j slides along +y: friction drags i along +y
  HeadOnContact c;
  c.vel_j = {0, 1, 0};
  c.compute(make_params(1000.0, 1000.0, 0.5));
  CHECK_VEC3_NEAR(c.ft, (Vec3d{0, 1, 0}), tol);  // kt * dt * vt
  CHECK_VEC3_NEAR(c.f_i, (Vec3d{-100, 1, 0}), tol);
}

UNIT_TEST(hooke_friction_history_and_coulomb_limit) {
  HeadOnContact c;
  c.vel_j = {0, 1, 0};
  const ContactParams cp = make_params(1000.0, 1000.0, 0.5);
  for (int it = 0; it < 10; it++) {
    c.compute(cp);
  }
  CHECK_VEC3_NEAR(c.ft, (Vec3d{0, 10, 0}), tol);  // the tangential force is incremental

  for (int it = 0; it < 100; it++) {
    c.compute(cp);
  }
  CHECK_VEC3_NEAR(c.ft, (Vec3d{0, 50, 0}), tol);  // sliding: |ft| = mu * kn * |dn|
  CHECK_VEC3_NEAR(c.f_i, (Vec3d{-100, 50, 0}), tol);
}

UNIT_TEST(hooke_no_friction_without_mu) {
  HeadOnContact c;
  c.vel_j = {0, 1, 0};
  c.compute(make_params(1000.0, 1000.0, 0.0));
  CHECK_VEC3_NEAR(c.ft, (Vec3d{0, 0, 0}), tol);
  CHECK_VEC3_NEAR(c.f_i, (Vec3d{-100, 0, 0}), tol);
}

UNIT_TEST(hooke_newton_third_law) {
  HeadOnContact c;
  c.vel_i = {0.3, -0.2, 0.1};
  c.vel_j = {-0.1, 0.4, 0.2};
  const ContactParams cp = make_params(1000.0, 500.0, 0.3, 0.2);
  HeadOnContact s = c.swapped();
  c.compute(cp);
  s.compute(cp);
  CHECK_VEC3_NEAR(s.f_i, -1.0 * c.f_i, tol);  // f_j = -f_i
  CHECK_VEC3_NEAR(s.ft, -1.0 * c.ft, tol);
}

UNIT_TEST(hooke_force_is_linear_in_overlap) {
  HeadOnContact c1, c2;
  c1.dn = -0.01;
  c2.dn = -0.02;
  c1.compute(make_params(1000.0));
  c2.compute(make_params(1000.0));
  CHECK_NEAR(c2.f_i.x, 2.0 * c1.f_i.x, tol);
}

// ============================================================
// === Hooke + cohesion
// ============================================================

UNIT_TEST(cohesive_before_contact_is_attractive) {
  ContactParams cp = make_params(1000.0);
  cp.fc_ = 2.0;
  cp.dncut_ = 0.1;

  HeadOnContact c;
  c.dn = 0.05;  // gap of half dncut
  c.ft = {1, 1, 1};
  c.compute<CohesiveLawType::Cohesive>(cp);
  CHECK_VEC3_NEAR(c.f_i, (Vec3d{1, 0, 0}), tol);  // fc/dncut * dn - fc = -1, i is attracted toward j
  CHECK_VEC3_NEAR(c.ft, (Vec3d{0, 0, 0}), tol);   // no friction history without contact

  HeadOnContact at_contact;
  at_contact.dn = 0.0;
  at_contact.compute<CohesiveLawType::Cohesive>(cp);
  CHECK_VEC3_NEAR(at_contact.f_i, (Vec3d{2, 0, 0}), tol);  // fc at dn = 0

  HeadOnContact at_cut;
  at_cut.dn = 0.1;
  at_cut.compute<CohesiveLawType::Cohesive>(cp);
  CHECK_VEC3_NEAR(at_cut.f_i, (Vec3d{0, 0, 0}), tol);  // vanishes at dncut
}

UNIT_TEST(cohesive_in_contact) {
  ContactParams cp = make_params(1000.0);
  cp.fc_ = 2.0;
  cp.dncut_ = 0.1;
  HeadOnContact c;
  c.compute<CohesiveLawType::Cohesive>(cp);
  CHECK_VEC3_NEAR(c.f_i, (Vec3d{-98, 0, 0}), tol);  // kn * |dn| - fc
}

UNIT_TEST(cohesive_coulomb_limit) {
  ContactParams cp = make_params(1000.0, 1000.0, 0.5);
  cp.fc_ = 2.0;
  cp.dncut_ = 0.1;
  HeadOnContact c;
  c.vel_j = {0, 1, 0};
  for (int it = 0; it < 100; it++) {
    c.compute<CohesiveLawType::Cohesive>(cp);
  }
  CHECK_VEC3_NEAR(c.ft, (Vec3d{0, 50, 0}), tol);  // mu * (fn + fc) = 0.5 * (98 + 2)
}

// ============================================================
// === Hooke + DMT adhesion
// ============================================================

UNIT_TEST(dmt_adhesion) {
  ContactParams cp = make_params(1000.0);
  cp.gamma_ = 0.1;
  HeadOnContact c;
  c.compute<CohesiveLawType::DMT>(cp);
  CHECK_VEC3_NEAR(c.f_i, (Vec3d{-(100 - 2 * M_PI * 0.5 * 0.1), 0, 0}), tol);  // repulsion - 2 pi reff gamma

  ContactParams no_gamma = make_params(1000.0);
  HeadOnContact d;
  d.compute<CohesiveLawType::DMT>(no_gamma);
  CHECK_VEC3_NEAR(d.f_i, (Vec3d{-100, 0, 0}), tol);  // same as Hooke without gamma
}

UNIT_TEST_MAIN()
