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
#include <cmath>
#include <stdexcept>  // contact_force.hpp uses std::logic_error
#include <utility>

#include <onika/math/basic_types_operators.h>

#include <exaDEM/forcefield/contact_force.hpp>

#include "unit_test.hpp"

using exaDEM::HertzParams;
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
  void compute(const HertzParams& hp) {
    exaDEM::contact_force_core<ContactLawType::Hertz, CohesiveLaw>(
        dn, n, dt, hp, meff, reff, ft, contact_position, pos_i, vel_i, f_i, mom_i, vrot_i, pos_j, vel_j, vrot_j);
  }
};

HertzParams make_params(double estar, double kt = 0, double mu = 0, double damp_rate = 0) {
  HertzParams hp;
  hp.estar_ = estar;
  hp.kt_ = kt;
  hp.mu_ = mu;
  hp.damp_rate_ = damp_rate;
  return hp;
}

// elastic Hertz force: 4/3 E* sqrt(R*) delta^(3/2)
double hertz_force(double estar, double reff, double delta) {
  return (4.0 / 3.0) * estar * std::sqrt(reff) * std::pow(delta, 1.5);
}
}  // namespace

UNIT_TEST(static_overlap_is_repulsive) {
  HeadOnContact c;
  c.compute(make_params(1000.0));
  CHECK_VEC3_NEAR(c.f_i, (Vec3d{-hertz_force(1000.0, 0.5, 0.1), 0, 0}), tol);  // pushes i away from j
  CHECK_VEC3_NEAR(c.ft, (Vec3d{0, 0, 0}), tol);
  CHECK_VEC3_NEAR(c.mom_i, (Vec3d{0, 0, 0}), tol);
}

UNIT_TEST(force_overwrites_f_i) {
  HeadOnContact c;
  c.f_i = {1e6, 1e6, 1e6};
  c.compute(make_params(1000.0));
  CHECK_VEC3_NEAR(c.f_i, (Vec3d{-hertz_force(1000.0, 0.5, 0.1), 0, 0}), tol);
}

UNIT_TEST(force_scales_with_overlap_power_three_halves) {
  HeadOnContact c1, c2;
  c1.dn = -0.01;
  c2.dn = -0.04;
  c1.compute(make_params(1000.0));
  c2.compute(make_params(1000.0));
  CHECK_NEAR(c2.f_i.x, 8.0 * c1.f_i.x, tol);  // 4^(3/2) = 8
}

UNIT_TEST(force_scales_with_estar_and_sqrt_reff) {
  HeadOnContact c1, c2;
  c2.reff = 2.0;  // R* is 4 times bigger
  c1.compute(make_params(1000.0));
  c2.compute(make_params(3000.0));
  CHECK_NEAR(c2.f_i.x, 3.0 * 2.0 * c1.f_i.x, tol);
}

UNIT_TEST(no_force_without_overlap) {
  HeadOnContact c;
  c.dn = 0.1;
  c.compute(make_params(1000.0));
  CHECK_VEC3_NEAR(c.f_i, (Vec3d{0, 0, 0}), tol);
}

UNIT_TEST(damping) {
  const double fe = hertz_force(1000.0, 0.5, 0.1);
  const double k = 2.0 * 1000.0 * std::sqrt(0.5 * 0.1);  // tangent stiffness dFe/d(delta)
  const double damp = 0.5 * 2.0 * std::sqrt(k * 0.5);

  HeadOnContact approaching;
  approaching.vel_i = {0.01, 0, 0};
  approaching.compute(make_params(1000.0, 0, 0, 0.5));
  CHECK_NEAR(approaching.f_i.x, -(fe + damp * 0.01), tol);  // damping increases the repulsion

  HeadOnContact separating;
  separating.vel_i = {-0.01, 0, 0};
  separating.compute(make_params(1000.0, 0, 0, 0.5));
  CHECK_NEAR(separating.f_i.x, -(fe - damp * 0.01), tol);  // and decreases it while separating

  HeadOnContact fast_separating;
  fast_separating.vel_i = {-10, 0, 0};
  fast_separating.compute(make_params(1000.0, 0, 0, 0.5));
  CHECK_VEC3_NEAR(fast_separating.f_i, (Vec3d{0, 0, 0}), tol);  // never attractive

  HeadOnContact undamped;
  undamped.vel_i = {1, 0, 0};
  undamped.compute(make_params(1000.0, 0, 0, 0.0));
  CHECK_NEAR(undamped.f_i.x, -fe, tol);  // no damping without damp_rate
}

UNIT_TEST(friction_follows_relative_velocity) {
  // j slides along +y: friction drags i along +y
  HeadOnContact c;
  c.vel_j = {0, 1, 0};
  c.compute(make_params(1000.0, 1000.0, 0.5));
  CHECK_VEC3_NEAR(c.ft, (Vec3d{0, 1, 0}), tol);  // kt * dt * vt
}

UNIT_TEST(friction_coulomb_limit) {
  const double fe = hertz_force(1000.0, 0.5, 0.1);
  HeadOnContact c;
  c.vel_j = {0, 1, 0};
  const HertzParams hp = make_params(1000.0, 1000.0, 0.5);
  for (int it = 0; it < 1000; it++) {
    c.compute(hp);
  }
  CHECK_VEC3_NEAR(c.ft, (Vec3d{0, 0.5 * fe, 0}), tol);  // sliding: |ft| = mu * Fe
  CHECK_VEC3_NEAR(c.f_i, (Vec3d{-fe, 0.5 * fe, 0}), tol);
}

UNIT_TEST(no_friction_without_mu) {
  HeadOnContact c;
  c.vel_j = {0, 1, 0};
  c.compute(make_params(1000.0, 1000.0, 0.0));
  CHECK_VEC3_NEAR(c.ft, (Vec3d{0, 0, 0}), tol);
}

UNIT_TEST(newton_third_law) {
  HeadOnContact c;
  c.vel_i = {0.003, -0.002, 0.001};
  c.vel_j = {-0.001, 0.004, 0.002};
  const HertzParams hp = make_params(1000.0, 500.0, 0.3, 0.2);
  HeadOnContact s = c.swapped();
  c.compute(hp);
  s.compute(hp);
  CHECK_VEC3_NEAR(s.f_i, -1.0 * c.f_i, tol);  // f_j = -f_i
  CHECK_VEC3_NEAR(s.ft, -1.0 * c.ft, tol);
}

UNIT_TEST(cohesive_before_contact_is_attractive) {
  HertzParams hp = make_params(1000.0);
  hp.fc_ = 2.0;
  hp.dncut_ = 0.3;
  HeadOnContact c;
  c.dn = 0.15;  // gap of 0.15 < dncut
  c.compute<CohesiveLawType::Cohesive>(hp);
  CHECK_VEC3_NEAR(c.f_i, (Vec3d{(2.0 / 0.3) * 0.15 * -1.0 - 2.0 * -1.0, 0, 0}), tol);  // (fc/dncut) dn - fc, along n
}

UNIT_TEST(cohesive_in_contact) {
  HertzParams hp = make_params(1000.0);
  hp.fc_ = 2.0;
  hp.dncut_ = 0.3;
  HeadOnContact c;
  c.compute<CohesiveLawType::Cohesive>(hp);
  CHECK_VEC3_NEAR(c.f_i, (Vec3d{-(hertz_force(1000.0, 0.5, 0.1) - 2.0), 0, 0}), tol);  // Hertz - fc
}

UNIT_TEST(dmt_adhesion) {
  HertzParams hp = make_params(1000.0);
  hp.gamma_ = 0.1;
  HeadOnContact c;
  c.compute<CohesiveLawType::DMT>(hp);
  CHECK_VEC3_NEAR(c.f_i, (Vec3d{-(hertz_force(1000.0, 0.5, 0.1) - 2 * M_PI * 0.5 * 0.1), 0, 0}), tol);

  HertzParams no_gamma = make_params(1000.0);
  HeadOnContact d;
  d.compute<CohesiveLawType::DMT>(no_gamma);
  CHECK_VEC3_NEAR(d.f_i, (Vec3d{-hertz_force(1000.0, 0.5, 0.1), 0, 0}), tol);  // same as without gamma
}

UNIT_TEST(reduced_young_modulus) {
  // two identical materials: E* = E / (2 (1 - nu^2))
  CHECK_NEAR(exaDEM::compute_reduced_young_modulus(70.0e9, 0.33, 70.0e9, 0.33), 70.0e9 / (2.0 * (1.0 - 0.33 * 0.33)),
             1e-3);
  // symmetric
  CHECK_NEAR(exaDEM::compute_reduced_young_modulus(63.0e9, 0.23, 210.0e9, 0.30),
             exaDEM::compute_reduced_young_modulus(210.0e9, 0.30, 63.0e9, 0.23), 1e-3);
  // rigid second body: E* = E / (1 - nu^2)
  CHECK_NEAR(exaDEM::compute_reduced_young_modulus(70.0e9, 0.33, 1.0e30, 0.0), 70.0e9 / (1.0 - 0.33 * 0.33), 1e-3);
}

UNIT_TEST_MAIN()
