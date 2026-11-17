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
#include <vector>

#include <onika/math/basic_types_operators.h>

#include <exaDEM/classifier/classifier_compare.hpp>

#include "unit_test.hpp"

using exaDEM::Classifier;
using exaDEM::ClassifierCompareOptions;
using exaDEM::InnerBondInteraction;
using exaDEM::Interaction;
using exaDEM::InteractionPair;
using exaDEM::InteractionType;
using exaDEM::InteractionTypeId;
using exaDEM::PlaceholderInteraction;
using exanb::Vec3d;
namespace attr = exaDEM::attr;

namespace {
constexpr int VF = InteractionTypeId::VertexFace;
constexpr int VS = InteractionTypeId::VertexSurface;
constexpr int IB = InteractionTypeId::InnerBondId;

PlaceholderInteraction make_contact(int type, uint64_t k) {
  Interaction I;
  I.pair_.pi_ = {k, uint32_t(10 + k % 3), uint16_t(k), uint32_t(k % 4)};
  I.pair_.pj_ = {k + 100, uint32_t(20 + k % 2), uint16_t(k + 1), uint32_t(k % 5)};
  I.pair_.type_ = type;
  I.pair_.swap_ = k % 2;
  I.reset();
  I[attr::friction] = Vec3d{double(k), 1.0, -2.0};
  I[attr::moment] = Vec3d{0.5, double(k), 0};
  PlaceholderInteraction p;
  p = I;
  return p;
}

PlaceholderInteraction make_bond(uint64_t k) {
  InnerBondInteraction I;
  I.pair_.pi_ = {k, 0, uint16_t(k), uint32_t(k)};
  I.pair_.pj_ = {k + 100, 0, uint16_t(k + 1), 0};
  I.pair_.type_ = IB;
  I[attr::en] = 1.0 + k;
  I[attr::dn0] = -0.01 * k;
  PlaceholderInteraction p;
  p = I;
  return p;
}

template <InteractionType IT>
void fill(Classifier& c, int type, std::vector<PlaceholderInteraction> v) {
  c.get_data<IT>(type).insert(v, type);
}

// Reference: 20 VF contacts, 5 contacts with a surface, 4 inner bonds.
struct Lists {
  std::vector<PlaceholderInteraction> vf, vs, ib;
  Lists() {
    for (uint64_t k = 0; k < 20; k++) vf.push_back(make_contact(VF, k));
    for (uint64_t k = 0; k < 5; k++) vs.push_back(make_contact(VS, k));
    for (uint64_t k = 0; k < 4; k++) ib.push_back(make_bond(k));
  }
  void fill_classifier(Classifier& c) const {
    fill<InteractionType::ParticleParticle>(c, VF, vf);
    fill<InteractionType::ParticleDriver>(c, VS, vs);
    fill<InteractionType::InnerBond>(c, IB, ib);
  }
};

const exaDEM::ClassifierTypeDiff& diff_of(const exaDEM::ClassifierDiff& d, int type) { return d.types_[type]; }
}  // namespace

UNIT_TEST(compare_equal_classifiers_in_another_order) {
  Lists ref;
  Lists test = ref;
  std::reverse(test.vf.begin(), test.vf.end());
  std::rotate(test.vs.begin(), test.vs.begin() + 2, test.vs.end());
  std::swap(test.ib[0], test.ib[3]);
  Classifier a, b;
  ref.fill_classifier(a);
  test.fill_classifier(b);
  auto d = exaDEM::compare_classifiers(a, b);
  CHECK(d.ok());
  CHECK(diff_of(d, VF).n_ref_ == 20);
  CHECK(diff_of(d, VF).n_test_ == 20);
  CHECK(diff_of(d, VS).n_ref_ == 5);
  CHECK(diff_of(d, IB).n_ref_ == 4);
  CHECK(d.report().find("[DIFFERENT]") == std::string::npos);
}

UNIT_TEST(compare_empty_classifiers) {
  Classifier a, b;
  auto d = exaDEM::compare_classifiers(a, b);
  CHECK(d.ok());
  CHECK(d.report().empty());
}

UNIT_TEST(compare_missing_and_extra_interactions) {
  Lists ref;
  Lists test = ref;
  test.vf.erase(test.vf.begin() + 7);       // missing in test
  test.vs.push_back(make_contact(VS, 42));  // extra in test
  Classifier a, b;
  ref.fill_classifier(a);
  test.fill_classifier(b);
  auto d = exaDEM::compare_classifiers(a, b);
  CHECK(!d.ok());
  CHECK(diff_of(d, VF).n_only_ref_ == 1);
  CHECK(diff_of(d, VF).n_only_test_ == 0);
  CHECK(diff_of(d, VF).only_ref_.size() == 1);
  CHECK(diff_of(d, VF).only_ref_[0] == ref.vf[7]);
  CHECK(diff_of(d, VS).n_only_test_ == 1);
  CHECK(diff_of(d, VS).only_test_[0].pair_.pi_.id_ == 42);
  CHECK(diff_of(d, IB).ok());
  CHECK(d.report().find("only in ref") != std::string::npos);
}

UNIT_TEST(compare_detects_any_pair_field) {
  // every field of the pair identifies an interaction: a change gives one missing + one extra interaction
  auto check_change = [](auto&& change) {
    Lists ref;
    Lists test = ref;
    change(test.vf[3].pair_);
    Classifier a, b;
    ref.fill_classifier(a);
    test.fill_classifier(b);
    auto d = exaDEM::compare_classifiers(a, b);
    CHECK(diff_of(d, VF).n_only_ref_ == 1);
    CHECK(diff_of(d, VF).n_only_test_ == 1);
  };
  check_change([](InteractionPair& p) { p.ghost_ = InteractionPair::OwnerGhost; });
  check_change([](InteractionPair& p) { p.swap_ = !p.swap_; });
  check_change([](InteractionPair& p) { p.pi_.sub_ += 1; });
  check_change([](InteractionPair& p) { p.pj_.sub_ += 1; });
  check_change([](InteractionPair& p) { p.pi_.cell_ += 1; });
  check_change([](InteractionPair& p) { p.pj_.p_ += 1; });
  check_change([](InteractionPair& p) { p.pj_.id_ += 1000; });
}

UNIT_TEST(compare_values_with_tolerance) {
  auto compare_with_friction_delta = [](double delta, ClassifierCompareOptions opt = {}) {
    Lists ref;
    Lists test = ref;
    test.vf[5].as<Interaction>()[attr::friction].y += delta;
    Classifier a, b;
    ref.fill_classifier(a);
    test.fill_classifier(b);
    return exaDEM::compare_classifiers(a, b, opt);
  };
  CHECK(compare_with_friction_delta(1e-14).ok());  // below atol + rtol * |1|

  auto d = compare_with_friction_delta(1e-6);
  CHECK(!d.ok());
  CHECK(diff_of(d, VF).n_only_ref_ == 0);
  CHECK(diff_of(d, VF).n_value_mismatch_ == 1);
  CHECK(diff_of(d, VF).mismatch_fields_[0] == "friction");

  ClassifierCompareOptions loose;
  loose.atol_ = 1e-3;
  CHECK(compare_with_friction_delta(1e-6, loose).ok());

  ClassifierCompareOptions lists_only;
  lists_only.compare_values_ = false;
  CHECK(compare_with_friction_delta(1.0, lists_only).ok());
}

UNIT_TEST(compare_values_names_the_field) {
  auto first_field = [](auto&& change, int type) {
    Lists ref;
    Lists test = ref;
    change(test);
    Classifier a, b;
    ref.fill_classifier(a);
    test.fill_classifier(b);
    auto d = exaDEM::compare_classifiers(a, b);
    const auto& td = diff_of(d, type);
    return td.mismatch_fields_.empty() ? std::string() : td.mismatch_fields_[0];
  };
  CHECK(first_field([](Lists& l) { l.vs[1].as<Interaction>()[attr::moment].z = 3; }, VS) == "moment");
  CHECK(first_field([](Lists& l) { l.vf[0].as<Interaction>()[attr::weight] = 0.5; }, VF) == "weight");
  CHECK(first_field([](Lists& l) { l.ib[2].as<InnerBondInteraction>()[attr::unbroken] = false; }, IB) == "unbroken");
  CHECK(first_field([](Lists& l) { l.ib[1].as<InnerBondInteraction>()[attr::dn0] = 1.0; }, IB) == "dn0");
  CHECK(first_field(
            [](Lists& l) {
              l.ib[1].as<InnerBondInteraction>()[attr::criterion].mode_ = exaDEM::RuptureMode::EnergyMixedMode;
            },
            IB) == "criterion");
}

UNIT_TEST(compare_detects_duplicates) {
  Lists ref;
  Lists test = ref;
  test.vf.push_back(test.vf[4]);  // stored twice
  Classifier a, b;
  ref.fill_classifier(a);
  test.fill_classifier(b);
  auto d = exaDEM::compare_classifiers(a, b);
  CHECK(!d.ok());
  CHECK(diff_of(d, VF).n_duplicates_ref_ == 0);
  CHECK(diff_of(d, VF).n_duplicates_test_ == 1);
  CHECK(diff_of(d, VF).n_only_test_ == 1);  // multiset semantics: the second copy is unmatched
}

UNIT_TEST(compare_limits_the_details) {
  Lists ref;
  Lists test;
  test.vf.clear();  // 20 interactions missing
  Classifier a, b;
  ref.fill_classifier(a);
  test.fill_classifier(b);
  ClassifierCompareOptions opt;
  opt.max_details_ = 3;
  auto d = exaDEM::compare_classifiers(a, b, opt);
  CHECK(diff_of(d, VF).n_only_ref_ == 20);
  CHECK(diff_of(d, VF).only_ref_.size() == 3);
}

UNIT_TEST(extract_sorted_interactions_is_canonical) {
  Lists l;
  Classifier a, b;
  l.fill_classifier(a);
  std::reverse(l.vf.begin(), l.vf.end());
  l.fill_classifier(b);
  auto va = exaDEM::extract_sorted_interactions(a, VF);
  auto vb = exaDEM::extract_sorted_interactions(b, VF);
  CHECK(va.size() == 20);
  bool same = va.size() == vb.size();
  for (size_t i = 0; same && i < va.size(); i++) {
    same = (va[i] == vb[i]) && va[i].as<Interaction>()[attr::friction] == vb[i].as<Interaction>()[attr::friction];
  }
  CHECK(same);
  CHECK(exaDEM::extract_sorted_interactions(a, InteractionTypeId::EdgeEdge).empty());
}

UNIT_TEST_MAIN()
