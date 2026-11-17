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

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <exaDEM/classifier/classifier.hpp>

/**
 * @file classifier_compare.hpp
 * @brief Order-independent comparison of two classifiers: interaction lists, then specialized fields with a tolerance.
 */

namespace exaDEM {

/** @brief Comparison options. */
struct ClassifierCompareOptions {
  double atol_ = 1e-12;         ///< absolute tolerance on the specialized fields
  double rtol_ = 1e-9;          ///< relative tolerance on the specialized fields
  bool compare_values_ = true;  ///< compare the specialized fields, otherwise only the interaction lists
  size_t max_details_ = 10;     ///< maximum number of interactions stored per category and per type
};

/** @brief Differences found for one interaction type. */
struct ClassifierTypeDiff {
  int type_ = InteractionTypeId::Undefined;
  size_t n_ref_ = 0;              ///< number of interactions in the reference
  size_t n_test_ = 0;             ///< number of interactions in the tested classifier
  size_t n_only_ref_ = 0;         ///< interactions missing from the tested classifier
  size_t n_only_test_ = 0;        ///< interactions missing from the reference
  size_t n_value_mismatch_ = 0;   ///< matched interactions with different specialized fields
  size_t n_duplicates_ref_ = 0;   ///< identical interactions stored several times in the reference
  size_t n_duplicates_test_ = 0;  ///< identical interactions stored several times in the tested classifier
  std::vector<PlaceholderInteraction> only_ref_;  ///< first n_only_ref_ (up to max_details_)
  std::vector<PlaceholderInteraction> only_test_;
  std::vector<std::pair<PlaceholderInteraction, PlaceholderInteraction>> value_mismatch_;  ///< (ref, test)
  std::vector<std::string> mismatch_fields_;  ///< name of the first differing field of each value_mismatch_ entry

  bool ok() const {
    return n_only_ref_ == 0 && n_only_test_ == 0 && n_value_mismatch_ == 0 && n_duplicates_ref_ == 0 &&
           n_duplicates_test_ == 0;
  }
};

/** @brief Differences found for every interaction type. */
struct ClassifierDiff {
  std::array<ClassifierTypeDiff, InteractionTypeId::NTypes> types_;

  bool ok() const {
    return std::all_of(types_.begin(), types_.end(), [](const ClassifierTypeDiff& d) { return d.ok(); });
  }

  /** @brief Human-readable report: one line per non-empty type, then the stored details of the failing ones. */
  std::string report() const {
    std::ostringstream out;
    for (const auto& d : types_) {
      if (d.n_ref_ == 0 && d.n_test_ == 0) {
        continue;
      }
      out << "type " << d.type_ << ": ref " << d.n_ref_ << ", test " << d.n_test_ << ", only in ref " << d.n_only_ref_
          << ", only in test " << d.n_only_test_ << ", value mismatches " << d.n_value_mismatch_ << ", duplicates "
          << d.n_duplicates_ref_ << "/" << d.n_duplicates_test_ << (d.ok() ? " [OK]" : " [DIFFERENT]") << "\n";
    }
    for (const auto& d : types_) {
      auto print = [&out](const char* what, const PlaceholderInteraction& I) {
        const auto& o = I.pair_.owner();
        const auto& p = I.pair_.partner();
        out << "    " << what << " type " << I.pair_.type_ << " owner {id " << o.id_ << ", cell " << o.cell_ << ", p "
            << o.p_ << ", sub " << o.sub_ << "} partner {id " << p.id_ << ", cell " << p.cell_ << ", p " << p.p_
            << ", sub " << p.sub_ << "} swap " << int(I.pair_.swap_) << " ghost " << int(I.pair_.ghost_) << "\n";
      };
      for (const auto& I : d.only_ref_) {
        print("only in ref :", I);
      }
      for (const auto& I : d.only_test_) {
        print("only in test:", I);
      }
      auto print_fields = [&out](const char* what, const PlaceholderInteraction& I) {
        out << "      " << what;
        if (I.type() == InteractionTypeId::InnerBondId) {
          const auto& b = I.as<InnerBondInteraction>();
          out << " friction " << b[attr::friction] << " en " << b[attr::en] << " tds " << b[attr::tds] << " et "
              << b[attr::et] << " dn0 " << b[attr::dn0] << " weight " << b[attr::weight] << " criterion "
              << b[attr::criterion] << " unbroken " << int(b[attr::unbroken]) << "\n";
        } else {
          const auto& c = I.as<Interaction>();
          out << " friction " << c[attr::friction] << " moment " << c[attr::moment] << " weight " << c[attr::weight]
              << "\n";
        }
      };
      for (size_t i = 0; i < d.value_mismatch_.size(); i++) {
        print(("value of '" + d.mismatch_fields_[i] + "' differs:").c_str(), d.value_mismatch_[i].first);
        print_fields("ref :", d.value_mismatch_[i].first);
        print_fields("test:", d.value_mismatch_[i].second);
      }
    }
    return out.str();
  }
};

namespace classifier_compare_details {

/// @brief Everything that identifies an interaction, i.e. all the InteractionPair fields.
inline auto identity(const PlaceholderInteraction& I) {
  const auto& o = I.pair_.owner();
  const auto& p = I.pair_.partner();
  return std::make_tuple(o.id_, p.id_, I.pair_.type_, I.pair_.swap_, o.sub_, p.sub_, I.pair_.ghost_, o.cell_, o.p_,
                         p.cell_, p.p_);
}

inline bool less(const PlaceholderInteraction& a, const PlaceholderInteraction& b) { return identity(a) < identity(b); }

inline bool same(const PlaceholderInteraction& a, const PlaceholderInteraction& b) {
  return identity(a) == identity(b);
}

/// @brief Compares the specialized fields of two interactions, stores the name of the first one that differs.
struct FieldComparator {
  const ClassifierCompareOptions& opt_;
  std::string first_mismatch_;

  bool close(double a, double b) const { return std::fabs(a - b) <= opt_.atol_ + opt_.rtol_ * std::fabs(a); }
  bool close(const Vec3d& a, const Vec3d& b) const { return close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z); }
  bool close(const RuptureCriteria& a, const RuptureCriteria& b) const {
    return a.mode_ == b.mode_ && close(a.criteria_1_, b.criteria_1_) && close(a.criteria_2_, b.criteria_2_);
  }
  bool close(uint8_t a, uint8_t b) const { return a == b; }

  template <typename T>
  void check(const char* name, const T& a, const T& b) {
    if (first_mismatch_.empty() && !close(a, b)) {
      first_mismatch_ = name;
    }
  }

  bool operator()(const PlaceholderInteraction& ref, const PlaceholderInteraction& test) {
    first_mismatch_.clear();
    if (ref.type() == InteractionTypeId::InnerBondId) {
      const auto& a = ref.as<InnerBondInteraction>();
      const auto& b = test.as<InnerBondInteraction>();
      check("friction", a[attr::friction], b[attr::friction]);
      check("en", a[attr::en], b[attr::en]);
      check("tds", a[attr::tds], b[attr::tds]);
      check("et", a[attr::et], b[attr::et]);
      check("dn0", a[attr::dn0], b[attr::dn0]);
      check("weight", a[attr::weight], b[attr::weight]);
      check("criterion", a[attr::criterion], b[attr::criterion]);
      check("unbroken", a[attr::unbroken], b[attr::unbroken]);
    } else {
      const auto& a = ref.as<Interaction>();
      const auto& b = test.as<Interaction>();
      check("friction", a[attr::friction], b[attr::friction]);
      check("moment", a[attr::moment], b[attr::moment]);
      check("weight", a[attr::weight], b[attr::weight]);
    }
    return first_mismatch_.empty();
  }
};

/// @brief Copies one wave into a vector of PlaceholderInteraction.
struct ExtractFunc {
  std::vector<PlaceholderInteraction>& out_;
  template <InteractionType IT>
  void operator()(ClassifierContainer<IT>& container) {
    out_.resize(container.size());
    for (size_t i = 0; i < container.size(); i++) {
      out_[i] = container[i];
    }
  }
};

/// @brief Counts the interactions of a sorted list that are equal to their predecessor.
inline size_t count_duplicates(const std::vector<PlaceholderInteraction>& sorted) {
  size_t res = 0;
  for (size_t i = 1; i < sorted.size(); i++) {
    res += same(sorted[i - 1], sorted[i]);
  }
  return res;
}
}  // namespace classifier_compare_details

/** @brief Extracts the interactions of one type, sorted with the canonical order used by compare_classifiers. */
inline std::vector<PlaceholderInteraction> extract_sorted_interactions(Classifier& classifier, int type) {
  std::vector<PlaceholderInteraction> res;
  classifier_compare_details::ExtractFunc func = {res};
  CDispatcher::dispatch(type, classifier, func);
  std::sort(res.begin(), res.end(), classifier_compare_details::less);
  return res;
}

/** @brief Compares two sorted lists of interactions of the same type. */
inline ClassifierTypeDiff compare_sorted_interactions(int type, const std::vector<PlaceholderInteraction>& ref,
                                                      const std::vector<PlaceholderInteraction>& test,
                                                      const ClassifierCompareOptions& opt = {}) {
  using namespace classifier_compare_details;
  ClassifierTypeDiff d;
  d.type_ = type;
  d.n_ref_ = ref.size();
  d.n_test_ = test.size();
  d.n_duplicates_ref_ = count_duplicates(ref);
  d.n_duplicates_test_ = count_duplicates(test);
  FieldComparator fields = {opt, {}};

  auto store = [&opt](auto& vec, const auto& value) {
    if (vec.size() < opt.max_details_) {
      vec.push_back(value);
    }
  };

  size_t i = 0, j = 0;
  while (i < ref.size() || j < test.size()) {
    if (j == test.size() || (i < ref.size() && less(ref[i], test[j]))) {
      d.n_only_ref_++;
      store(d.only_ref_, ref[i++]);
    } else if (i == ref.size() || less(test[j], ref[i])) {
      d.n_only_test_++;
      store(d.only_test_, test[j++]);
    } else {
      if (opt.compare_values_ && !fields(ref[i], test[j])) {
        d.n_value_mismatch_++;
        if (d.value_mismatch_.size() < opt.max_details_) {
          d.value_mismatch_.push_back({ref[i], test[j]});
          d.mismatch_fields_.push_back(fields.first_mismatch_);
        }
      }
      i++;
      j++;
    }
  }
  return d;
}

/** @brief Compares two classifiers, independently of the order of the interactions in each wave. */
inline ClassifierDiff compare_classifiers(Classifier& ref, Classifier& test, const ClassifierCompareOptions& opt = {}) {
  ClassifierDiff res;
  for (int type = 0; type < InteractionTypeId::NTypes; type++) {
    res.types_[type] = compare_sorted_interactions(type, extract_sorted_interactions(ref, type),
                                                   extract_sorted_interactions(test, type), opt);
  }
  return res;
}
}  // namespace exaDEM
