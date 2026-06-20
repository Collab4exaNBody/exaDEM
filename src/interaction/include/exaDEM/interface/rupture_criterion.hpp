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

#include <cassert>
#include <iostream>
#include <string>

namespace exaDEM {
enum RuptureMode { MixedMode, SeparateModes, None };

/**
 * @brief Returns a human-readable name for a RuptureMode value.
 */
inline std::string display(RuptureMode mode) {
  switch (mode) {
    case RuptureMode::MixedMode:
      return "MixedMode";
    case RuptureMode::SeparateModes:
      return "SeparateModes";
    default:
      return "None";
  }
}

struct RuptureCriteria {
  double criteria_1 = 0;             /// stores the normal+tangential rupture criterion (MixedMode) or the normal rupture criterion (SeparateModes)
  double criteria_2 = 0;             /// stores the tangential rupture criterion (SeparateModes only)
  RuptureMode mode = RuptureMode::None;

  ONIKA_HOST_DEVICE_FUNC inline double& criterion() {
    assert(mode == RuptureMode::MixedMode);
    return criteria_1;
  }

  ONIKA_HOST_DEVICE_FUNC inline double criterion() const {
    assert(mode == RuptureMode::MixedMode);
    return criteria_1;
  }

  ONIKA_HOST_DEVICE_FUNC inline double& normal_criterion() {
    assert(mode == RuptureMode::SeparateModes);
    return criteria_1;
  }

  ONIKA_HOST_DEVICE_FUNC inline double normal_criterion() const {
    assert(mode == RuptureMode::SeparateModes);
    return criteria_1;
  }

  ONIKA_HOST_DEVICE_FUNC inline double& tangential_criterion() {
    assert(mode == RuptureMode::SeparateModes);
    return criteria_2;
  }

  ONIKA_HOST_DEVICE_FUNC inline double tangential_criterion() const {
    assert(mode == RuptureMode::SeparateModes);
    return criteria_2;
  }
};

inline std::ostream& operator<<(std::ostream& out, const RuptureCriteria& c) {
  out << "{ mode: " << display(c.mode) << ", criteria_1: " << c.criteria_1 << ", criteria_2: " << c.criteria_2
      << " }";
  return out;
}
}  // namespace exaDEM
