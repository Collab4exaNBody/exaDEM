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
enum RuptureMode { EnergyMixedMode, EnergySeparateModes, StressEnergySeparateMode, None };

/**
 * @brief Returns a human-readable name for a RuptureMode value.
 */
inline std::string display(RuptureMode mode) {
  switch (mode) {
    case RuptureMode::EnergyMixedMode:
      return "EnergyMixedMode";
    case RuptureMode::EnergySeparateModes:
      return "EnergySeparateModes";
    case RuptureMode::StressEnergySeparateMode:
      return "StressEnergySeparateMode";
    default:
      return "None";
  }
}

struct RuptureCriteria {
  double criteria_1 = 0;             /// stores the normal+tangential rupture criterion (EnergyMixedMode and StressEnergySeparateMode) or the normal rupture criterion (EnergySeparateModes)
  double criteria_2 = 0;             /// stores the tangential rupture criterion (EnergySeparateModes only) or the stress rupture criterion (StressEnergySeparateMode only)
  RuptureMode mode = RuptureMode::None; 

  ONIKA_HOST_DEVICE_FUNC inline double& energy_criterion() {
    assert(mode == RuptureMode::EnergyMixedMode||mode == RuptureMode::StressEnergySeparateMode);
    return criteria_1;
  }

  ONIKA_HOST_DEVICE_FUNC inline double energy_criterion() const {
    assert(mode == RuptureMode::EnergyMixedMode||mode == RuptureMode::StressEnergySeparateMode);
    return criteria_1;
  }

  ONIKA_HOST_DEVICE_FUNC inline double& energy_normal_criterion() {
    assert(mode == RuptureMode::EnergySeparateModes);
    return criteria_1;
  }

  ONIKA_HOST_DEVICE_FUNC inline double energy_normal_criterion() const {
    assert(mode == RuptureMode::EnergySeparateModes);
    return criteria_1;
  }

  ONIKA_HOST_DEVICE_FUNC inline double& energy_tangential_criterion() {
    assert(mode == RuptureMode::EnergySeparateModes);
    return criteria_2;
  }

  ONIKA_HOST_DEVICE_FUNC inline double energy_tangential_criterion() const {
    assert(mode == RuptureMode::EnergySeparateModes);
    return criteria_2;
  }
};

inline std::ostream& operator<<(std::ostream& out, const RuptureCriteria& c) {
  out << "{ mode: " << display(c.mode) << ", criteria_1: " << c.criteria_1 << ", criteria_2: " << c.criteria_2
      << " }";
  return out;
}
}  // namespace exaDEM
