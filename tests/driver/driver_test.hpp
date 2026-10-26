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

// Driver headers are not self-contained: they are included in the same order as in the driver operators.
#include <exanb/core/grid.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <onika/scg/operator.h>

#include <exaDEM/drivers.hpp>

#include <sstream>
#include <string>

#include "unit_test.hpp"

namespace exaDEM::unit_test {
// Unit vector check.
inline bool is_unit(const exanb::Vec3d& v, double tol = 1e-12) { return std::fabs(exanb::norm(v) - 1.0) <= tol; }

inline bool is_finite(const exanb::Vec3d& v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); }

inline bool contains(const std::string& str, const std::string& pattern) {
  return str.find(pattern) != std::string::npos;
}
}  // namespace exaDEM::unit_test
