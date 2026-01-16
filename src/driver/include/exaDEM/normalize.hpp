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

namespace exanb {
/**
 * @brief Calculate the length of a 3D vector.
 * @param v The input vector.
 * @return The length of the vector.
 */
inline double length(Vec3d& v) {
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

/**
 * @brief Calculate the length of a const 3D vector.
 * @param v The input vector.
 * @return The length of the vector.
 */
inline double length(const Vec3d& v) {
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

/**
 * @brief Normalize a 3D vector.
 * @param v The input vector to be normalized.
 */
inline void _normalize(Vec3d& v) {
  v = v / exanb::norm(v);
}
}  // namespace exanb
