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
#include <onika/math/basic_types.h>
#include <onika/math/basic_types_operators.h>
#include <onika/math/quaternion_operators.h>

#include <exaDEM/type/OBB.hpp>
#include <exaDEM/type/OBBtree.hpp>

namespace exaDEM {
// exanb::Quaternion's operator*(Quaternion, Quaternion) is a componentwise product, not the
// Hamilton product, so it cannot be used to compose two rotations. This is the correct
// composition, used to express one particle's orientation relative to another's (OBB tree).
ONIKA_HOST_DEVICE_FUNC inline exanb::Quaternion hamilton_product(const exanb::Quaternion& q1,
                                                                 const exanb::Quaternion& q2) {
  return exanb::Quaternion{q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
                           q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
                           q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
                           q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w};
}

ONIKA_HOST_DEVICE_FUNC inline OBB compute_obb(const OBB& in_obb, const exanb::Vec3d& in_pos,
                                              const exanb::Quaternion& in_q, const double homothety) {
  OBB obb = in_obb;
  obb.rotate_in_place(in_q);
  if (homothety != 1.0) {
    obb.extent *= homothety;
    obb.center *= homothety;
  }
  obb.center += in_pos;
  return obb;
}
}  // namespace exaDEM
