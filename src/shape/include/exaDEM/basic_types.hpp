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
#include <onika/math/quaternion_operators.h>
#include <onika/math/basic_types.h>
#include <onika/math/basic_types_operators.h>
#include <exaDEM/type/OBB.hpp>
#include <exaDEM/type/OBBtree.hpp>

namespace exaDEM {
using exanb::Vec3d;
using exanb::Quaternion;
ONIKA_HOST_DEVICE_FUNC inline vec3r conv_to_vec3r(const exanb::Vec3d& v) {
  return vec3r{v.x, v.y, v.z};
}

ONIKA_HOST_DEVICE_FUNC inline vec3r conv_to_vec3r(exanb::Vec3d& v) {
  return vec3r{v.x, v.y, v.z};
}

ONIKA_HOST_DEVICE_FUNC inline exanb::Vec3d conv_to_Vec3d(vec3r& v) {
  return exanb::Vec3d{v[0], v[1], v[2]};
}

ONIKA_HOST_DEVICE_FUNC inline quat conv_to_quat(const exanb::Quaternion& Q) {
  return quat{vec3r{Q.x, Q.y, Q.z}, Q.w};
}

ONIKA_HOST_DEVICE_FUNC inline OBB compute_obb(const OBB& in_obb, const exanb::Vec3d& in_pos,
                                              const exanb::Quaternion& in_q, const double homothety) {
  OBB obb = in_obb;
  auto p = conv_to_vec3r(in_pos);
  auto q = conv_to_quat(in_q);
  obb.rotate(q);
  if (homothety != 1.0) {
    obb.extent *= homothety;
    obb.center *= homothety;
  }
  obb.center += p;
  return obb;
}
}  // namespace exaDEM
