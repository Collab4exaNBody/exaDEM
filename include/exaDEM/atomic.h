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

#include <onika/cuda/cuda.h>
#include <onika/math/basic_types.h>

namespace exaDEM {
ONIKA_HOST_DEVICE_FUNC
static void lockAndAdd(exanb::Vec3d& val, const exanb::Vec3d& add) {
  ONIKA_CU_ATOMIC_ADD(val.x, add.x);
  ONIKA_CU_ATOMIC_ADD(val.y, add.y);
  ONIKA_CU_ATOMIC_ADD(val.z, add.z);
}

ONIKA_HOST_DEVICE_FUNC
static inline void lockAndAdd(exanb::Vec3d& val, const exanb::Vec3d&& add) {
  ONIKA_CU_ATOMIC_ADD(val.x, add.x);
  ONIKA_CU_ATOMIC_ADD(val.y, add.y);
  ONIKA_CU_ATOMIC_ADD(val.z, add.z);
}

ONIKA_HOST_DEVICE_FUNC
static inline void mat3d_atomic_add_contribution(exanb::Mat3d& dst, const exanb::Mat3d& src) {
  ONIKA_CU_ATOMIC_ADD(dst.m11, src.m11);
  ONIKA_CU_ATOMIC_ADD(dst.m12, src.m12);
  ONIKA_CU_ATOMIC_ADD(dst.m13, src.m13);
  ONIKA_CU_ATOMIC_ADD(dst.m21, src.m21);
  ONIKA_CU_ATOMIC_ADD(dst.m22, src.m22);
  ONIKA_CU_ATOMIC_ADD(dst.m23, src.m23);
  ONIKA_CU_ATOMIC_ADD(dst.m31, src.m31);
  ONIKA_CU_ATOMIC_ADD(dst.m32, src.m32);
  ONIKA_CU_ATOMIC_ADD(dst.m33, src.m33);
}

ONIKA_HOST_DEVICE_FUNC
static inline void mat3d_atomic_add_block_contribution(exanb::Mat3d& dst, const exanb::Mat3d& src) {
  ONIKA_CU_BLOCK_ATOMIC_ADD(dst.m11, src.m11);
  ONIKA_CU_BLOCK_ATOMIC_ADD(dst.m12, src.m12);
  ONIKA_CU_BLOCK_ATOMIC_ADD(dst.m13, src.m13);
  ONIKA_CU_BLOCK_ATOMIC_ADD(dst.m21, src.m21);
  ONIKA_CU_BLOCK_ATOMIC_ADD(dst.m22, src.m22);
  ONIKA_CU_BLOCK_ATOMIC_ADD(dst.m23, src.m23);
  ONIKA_CU_BLOCK_ATOMIC_ADD(dst.m31, src.m31);
  ONIKA_CU_BLOCK_ATOMIC_ADD(dst.m32, src.m32);
  ONIKA_CU_BLOCK_ATOMIC_ADD(dst.m33, src.m33);
}
}  // namespace exaDEM
