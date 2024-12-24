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

#include <exanb/core/basic_types_def.h>
#include <exanb/core/quaternion.h>
#include <exanb/core/declare_field.h>
#include <onika/oarray.h>

#include <cstdint>
typedef ::onika::oarray_t<::exanb::Vec3d, EXADEM_MAX_VERTICES> VerticesType;

// for rx, use xstamp::field::rx as a field descriptor
XSTAMP_DECLARE_FIELD(uint64_t, id, "particle id");
XSTAMP_DECLARE_FIELD(uint32_t, type, "particle type");
XSTAMP_DECLARE_FIELD(double, rx, "particle position X");
XSTAMP_DECLARE_FIELD(double, ry, "particle position Y");
XSTAMP_DECLARE_FIELD(double, rz, "particle position Z");
XSTAMP_DECLARE_FIELD(double, vx, "particle velocity X");
XSTAMP_DECLARE_FIELD(double, vy, "particle velocity Y");
XSTAMP_DECLARE_FIELD(double, vz, "particle velocity Z");
XSTAMP_DECLARE_FIELD(double, ax, "particle acceleration X");
XSTAMP_DECLARE_FIELD(double, ay, "particle acceleration Y");
XSTAMP_DECLARE_FIELD(double, az, "particle acceleration Z");
// DEM - reuse orient and angmom
XSTAMP_DECLARE_FIELD(double, mass, "particle mass");
XSTAMP_DECLARE_FIELD(double, homothety, "particle shape homothety");
XSTAMP_DECLARE_FIELD(double, radius, "radius");
XSTAMP_DECLARE_FIELD(::exanb::Quaternion, orient, "angular position");
XSTAMP_DECLARE_FIELD(::exanb::Vec3d, mom, "moment");
XSTAMP_DECLARE_FIELD(::exanb::Vec3d, vrot, "angular velocity");     //
XSTAMP_DECLARE_FIELD(::exanb::Vec3d, arot, "angular acceleration"); //
XSTAMP_DECLARE_FIELD(::exanb::Vec3d, inertia, "inertia values (same value in the diagonal)");
XSTAMP_DECLARE_FIELD(VerticesType, vertices, "list to compute vertices"); //
XSTAMP_DECLARE_FIELD(::exanb::Mat3d, stress, "stress tensor"); //

// aliases
XSTAMP_DECLARE_ALIAS(fx, ax)
XSTAMP_DECLARE_ALIAS(fy, ay)
XSTAMP_DECLARE_ALIAS(fz, az)
// XSTAMP_DECLARE_ALIAS( couple, angacc);
// XSTAMP_DECLARE_ALIAS( angmom, angvel);

namespace exanb
{
  struct __unused_field_id
  {
  };
  static inline constexpr bool HAS_POSITION_BACKUP_FIELDS = false;
  static constexpr __unused_field_id PositionBackupFieldX = {};
  static constexpr __unused_field_id PositionBackupFieldY = {};
  static constexpr __unused_field_id PositionBackupFieldZ = {};
} // namespace exanb
