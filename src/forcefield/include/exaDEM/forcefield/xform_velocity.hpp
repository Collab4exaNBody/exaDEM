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

#include <exanb/core/xform.h>

namespace exaDEM {
using exanb::Mat3d;
using exanb::Vec3d;

// Deforming periodic cell (barostat): diagonal xform + its strain rate vh / h
struct DeformingCellXForm {
  Mat3d m_matrix;
  Vec3d m_strain_rate;
  ONIKA_HOST_DEVICE_FUNC inline Vec3d transformCoord(const Vec3d& x) const { return m_matrix * x; }
  ONIKA_HOST_DEVICE_FUNC inline Vec3d transformVel(const Vec3d& v, const Vec3d& r) const {
    return v + Vec3d{m_strain_rate.x * r.x, m_strain_rate.y * r.y, m_strain_rate.z * r.z};
  }
};

template <typename XFormT>
ONIKA_HOST_DEVICE_FUNC inline Vec3d transform_vel(const XFormT&, const Vec3d& v, const Vec3d&) {
  return v;
}

ONIKA_HOST_DEVICE_FUNC inline Vec3d transform_vel(const DeformingCellXForm& x, const Vec3d& v, const Vec3d& r) {
  return x.transformVel(v, r);
}
}  // namespace exaDEM
