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
#include <onika/math/basic_types_operators.h>

// These kernels are closed / idenitcal to the kernels defined whithin the numerical scheme plugin
namespace exaDEM {
struct DriverPushToQuaternionFunctor {
  double m_dt;
  double m_dt_2;
  double m_dt2_2;

  ONIKA_HOST_DEVICE_FUNC
  inline void operator()(exanb::Quaternion& Q, exanb::Vec3d& vrot, const exanb::Vec3d arot) const {
    Q = Q + dot(Q, vrot) * m_dt;
    Q = normalize(Q);
    vrot += m_dt_2 * arot;
  }
};

struct DriverPushToAngularVelocityFunctor {
  double m_dt_2;
  ONIKA_HOST_DEVICE_FUNC
  inline void operator()(exanb::Vec3d& vrot, const exanb::Vec3d& arot) const { vrot += arot * m_dt_2; }
};

struct DriverPushToAngularAccelerationFunctor {
  ONIKA_HOST_DEVICE_FUNC
  inline void operator()(const exanb::Quaternion& Q, const exanb::Vec3d& mom, const exanb::Vec3d& vrot,
                         exanb::Vec3d& arot, const exanb::Vec3d& inertia) const {
    using exanb::Quaternion;
    using exanb::Vec3d;
    Quaternion Qinv = get_conjugated(Q);
    const auto omega = Qinv * vrot;  // Express omega in the body framework
    const auto M = Qinv * mom;       // Express torque in the body framework
    Vec3d domega{(M.x - (inertia.z - inertia.y) * omega.y * omega.z) / inertia.x,
                 (M.y - (inertia.x - inertia.z) * omega.z * omega.x) / inertia.y,
                 (M.z - (inertia.y - inertia.x) * omega.x * omega.y) / inertia.z};
    arot = Q * domega;  // Express arot in the global framework
  }
};
}  // namespace exaDEM
