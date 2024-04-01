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

#include <exanb/core/quaternion_operators.h>

namespace exaDEM
{

  struct PushToQuaternionFunctor
  {
    double m_dt;
    double m_dt_2;
    double m_dt2_2;

    ONIKA_HOST_DEVICE_FUNC inline void operator () (exanb::Quaternion& Q, exanb::Vec3d& vrot, const exanb::Vec3d arot) const
    {
      using namespace exanb;
/*      double omega2_2 = 0.5 * dot(vrot , vrot);
      auto Qxyz = Vec3d{Q.x,Q.y,Q.z};
      auto tmpxyz = 0.5 * (m_dt * (Q.w * vrot + cross(vrot, Qxyz)) +
               m_dt2_2 * (Q.w * arot + cross(arot, Qxyz) - omega2_2 * Qxyz));
      auto tmpw = -0.5 * ( dot(vrot,Qxyz) * m_dt + ( dot(arot , Qxyz) + omega2_2) * m_dt2_2);

      Q = Q + Quaternion{tmpxyz.x,tmpxyz.y,tmpxyz.z, tmpw};
      Q = normalize(Q);
*/

//    Particles[i].Q += ((Particles[i].Q.dot(Particles[i].vrot)) *=dt);

			Q = Q + dot(Q, vrot) * m_dt;
      Q = normalize(Q);

      vrot += m_dt_2 * arot;
    }
  };
}

namespace exanb
{

  template<> struct ComputeCellParticlesTraits<exaDEM::PushToQuaternionFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };

}


