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
#include <exanb/compute/compute_cell_particles.h>

namespace exaDEM {
using namespace exanb;
struct QuadraticForceFunctor {
  double cx_mu;
  ONIKA_HOST_DEVICE_FUNC inline void operator()(double& fx, double& fy, double& fz, const double vx, const double vy,
                                                const double vz) const {
    Vec3d v = {vx, vy, vz};
    double vel = exanb::norm(v);
    fx -= cx_mu * vel * vx;
    fy -= cx_mu * vel * vy;
    fz -= cx_mu * vel * vz;
  }
};
}  // namespace exaDEM

namespace exanb {
template <>
struct ComputeCellParticlesTraits<exaDEM::QuadraticForceFunctor> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace exanb
