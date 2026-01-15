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

namespace exaDEM {
template <typename FieldT>
struct poly_div_field_volume {
  const shape* shps;
  ONIKA_HOST_DEVICE_FUNC inline void operator()(const uint32_t type,
                                                double homothety,
                                                FieldT& value) const {
    const double volume = shps[type].get_volume(homothety);
    value = value / volume;
  }
};

template <typename FieldT>
struct sphere_div_field_volume {
  double coeff_4t3_pi = (4 / 3) * std::atan(-1);
  ONIKA_HOST_DEVICE_FUNC inline void operator()(const double radius,
                                                FieldT& value) const {
    const double volume = coeff_4t3_pi * radius * radius * radius;
    value = value / volume;
  }
};
}  // namespace exaDEM

namespace exanb {
template <typename FieldT>
struct ComputeCellParticlesTraits<exaDEM::poly_div_field_volume<FieldT>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};

template <typename FieldT>
struct ComputeCellParticlesTraits<exaDEM::sphere_div_field_volume<FieldT>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace exanb
