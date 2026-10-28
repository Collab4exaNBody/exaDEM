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

#include <exaDEM/forcefield/contact_force_common.hpp>
#include <exaDEM/forcefield/contact_force_hertz.hpp>
#include <exaDEM/forcefield/contact_force_hooke.hpp>

namespace exaDEM {

/** Parameters of a contact law: ContactParams for the Hooke law, HertzParams for the Hertz law. */
template <ContactLawType ContactLaw>
struct ContactLawParams;

template <>
struct ContactLawParams<ContactLawType::Hooke> {
  using type = ContactParams;
};

template <>
struct ContactLawParams<ContactLawType::Hertz> {
  using type = HertzParams;
};

template <ContactLawType ContactLaw>
using ContactLawParamsT = typename ContactLawParams<ContactLaw>::type;

/**
 * @brief Computes the contact force of the contact law ContactLaw (see contact_force_hooke.hpp and
 * contact_force_hertz.hpp), `hkp` being the parameters of the law (ContactLawParamsT<ContactLaw>).
 */
template <ContactLawType ContactLaw, CohesiveLawType CohesiveLaw, typename ParamsT>
ONIKA_HOST_DEVICE_FUNC inline void contact_force_core(const double dn,
                                                      const Vec3d& n,  // -normal
                                                      const double dt, const ParamsT& hkp, const double meff,
                                                      const double reff,
                                                      Vec3d& ft,  // tangential force between particle i and j
                                                      const Vec3d& contact_position,
                                                      const Vec3d& pos_i,   // positions i
                                                      const Vec3d& vel_i,   // velocities i
                                                      Vec3d& f_i,           // forces i
                                                      Vec3d& mom_i,         // moments i
                                                      const Vec3d& vrot_i,  // angular velocities i
                                                      const Vec3d& pos_j,   // positions j
                                                      const Vec3d& vel_j,   // velocities j
                                                      const Vec3d& vrot_j   // angular velocities j
) {
  static_assert(std::is_same_v<ParamsT, ContactLawParamsT<ContactLaw>>, "wrong parameters for this contact law");
  if constexpr (ContactLaw == ContactLawType::Hooke) {
    hooke_contact_force_core<CohesiveLaw>(dn, n, dt, hkp, meff, reff, ft, contact_position, pos_i, vel_i, f_i, mom_i,
                                          vrot_i, pos_j, vel_j, vrot_j);
  } else {
    hertz_contact_force_core<CohesiveLaw>(dn, n, dt, hkp, meff, reff, ft, contact_position, pos_i, vel_i, f_i, mom_i,
                                          vrot_i, pos_j, vel_j, vrot_j);
  }
}
}  // namespace exaDEM
