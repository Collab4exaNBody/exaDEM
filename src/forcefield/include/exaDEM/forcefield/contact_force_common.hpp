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

#include <cmath>
#include <stdexcept>

#include <exaDEM/forcefield/common_kernels.hpp>

// === Contact laws

enum class ContactLawType {
  Hooke,
  Hertz  // spheres only
  // ...
};

enum class CohesiveLawType {
  None,
  Cohesive,
  DMT
  // JKR
  // stick ...
};

enum class LawComboType {
  Hooke_None,
  Hooke_Cohesive,
  Hooke_DMT,

  Hertz_None,
  Hertz_Cohesive,
  Hertz_DMT,

  // JKR, stick, ...
};

template <LawComboType>
struct LawComboTraits;

#define EXADEM_LAW_COMBO_TRAITS(COMBO, HOOKE, HERTZ, COHESIVE, DMT) \
  template <>                                                       \
  struct LawComboTraits<LawComboType::COMBO> {                      \
    static constexpr bool hooke = HOOKE;                            \
    static constexpr bool hertz = HERTZ;                            \
    static constexpr bool cohesive = COHESIVE;                      \
    static constexpr bool dmt = DMT;                                \
  };

EXADEM_LAW_COMBO_TRAITS(Hooke_None, true, false, false, false)
EXADEM_LAW_COMBO_TRAITS(Hooke_Cohesive, true, false, true, false)
EXADEM_LAW_COMBO_TRAITS(Hooke_DMT, true, false, false, true)
EXADEM_LAW_COMBO_TRAITS(Hertz_None, false, true, false, false)
EXADEM_LAW_COMBO_TRAITS(Hertz_Cohesive, false, true, true, false)
EXADEM_LAW_COMBO_TRAITS(Hertz_DMT, false, true, false, true)

#undef EXADEM_LAW_COMBO_TRAITS

constexpr LawComboType makeLawCombo(ContactLawType c, CohesiveLawType h) {
  switch (c) {
    case ContactLawType::Hooke:
      switch (h) {
        case CohesiveLawType::None:
          return LawComboType::Hooke_None;
        case CohesiveLawType::Cohesive:
          return LawComboType::Hooke_Cohesive;
        case CohesiveLawType::DMT:
          return LawComboType::Hooke_DMT;
      }
      break;

    case ContactLawType::Hertz:
      switch (h) {
        case CohesiveLawType::None:
          return LawComboType::Hertz_None;
        case CohesiveLawType::Cohesive:
          return LawComboType::Hertz_Cohesive;
        case CohesiveLawType::DMT:
          return LawComboType::Hertz_DMT;
      }
      break;
  }

  throw std::logic_error("Unsupported law combination");
}

namespace exaDEM {

ONIKA_HOST_DEVICE_FUNC inline void reset(Vec3d& in) { in = Vec3d{0.0, 0.0, 0.0}; }

/**
 * @brief Applies the DMT (Derjaguin–Muller–Toporov) adhesion force to a contact.
 *
 * This function computes and adds the attractive DMT contribution when two
 * particles or surfaces are in contact (i.e., when the normal overlap `dn`
 * is negative).

 * @param gamma Adhesion energy per unit of surface. The adhesion is disabled if gamma <= 0.
 * @param reff  Effective contact radius used in the DMT formulation.
 * @param dn    Normal overlap. The force is applied only when `dn < 0`.
 * @param n     Unit normal vector at the contact point, oriented from j to i.
 * @param f_i   Reference to the resulting force applied to particle i.
 */
ONIKA_HOST_DEVICE_FUNC inline void apply_dmt_force(const double gamma, double reff, double dn, const Vec3d& n,
                                                   Vec3d& f_i) {
  // Disabled if gamma <= 0
  if (gamma <= 0.0) return;

  // DMT : attractive force only if  contact (dn < 0)
  if (dn < 0.0) {
    double F_dmt = 2.0 * M_PI * reff * gamma;

    // attractive force : -F_dmt * n
    f_i -= F_dmt * n;
  }
}

/**
 * @brief Updates the rolling moment of particle i (rotational stiffness kr).
 *
 * @param kr Rotational stiffness coefficient.
 * @param dt Time step.
 * @param fn Normal force (used by the rolling friction limit).
 */
ONIKA_HOST_DEVICE_FUNC inline void update_rolling_moment(const double kr, const double dt,
                                                         const Vec3d& contact_position, const Vec3d& pos_i,
                                                         const Vec3d& vrot_i, const Vec3d& vrot_j, const Vec3d& fn,
                                                         Vec3d& mom_i) {
  mom_i += kr * (vrot_j - vrot_i) * dt;

  Vec3d branch = contact_position - pos_i;
  double r = (exanb::dot(branch, vrot_i)) / (exanb::dot(vrot_i, vrot_i));
  branch -= r * vrot_i;

  constexpr double mur = 0;
  double threshold_mom = std::abs(mur * exanb::norm(branch) * exanb::norm(fn));  // even without fabs, the value should
                                                                                 // be positive
  double mom_square = exanb::dot(mom_i, mom_i);
  if (mom_square > 0.0 && mom_square > threshold_mom * threshold_mom)
    mom_i = mom_i * (threshold_mom / sqrt(mom_square));
}
}  // namespace exaDEM
