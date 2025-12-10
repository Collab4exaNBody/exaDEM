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
#include <exaDEM/common_compute_kernels.h>
#include <exaDEM/contact_force_parameters.h>

enum class ContactLawType
{
    Hooke
    // Hertz 
    // ...
};

enum class CohesiveLawType
{
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

    // Hertz, JKR, stick, ...
    // Hertz_None,
    // Hertz_Cohesive,
    // ...
};

template<LawComboType>
struct LawComboTraits;

// ===== Hooke - None =====
template<>
struct LawComboTraits<LawComboType::Hooke_None>
{
    static constexpr bool hooke    = true;
    static constexpr bool cohesive = false;
    static constexpr bool dmt      = false;
};

// ===== Hooke - Cohesive =====
template<>
struct LawComboTraits<LawComboType::Hooke_Cohesive>
{
    static constexpr bool hooke    = true;
    static constexpr bool cohesive = true;
    static constexpr bool dmt      = false;
};

// ===== Hooke - DMT =====
template<>
struct LawComboTraits<LawComboType::Hooke_DMT>
{
    static constexpr bool hooke    = true;
    static constexpr bool cohesive = false;
    static constexpr bool dmt      = true;
};

constexpr LawComboType makeLawCombo(ContactLawType c, CohesiveLawType h)
{
    switch (c)
    {
        case ContactLawType::Hooke:
            switch (h)
            {
                case CohesiveLawType::None:     return LawComboType::Hooke_None;
                case CohesiveLawType::Cohesive: return LawComboType::Hooke_Cohesive;
                case CohesiveLawType::DMT:      return LawComboType::Hooke_DMT;
            }
            break;

        // case ContactLawType::Hertz:
        //     ...
    }

    throw std::logic_error("Unsupported law combination");
}



namespace exaDEM
{
  
  ONIKA_HOST_DEVICE_FUNC inline void reset(Vec3d &in) { in = Vec3d{0.0, 0.0, 0.0}; }

  /**
   * @brief Applies the DMT (Derjaguin–Muller–Toporov) adhesion force to a contact.
   *
   * This function computes and adds the attractive DMT contribution when two
   * particles or surfaces are in contact (i.e., when the normal overlap `dn`
   * is negative).
   
   * @param hkp   Contact parameters, including the surface energy `gamma`.
   * @param reff  Effective contact radius used in the DMT formulation.
   * @param dn    Normal overlap. The force is applied only when `dn < 0`.
   * @param n     Unit normal vector at the contact point, oriented from j to i.
   * @param f_i   Reference to the resulting force applied to particle i.
   */
  ONIKA_HOST_DEVICE_FUNC inline void apply_dmt_force(const ContactParams& hkp, double reff,
     double dn, 
     const Vec3d& n, 
     Vec3d& f_i)
  {
    // Disabled if gamma <= 0
    if(hkp.gamma <= 0.0) return;

    // DMT : attractive force only if  contact (dn < 0)
    if(dn < 0.0)
    {
        double F_dmt = 2.0 * M_PI * reff * hkp.gamma;

        // attractive force : -F_dmt * n
        f_i -= F_dmt * n;
    }
  }

  ONIKA_HOST_DEVICE_FUNC inline void cohesive_force_core(const double dn, const Vec3d &n, const double dncut, const double fc, Vec3d &f)
  {
    if (dncut == 0)
      return;

    if (dn <= dncut)
    {
      const double fn_value = (fc / dncut) * dn - fc;
      const Vec3d fn = fn_value * n;

      // === update forces
      f.x += fn.x;
      f.y += fn.y;
      f.z += fn.z;
    }
  }

  template<ContactLawType ContactLaw, CohesiveLawType CohesiveLaw>
    ONIKA_HOST_DEVICE_FUNC inline void contact_force_core(const double dn,
        const Vec3d &n, // -normal
        const double dt,
        const ContactParams &hkp,
        const double meff,
        const double reff,
        Vec3d &ft, // tangential force between particle i and j
        const Vec3d &contact_position,
        const Vec3d &pos_i,  // positions i
        const Vec3d &vel_i,  // velocities i
        Vec3d &f_i,          // forces i
        Vec3d &mom_i,        // moments i
        const Vec3d &vrot_i, // angular velocities i
        const Vec3d &pos_j,  // positions j
        const Vec3d &vel_j,  // velocities j
        const Vec3d &vrot_j  // angular velocities j 
        )
      {

      constexpr auto LawCombo = makeLawCombo(ContactLaw, CohesiveLaw);
     
      // === Cohesive law : before contact
      if constexpr (LawComboTraits<LawCombo>::cohesive)
      {
        static_assert(LawComboTraits<LawCombo>::hooke, "Cohesive law must be combined with a contact law."  );//||hertz later
        if( dn >= 0 ) // dn <= hkp.dncut if contact
        {
          const double fn_value = (hkp.fc / hkp.dncut) * dn - hkp.fc;
          f_i += fn_value * n;
          ft = {0, 0, 0};
          return;
        }
      }

      // === Compute damping coefficient
      const double damp = compute_damp(hkp.damp_rate, hkp.kn, meff); 

      // === Relative velocity (j relative to i)
      auto vel = compute_relative_velocity(contact_position, pos_i, vel_i, vrot_i, pos_j, vel_j, vrot_j);
    
      // === Compute relative velocity
      const double vn = exanb::dot(vel, n);
      Vec3d fn; // normal force

      // === Compute Contact forces

      // Hooke 
      if constexpr (LawCombo == LawComboType::Hooke_None || LawCombo == LawComboType::Hooke_DMT)
      {
         // - Normal force (elastic contact + viscous damping)
         fn = compute_normal_force(hkp.kn, damp, dn, vn, n); // (fc ==> cohesive force)

         // - Tangential force (friction)
         ft += exaDEM::compute_tangential_force(hkp.kt, dt, vn, n, vel);
      
         // - Fit tangential force
         auto threshold_ft = exaDEM::compute_threshold_ft(hkp.mu, hkp.kn, dn);
         exaDEM::fit_tangential_force(threshold_ft, ft);
      }

      // Cohesive 
      if constexpr (LawCombo == LawComboType::Hooke_Cohesive )
      {
         // - Normal force (elatic contact + viscous damping)
         fn = exaDEM::compute_normal_force_value_with_cohesive_force(hkp.kn, hkp.fc, damp, dn, vn, n);

         // - Tangential force (friction)
         ft += exaDEM::compute_tangential_force(hkp.kt, dt, vn, n, vel);

         const double fn_norm = exanb::dot(fn, n);

         // - Fit tangential force
         auto threshold_ft = exaDEM::compute_threshold_ft_with_cohesive_force(hkp.mu, fn_norm, hkp.fc);
         exaDEM::fit_tangential_force(threshold_ft, ft);
      }

      // Other contact laws can be added here
      // if constexpr (ContactLaw == ContactLawType::Hertz)
      // (...)

      // === Sum forces
      f_i = fn + ft;

      // === Adhesion Laws

      // DMT adhesive force
      if constexpr (LawComboTraits<LawCombo>::dmt)
      {
        apply_dmt_force(hkp, reff, dn, n, f_i);
      }
      // Other adhesion laws can be added here
      // if constexpr (LawComboTraits<LawCombo>::jkr)

      // === update moments
      mom_i += hkp.kr * (vrot_j - vrot_i) * dt;

      ///*
      // test
      Vec3d branch = contact_position - pos_i;
      double r = (exanb::dot(branch, vrot_i)) / (exanb::dot(vrot_i, vrot_i));
      branch -= r * vrot_i;

      constexpr double mur = 0;
      double threshold_mom = std::abs(mur * exanb::norm(branch) * exanb::norm(fn)); // even without fabs, the value should
                                                                                    // be positive
      double mom_square = exanb::dot(mom_i, mom_i);
      if (mom_square > 0.0 && mom_square > threshold_mom * threshold_mom)
        mom_i = mom_i * (threshold_mom / sqrt(mom_square));
      //*/

    }

  ONIKA_HOST_DEVICE_FUNC inline Vec3d compute_moments(const Vec3d &contact_position,
      const Vec3d &p, // position
      const Vec3d &f, // forces
      const Vec3d &m) // I.mom
  {
    const auto Ci = (contact_position - p);
    const auto Pimoment = exanb::cross(Ci, f) + m;
    return Pimoment;
  }
} // namespace exaDEM
