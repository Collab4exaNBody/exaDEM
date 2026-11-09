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

#include <onika/physics/units.h>
#include <onika/string_utils.h>
#include <yaml-cpp/yaml.h>

#include <exaDEM/forcefield/contact_parameters.hpp>  // display_header, display_end_table

namespace exaDEM {
using onika::lout;
/**
 * @struct HertzParams
 * @brief Contact parameters of the Hertz contact law (spheres). See ContactParams for the Hooke law.
 *
 * The normal stiffness of the Hertz law is not a parameter: it depends on the overlap and on the radii of the
 * particles. It is defined by the reduced Young modulus E* (estar_), computed from the Young modulus and the Poisson
 * ratio of the two materials in contact.
 */
struct HertzParams {
  double estar_ = 0;   /**< Reduced Young modulus E* of the contact: 1/E* = (1-nu1^2)/E1 + (1-nu2^2)/E2. */
  double young_ = 0;   /**< Young modulus of the particle material (used to estimate the critical time step). */
  double poisson_ = 0; /**< Poisson ratio of the particle material (used to estimate the critical time step). */
  double kt_ = 0; /**< Tangential stiffness coefficient (force per unit displacement in the tangential direction). */
  double kr_ = 0; /**< Rotational stiffness coefficient (torque per unit angular displacement). */
  double mu_ = 0; /**< Friction coefficient (Coulomb friction model). */
  double damp_rate_ = 0; /**< Damping rate (ratio at the tangent stiffness of the contact, controls dissipation). */
  double fc_ = 0;        /**< Cohesive force threshold (e.g., for bonded contacts). */
  double dncut_ = 0;     /**< Distance cutoff below which contact is considered active for cohesion force. */
  double gamma_ = 0;     /**< Adhesion energie per unit of surface (default is 0). */
};

/**
 * @brief Reduced Young modulus E* of a contact between two materials.
 *
 * 1 / E* = (1 - nu1^2) / E1 + (1 - nu2^2) / E2
 */
inline double compute_reduced_young_modulus(double young1, double poisson1, double young2, double poisson2) {
  return 1.0 / ((1.0 - poisson1 * poisson1) / young1 + (1.0 - poisson2 * poisson2) / young2);
}

inline bool operator==(HertzParams& a, HertzParams& b) {
  return (a.estar_ == b.estar_) && (a.young_ == b.young_) && (a.poisson_ == b.poisson_) && (a.kt_ == b.kt_) &&
         (a.kr_ == b.kr_) && (a.mu_ == b.mu_) && (a.damp_rate_ == b.damp_rate_) && (a.fc_ == b.fc_) &&
         (a.dncut_ == b.dncut_) && (a.gamma_ == b.gamma_);
}

template <>
inline void display_header<HertzParams>() {
  lout << std::endl;
  lout << "============================================================================================================"
          "==================="
       << std::endl;
  lout << "|        typeA |        typeB |     estar |        kt |        kr |        mu |        fc |  damprate |    "
          "dncut  |    gamma  |"
       << std::endl;
  lout << "------------------------------------------------------------------------------------------------------------"
          "-------------------"
       << std::endl;
}

template <>
inline void display_end_table<HertzParams>() {
  lout << "============================================================================================================"
          "======="
       << std::endl;
}

inline std::string display(HertzParams& params) {
  return onika::format_string(" %.3e | %.3e | %.3e | %.3e | %.3e | %.3e | %.3e | %.3e |", params.estar_, params.kt_,
                              params.kr_, params.mu_, params.fc_, params.damp_rate_, params.dncut_, params.gamma_);
}

inline void display_multimat(std::string typeA, std::string typeB, HertzParams& params) {
  std::string line_types = onika::format_string("| %12s | %12s |", typeA, typeB);
  lout << line_types << display(params) << std::endl;
}

template <typename STREAM>
void streaming(STREAM& stream, HertzParams& params) {
  stream << "{ estar: " << params.estar_ << ", kt: " << params.kt_ << ", kr: " << params.kr_
         << ", damp_rate: " << params.damp_rate_ << ", dncut: " << params.dncut_ << ", fc: " << params.fc_
         << ", gamma: " << params.gamma_ << " }";
}

}  // namespace exaDEM

// Yaml conversion operators, allows to read potential parameters from config file
namespace YAML {
using exaDEM::HertzParams;
using exanb::lerr;
using onika::physics::Quantity;

template <>
struct convert<HertzParams> {
  static bool decode(const Node& node, HertzParams& v) {
    v = HertzParams{};  // initializes defaults values
    if (!node.IsMap()) {
      return false;
    }

    // Reduced Young modulus: given directly (estar), or computed from the Young modulus and the Poisson ratio of
    // the material (contact between two particles of this material).
    if (node["estar"]) {
      v.estar_ = node["estar"].as<Quantity>().convert();
    }
    if (node["young"] || node["poisson"]) {
      if (!node["young"] || !node["poisson"]) {
        lerr << "young and poisson must be defined together\n";
        return false;
      }
      v.young_ = node["young"].as<Quantity>().convert();
      v.poisson_ = node["poisson"].as<Quantity>().convert();
      if (!node["estar"]) {
        v.estar_ = exaDEM::compute_reduced_young_modulus(v.young_, v.poisson_, v.young_, v.poisson_);
      }
    }
    if (!node["estar"] && !node["young"]) {
      lerr << "estar, or young and poisson, is missing\n";
      return false;
    }

    if (!node["dncut"] && !node["fc"]) {
      v.dncut_ = 0.0;
      v.fc_ = 0.0;
    } else if (node["dncut"] && node["fc"]) {
      v.dncut_ = node["dncut"].as<Quantity>().convert();
      v.fc_ = node["fc"].as<Quantity>().convert();
    } else if (!node["dncut"]) {
      lerr << "dncut is missing\n";
      return false;
    } else if (!node["fc"]) {
      lerr << "fc is missing\n";
      return false;
    }

    if (!node["kt"]) {
      lerr << "kt is missing\n";
      return false;
    }
    if (!node["mu"]) {
      lerr << "mu is missing\n";
      return false;
    }
    if (!node["damp_rate"]) {
      lerr << "damp_rate is missing\n";
      return false;
    }

    v.gamma_ = node["gamma"] ? node["gamma"].as<Quantity>().convert() : 0.0;
    v.kt_ = node["kt"].as<Quantity>().convert();
    if (node["kr"]) {
      v.kr_ = node["kr"].as<Quantity>().convert();
    }
    v.mu_ = node["mu"].as<Quantity>().convert();
    v.damp_rate_ = node["damp_rate"].as<Quantity>().convert();

    return true;
  }
};
}  // namespace YAML
