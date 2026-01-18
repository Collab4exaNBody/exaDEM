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

#include <yaml-cpp/yaml.h>
#include <onika/string_utils.h>
#include <onika/physics/units.h>

namespace exaDEM {
using onika::lout;
/**
 * @struct ContactParams
 * @brief Encapsulates contact mechanics parameters for a contact interaction model.
 */
struct ContactParams {
  double kn = 0; /**< Normal stiffness coefficient (force per unit displacement in the normal direction). */
  double kt = 0; /**< Tangential stiffness coefficient (force per unit displacement in the tangential direction). */
  double kr = 0; /**< Rotational stiffness coefficient (torque per unit angular displacement). */
  double mu = 0; /**< Friction coefficient (Coulomb friction model). */
  double damp_rate = 0; /**< Damping rate for contact interaction (controls dissipation). */
  double fc = 0;        /**< Cohesive force threshold (e.g., for bonded contacts). */
  double dncut = 0;     /**< Distance cutoff below which contact is considered active for cohesion force. */
  double gamma = 0;     /**< Adhesion energie per unit of surface (default is 0). */
};

/**
 * @brief Equality operator for comparing two ContactParams instances.
 *
 * @param a First ContactParams instance.
 * @param b Second ContactParams instance.
 * @return true if all fields of both instances are equal; false otherwise.
 */
inline bool operator==(ContactParams& a, ContactParams& b) {
  return (a.dncut == b.dncut) && (a.kn == b.kn) && (a.kt == b.kt) && (a.kr == b.kr) && (a.mu == b.mu) &&
         (a.fc == b.fc) && (a.damp_rate == b.damp_rate) && (a.gamma == b.gamma);
}

/**
 * @brief Displays the header for a parameter table.
 */
template <typename CPT>
inline void display_header();

/**
 * @brief Displays the footer or closing line for a parameter table.
 */
template <typename CPT>
inline void display_end_table();

/**
 * @brief Displays a formatted header line for a table of ContactParams entries.
 */
template <>
inline void display_header<ContactParams>() {
  lout << "============================================================================================================"
          "==================="
       << std::endl;
  lout << "|        typeA |        typeB |        kn |        kt |        kr |        mu |        fc |  damprate |    "
          "dncut  |    gamma  |"
       << std::endl;
  lout << "------------------------------------------------------------------------------------------------------------"
          "-------------------"
       << std::endl;
}

/**
 * @brief Displays a formatted footer line for a table of ContactParams entries.
 */
template <>
inline void display_end_table<ContactParams>() {
  lout << "============================================================================================================"
          "======="
       << std::endl;
}

/**
 * @brief Formats the contact parameters into a single table row string.
 */
inline std::string display(ContactParams& params) {
  std::string line =
      onika::format_string(" %.3e | %.3e | %.3e | %.3e | %.3e | %.3e | %.3e | %.3e |", params.kn, params.kt, params.kr,
                           params.mu, params.fc, params.damp_rate, params.dncut, params.gamma);
  return line;
}

/**
 * @brief Displays a full formatted line in the contact parameters table.
 *
 * @param typeA Identifier for the first type (e.g., material A).
 * @param typeB Identifier for the second type (e.g., material B).
 * @param params Reference to the ContactParams instance to display.
 */
inline void display_multimat(std::string typeA, std::string typeB, ContactParams& params) {
  std::string line_types = onika::format_string("| %12s | %12s |", typeA, typeB);
  std::string line_params = display(params);
  lout << line_types << line_params << std::endl;
}

/**
 * @brief Streams the ContactParams as a key-value representation.
 *
 * @tparam STREAM The stream type (e.g., std::ostream).
 * @param stream Output stream to write to.
 * @param params Reference to the ContactParams instance to stream.
 */
template <typename STREAM>
void streaming(STREAM& stream, ContactParams& params) {
  stream << "{ kn: " << params.kn << ", kt: " << params.kt << ", kr: " << params.kr
         << ", damp_rate: " << params.damp_rate << ", dncut: " << params.dncut << ", fc: " << params.fc
         << ", gamma: " << params.gamma << " }";
}
}  // namespace exaDEM

// Yaml conversion operators, allows to read potential parameters from config file
namespace YAML {
using exaDEM::ContactParams;
using exanb::lerr;
using onika::physics::Quantity;

template <>
struct convert<ContactParams> {
  static bool decode(const Node& node, ContactParams& v) {
    v = ContactParams{};  // initializes defaults values
    if (!node.IsMap()) {
      return false;
    }
    if (!node["dncut"] && !node["fc"]) {
      v.dncut = 0.0;
      v.fc = 0.0;
    } else if (node["dncut"] && node["fc"]) {
      v.dncut = node["dncut"].as<Quantity>().convert();
      v.fc = node["fc"].as<Quantity>().convert();
    } else if (!node["dncut"]) {
      lerr << "dncut is missing\n";
      return false;
    } else if (!node["fc"]) {
      lerr << "fc is missing\n";
      return false;
    }

    if (!node["kn"]) {
    }
    if (!node["kt"]) {
      lerr << "kt is missing\n";
      return false;
    }
    if (!node["kr"]) {
      lerr << "kr is missing\n";
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

    if (node["gamma"]) {
      v.gamma = node["gamma"].as<Quantity>().convert();
    } else {
      v.gamma = 0.0;  // valeur par défaut
    }

    v.kn = node["kn"].as<Quantity>().convert();
    v.kt = node["kt"].as<Quantity>().convert();
    v.kr = node["kr"].as<Quantity>().convert();
    v.mu = node["mu"].as<Quantity>().convert();
    v.damp_rate = node["damp_rate"].as<Quantity>().convert();

    return true;
  }
};
}  // namespace YAML
