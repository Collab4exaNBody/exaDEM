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

#include <exaDEM/interface/rupture_criterion.hpp>

namespace exaDEM {
using onika::lout;
/**
 * @struct InnerBondParams
 * @brief Encapsulates contact mechanics parameters for a contact interaction model.
 */
struct InnerBondParams {
  double kn_ = 0.0; /**< Normal stiffness coefficient (force per unit displacement in the normal direction). */
  double kt_ = 0.0; /**< Tangential stiffness coefficient (force per unit displacement in the tangential direction). */
  double damp_rate_ = 0.0;
  /** Rupture mode for the fracture criterion. Set to MixedMode when the user provides "g" in the input file,
   * SeparateModes when the user provides "gn" and "gt" instead. */
  RuptureMode mode_ = RuptureMode::None;
  double crit1_ = 0.0; /**< MixedMode: combined fracture energy release rate (g). SeparateModes: normal fracture energy
                      release rate. */
  double crit2_ = 0.0; /**< SeparateModes only: tangential fracture energy release rate. */
};

/**
 * @brief Equality operator for comparing two InnerBondParams instances.
 *
 * @param a First InnerBondParams instance.
 * @param b Second InnerBondParams instance.
 * @return true if all fields of both instances are equal; false otherwise.
 */
inline bool operator==(InnerBondParams& a, InnerBondParams& b) {
  return (a.kn_ == b.kn_) && (a.kt_ == b.kt_) && (a.damp_rate_ == b.damp_rate_) && (a.mode_ == b.mode_) &&
         (a.crit1_ == b.crit1_) && (a.crit2_ == b.crit2_);
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
 * @brief Displays a formatted header line for a table of InnerBondParams entries.
 */
template <>
inline void display_header<InnerBondParams>() {
  lout << std::endl;
  lout << "============================================================================================================"
          "=========="
       << std::endl;
  lout << "|       groupA |       groupB |        kn |        kt | damp_rate |             mode : Mixed |         g |  "
          "       X |"
       << std::endl;
  lout << "|              |              |           |           |           |                Separated |        gn |  "
          "      gt |"
       << std::endl;
  lout << "|              |              |           |           |           |            Stress Energy |         g |  "
          "   sigma |"
       << std::endl;

  lout << "------------------------------------------------------------------------------------------------------------"
          "----------"
       << std::endl;
}

/**
 * @brief Displays a formatted footer line for a table of InnerBondParams entries.
 */
template <>
inline void display_end_table<InnerBondParams>() {
  lout << "=========================================================================================================="
          "============"
       << std::endl
       << std::endl;
}

/**
 * @brief Formats the contact parameters into a single table row string.
 */
inline std::string display(InnerBondParams& params) {
  std::string line =
      onika::format_string(" %.3e | %.3e | %.3e | %24s | %.3e | %.3e |", params.kn_, params.kt_, params.damp_rate_,
                           exaDEM::display(params.mode_), params.crit1_, params.crit2_);
  return line;
}

/**
 * @brief Displays a full formatted line in the contact parameters table.
 *
 * @param groupA Identifier for the first type (e.g., material A).
 * @param groupB Identifier for the second type (e.g., material B).
 * @param params Reference to the InnerBondParams instance to display.
 */
inline void display_multimat(std::string groupA, std::string groupB, InnerBondParams& params) {
  std::string line_types = onika::format_string("| %12s | %12s |", groupA, groupB);
  std::string line_params = display(params);
  lout << line_types << line_params << std::endl;
}

/**
 * @brief Streams the InnerBondParams as a key-value representation.
 *
 * @tparam STREAM The stream type (e.g., std::ostream).
 * @param stream Output stream to write to.
 * @param params Reference to the InnerBondParams instance to stream.
 */
template <typename STREAM>
void streaming(STREAM& stream, InnerBondParams& params) {
  stream << "{ kn: " << params.kn_ << ", kt: " << params.kt_ << ", damp_rate: " << params.damp_rate_
         << ", mode: " << exaDEM::display(params.mode_) << ", crit1: " << params.crit1_ << ", crit2: " << params.crit2_
         << " }";
}
}  // namespace exaDEM

// Yaml conversion operators, allows to read potential parameters from config file
namespace YAML {
using exaDEM::InnerBondParams;
using exanb::lerr;
using onika::physics::Quantity;

template <>
struct convert<InnerBondParams> {
  static bool decode(const Node& node, InnerBondParams& v) {
    v = InnerBondParams{};  // initializes defaults values
    if (!node.IsMap()) {
      return false;
    }
    if (!node["kn"]) {
      lerr << "kn is missing\n";
      return false;
    }
    if (!node["kt"]) {
      lerr << "kt is missing\n";
      return false;
    }
    if (!node["damp_rate"]) {
      lerr << "en is missing\n";
      return false;
    }

    v.kn_ = node["kn"].as<Quantity>().convert();
    v.kt_ = node["kt"].as<Quantity>().convert();
    v.damp_rate_ = node["damp_rate"].as<Quantity>().convert();

    // StressEnergySeparateMode: The user provides "sigma" and "g" in the input file. The fracture criterion is based on
    // a combination of stress and energy.
    if (node["sigma"]) {
      if (!node["g"]) {
        lerr << "g is missing\n";
        return false;
      }
      v.crit1_ = node["g"].as<Quantity>().convert();
      v.mode_ = exaDEM::RuptureMode::StressEnergySeparateMode;
      v.crit2_ = node["sigma"].as<Quantity>().convert();

    } else if (node["g"]) {  // EnergyMixedMode: The user provides "g" in the input file. The fracture criterion is
                             // based on the sum of normal and tangential energy.
      if (node["gn"] || node["gt"]) {
        lerr << "Please, define only g (remove gt or gn).\n";
        return false;
      }
      // EnergyMixedMode: a single fracture criterion is used (En + Et > 2 * area * g). crit2 is unused.
      v.mode_ = exaDEM::RuptureMode::EnergyMixedMode;
      v.crit1_ = node["g"].as<Quantity>().convert();
      v.crit2_ = 0.0;
    } else if (node["gn"] && node["gt"]) {
      // EnergySeparateMode: normal and tangential fracture criteria are checked independently.
      v.mode_ = exaDEM::RuptureMode::EnergySeparateMode;
      v.crit1_ = node["gn"].as<Quantity>().convert();
      v.crit2_ = node["gt"].as<Quantity>().convert();
    }

    if (v.mode_ == exaDEM::RuptureMode::None) {
      lerr << "g or (gn and gt) is missing\n";
      return false;
    }

    return true;
  }
};
}  // namespace YAML
