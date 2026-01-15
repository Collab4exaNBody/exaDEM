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
 * @struct InnerBondParams
 * @brief Encapsulates contact mechanics parameters for a contact interaction model.
 */
struct InnerBondParams {
  double kn = 0.0;         /**< Normal stiffness coefficient (force per unit displacement in the normal direction). */
  double kt = 0.0;         /**< Tangential stiffness coefficient (force per unit displacement in the tangential direction). */
  double damp_rate = 0.0;
  double g  = 0.0;
};

/**
 * @brief Equality operator for comparing two InnerBondParams instances.
 *
 * @param a First InnerBondParams instance.
 * @param b Second InnerBondParams instance.
 * @return true if all fields of both instances are equal; false otherwise.
 */
inline bool operator==(InnerBondParams& a, InnerBondParams& b) {
  return (a.kn == b.kn) &&
      (a.kt == b.kt) &&
      (a.damp_rate == b.damp_rate) &&
      (a.g == b.g);
}

/**
 * @brief Displays the header for a parameter table.
 */
template<typename CPT> inline void display_header();

/**
 * @brief Displays the footer or closing line for a parameter table.
 */
template<typename CPT> inline void display_end_table();

/**
 * @brief Displays a formatted header line for a table of InnerBondParams entries.
 */
template<> inline
void display_header<InnerBondParams>() {
  lout << "===============================================================================" << std::endl;
  lout << "|        typeA |        typeB |        kn |        kt | damp_rate |         g |" << std::endl;
  lout << "-------------------------------------------------------------------------------" << std::endl;
}

/**
 * @brief Displays a formatted footer line for a table of InnerBondParams entries.
 */
template<> inline
void display_end_table<InnerBondParams>() {
  lout << "===============================================================================" << std::endl;
}

/**
 * @brief Formats the contact parameters into a single table row string.
 */
inline std::string display(InnerBondParams& params) {
  std::string line = onika::format_string(
      " %.3e | %.3e | %.3e | %.3e |", 
      params.kn, 
      params.kt, 
      params.damp_rate, 
      params.g); 
  return line; 
}

/**
 * @brief Displays a full formatted line in the contact parameters table.
 *
 * @param typeA Identifier for the first type (e.g., material A).
 * @param typeB Identifier for the second type (e.g., material B).
 * @param params Reference to the InnerBondParams instance to display.
 */
inline void display_multimat(std::string typeA, std::string typeB, InnerBondParams& params) {
  std::string line_types = onika::format_string("| %12s | %12s |", typeA, typeB); 
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
template<typename STREAM>
void streaming(STREAM& stream, InnerBondParams& params) {
  stream << "{ kn: " << params.kn << 
      ", kt: " << params.kt << 
      ", damp_rate: " << params.damp_rate << 
      ", g: " << params.g << 
      " }";
}
} // namespace exaDEM

// Yaml conversion operators, allows to read potential parameters from config file
namespace YAML {
using exaDEM::InnerBondParams;
using exanb::lerr;
using onika::physics::Quantity;

template <> struct convert<InnerBondParams> {
static bool decode(const Node &node, InnerBondParams &v) {
  v = InnerBondParams{}; // initializes defaults values
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
  if (!node["g"]) {
    lerr << "g is missing\n";
    return false;
  }

  v.kn = node["kn"].as<Quantity>().convert();
  v.kt = node["kt"].as<Quantity>().convert();
  v.damp_rate = node["damp_rate"].as<Quantity>().convert();
  v.g = node["g"].as<Quantity>().convert();

  return true;
}
};
} // namespace YAML
