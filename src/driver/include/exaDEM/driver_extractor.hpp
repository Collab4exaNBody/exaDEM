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

#include <onika/flat_tuple.h>

#include <exaDEM/color_log.hpp>
#include <exaDEM/drivers.hpp>

namespace exaDEM {
namespace extractor {

// Macro defining all extractable fields from drivers with type, name, string representation, and index
// Used to generate enum and field types for driver data extraction
#define FIELD_LIST             \
  X(int, type, "type", 0)      \
  X(double, rx, "rx", 1)       \
  X(double, ry, "ry", 2)       \
  X(double, rz, "rz", 3)       \
  X(double, vx, "vx", 4)       \
  X(double, vy, "vy", 5)       \
  X(double, vz, "vz", 6)       \
  X(double, vrotx, "vrotx", 7) \
  X(double, vroty, "vroty", 8) \
  X(double, vrotz, "vrotz", 9) \
  X(double, fx, "fx", 10)      \
  X(double, fy, "fy", 11)      \
  X(double, fz, "fz", 12)

// Tuple type containing all extractable field types (driver properties that can be tracked)
using FieldsData = std::tuple<
#define X(type, name, str, idx) type,
    FIELD_LIST
#undef X
    void>;

// Enumeration of all extractable fields, each with a unique index
enum Dictionnary {
#define X(type, name, str, idx) name = idx,
  FIELD_LIST
#undef X
      COUNT  // Total number of fields
};

// Number of extractable fields
static constexpr size_t dict_size = static_cast<size_t>(Dictionnary::COUNT);

// Convert a field enum value to its string representation
inline std::string to_cstring(Dictionnary f) {
  switch (f) {
#define X(type, name, str, idx) \
  case Dictionnary::name:       \
    return str;
    FIELD_LIST
#undef X
    default:
      return "ERROR";
  }
  __builtin_unreachable();
}

// Convert a string to its corresponding field enum value
// Logs an error and terminates if the string doesn't match any known field
inline Dictionnary from_string(std::string_view s) {
#define X(type, name, str, idx) \
  if (s == str) return Dictionnary::name;
  FIELD_LIST
#undef X
  std::string msg = "Dictionnary: \"" + std::string(s) + "\" is not defined, please use: \n";
  for (size_t i = 0; i < dict_size; i++) {
    msg += to_cstring(static_cast<Dictionnary>(i)) + "\n";
  }
  color_log::error("Dictionnary::from_string", msg);
  __builtin_unreachable();
}

// Structure holding extracted field values from a driver
struct Fields {
#define X(type, name, str, idx) type name;  // Field member
  FIELD_LIST
#undef X
};

// Tracker specification: defines which fields to extract from a specific driver
struct Tracker {
  size_t id;                                   // Driver ID to track
  std::vector<extractor::Dictionnary> fields;  // List of fields to extract

  // Merge another tracker's fields into this one (union of field sets)
  void fuse(Tracker& in) {
    fields.insert(fields.end(), in.fields.begin(), in.fields.end());
    std::sort(fields.begin(), fields.end());
    fields.erase(std::unique(fields.begin(), fields.end()), fields.end());
  }

  // Print tracker information: driver ID and tracked fields
  void print() {
    using exanb::lout;
    lout << "Tracked Driver Id: " << id << " | ";
    if (fields.size() == 0) {
      lout << "No field are tracked" << std::endl;
    } else {
      lout << "Fields: [";
      for (auto& f : fields) {
        lout << " " << to_cstring(f);
      }
      lout << " ]" << std::endl;
    }
  }

  // Check if force fields (fx, fy, fz) are being tracked
  // (Force fields require interaction computation to be available)
  bool require_interaction() {
    for (auto& field : fields) {
      if (field == extractor::Dictionnary::fx || field == extractor::Dictionnary::fy ||
          field == extractor::Dictionnary::fz) {
        return true;
      }
    }
    return false;
  }
};
}  // namespace extractor

// Validate tracker compatibility with available drivers
// @param ded Tracker specification to validate
// @param drvs Available drivers
// @param error_msg Output: error description if validation fails
// @return true if tracker is compatible, false otherwise
inline bool compatibility(const extractor::Tracker& ded, Drivers& drvs, std::string& error_msg) {
  if (ded.id >= drvs.get_size()) {
    error_msg = "This Driver ID is not defined";
    return false;
  }

  // Add other exceptions here

  return true;
}

// Central manager for tracking and extracting driver field data during simulation
struct DriverExtractor {
  std::vector<extractor::Tracker> tracked_drivers;  // List of active trackers

  // Extract data from drivers according to registered trackers
  // (Implementation depends on extraction mechanism)
  void extract_driver_data(Drivers& driver) {}

  // Register a new tracker, merging with existing tracker for same driver if present
  void add(extractor::Tracker& tracker) {
    auto iterator = std::find_if(tracked_drivers.begin(), tracked_drivers.end(),
                                 [&tracker](extractor::Tracker& item) { return item.id == tracker.id; });
    if (iterator == tracked_drivers.end()) {
      tracked_drivers.push_back(tracker);
    } else {
      iterator->fuse(tracker);
    }
  }

  // Check if any tracked driver requires interaction computation
  // (Returns true if any tracker tracks force or moment fields)
  bool require_interaction() {
    for (auto& tracker : tracked_drivers) {
      if (tracker.require_interaction()) {
        return true;
      }
    }
    return false;
  }
};
}  // namespace exaDEM

// YAML conversion for Tracker objects
// Enables parsing tracker specifications from YAML configuration files
namespace YAML {
using exaDEM::extractor::Dictionnary;
using exaDEM::extractor::Tracker;

// YAML converter for Tracker structure
template <>
struct convert<Tracker> {
  // Decode YAML node into a Tracker object
  // Expected YAML format:
  //   id: <driver_id>
  //   fields: [field1, field2, ...]
  static bool decode(const Node& node, Tracker& tracker) {
    std::string function_name = "TrackedDriverExtractor::decode";
    // Validate node is a YAML map
    if (!node.IsMap()) {
      return false;
    }

    // Validate required 'id' field
    if (!node["id"]) {
      color_log::error(function_name, "id is missing.", false);
      return false;
    }

    // Validate required 'fields' field
    if (!node["fields"]) {
      color_log::error(function_name, "fields is missing.", false);
      return false;
    }

    // Parse driver ID
    tracker.id = node["id"].as<int>();

    // Parse field list and convert string names to enum values
    auto tmp = node["fields"].as<std::vector<std::string>>();
    for (auto& item : tmp) {
      tracker.fields.push_back(exaDEM::extractor::from_string(item));
    }
    return true;
  }
};
}  // namespace YAML