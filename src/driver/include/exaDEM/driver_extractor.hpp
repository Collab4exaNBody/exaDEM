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

namespace exaDEM {
namespace extractor {

#define FIELD_LIST \
  X(int,    type, "type", 0) \
  X(double, rx,   "rx",   1) \
  X(double, ry,   "ry",   2) \
  X(double, rz,   "rz",   3) \
  X(double, vx,   "vx",   4) \
  X(double, vy,   "vy",   5) \
  X(double, vz,   "vz",   6) \
  X(double, vrotx,"vrotx",7) \
  X(double, vroty,"vroty",8) \
  X(double, vrotz,"vrotz",9) \
  X(double, fx,   "fx",   10) \
  X(double, fy,   "fy",   11) \
  X(double, fz,   "fz",   12)

using FieldsData = std::tuple<
#define X(type, name, str, idx) type,
  FIELD_LIST
#undef X
  void
>;

enum Dictionnary {
#define X(type, name, str, idx) name = idx,
  FIELD_LIST
#undef X
  COUNT
};

static constexpr size_t dict_size = static_cast<size_t>(Dictionnary::COUNT);

inline std::string to_cstring(Dictionnary f) {
  switch (f) {
#define X(type, name, str, idx) case Dictionnary::name: return str;
    FIELD_LIST
#undef X
  }
  __builtin_unreachable();
}

inline Dictionnary from_string(std::string_view s) {
#define X(type, name, str, idx) if (s == str) return Dictionnary::name;
  FIELD_LIST
#undef X
  std::string msg = "Dictionnary: \"" + std::string(s) + "\" is not defined, please use: \n";
  for (size_t i = 0 ; i < dict_size ; i++) {
    msg += to_cstring(static_cast<Dictionnary>(i)) + "\n";
  } 
  color_log::error("Dictionnary::from_string", msg);
  __builtin_unreachable();
}

struct Fields {
#define X(type, name, str, idx) type name;
  FIELD_LIST
#undef X
};

struct Tracker {
  size_t id;
  std::vector<extractor::Dictionnary> fields;

  void fuse(Tracker& in) {
    fields.insert(fields.end(), in.fields.begin(), in.fields.end());
    std::sort(fields.begin(), fields.end());
    fields.erase(std::unique(fields.begin(), fields.end()), fields.end());
  }

  void print() {
    lout << "Tracked Driver Id: " << id << " | ";
    if (fields.size() == 0) {
      lout << "No field are tracked" << std::endl;
    } else {
      lout << "Fields: [";
      for (auto& f: fields) {
        lout << " " << to_cstring(f);
      }
      lout << " ]" << std::endl;
    }
  }

  bool require_interaction() {
    for (auto& field : fields) {
      if (field == extractor::Dictionnary::fx ||
          field == extractor::Dictionnary::fx ||
          field == extractor::Dictionnary::fz) {
        return true;
      }
    }
    return false;
  }
};
}

inline bool compatibility(const extractor::Tracker& ded, Drivers& drvs, std::string& error_msg) {

  if (ded.id >= drvs.get_size()) {
    error_msg = "This Driver ID is not defined";
    return false;
  }

  // Add other exceptions here

  return true;
}

struct DriverExtractor {
  std::vector<extractor::Tracker> tracked_drivers;

  void extract_driver_data(Drivers& driver) { }

  void add(extractor::Tracker& tracker) {
    auto iterator = std::find_if(tracked_drivers.begin(), tracked_drivers.end(),
                                 [&tracker] (extractor::Tracker& item) {
                                 return item.id == tracker.id;
                                 });
    if (iterator == tracked_drivers.end()) {
      tracked_drivers.push_back(tracker);
    } else {
      iterator->fuse(tracker);
    }
  }

  void summary() {
    lout << "======= Driver Extractor ========" << std::endl;
    for (auto& it: tracked_drivers) {
      it.print();
    }
    lout << "=================================" << std::endl;
  }

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

namespace YAML {
using exaDEM::extractor::Tracker;
using exaDEM::extractor::Dictionnary;

template <>
struct convert<Tracker> {
  static bool decode(const Node& node, Tracker& tracker) {
    std::string function_name = "TrackedDriverExtractor::decode";
    if (!node.IsMap()) {
      return false;
    }

    if (!node["id"]) {
      color_log::error(function_name, "id is missing.", false);
      return false;
    }

    if (!node["fields"]) {
      color_log::error(function_name, "fields is missing.", false);
      return false;
    }
    tracker.id = node["id"].as<int>();
    auto tmp = node["fields"].as<std::vector<std::string>>();
    for(auto& item: tmp) {
      tracker.fields.push_back(exaDEM::extractor::from_string(item));
    }
    return true;
  }
};
}
