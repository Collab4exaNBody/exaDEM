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

#include <exaDEM/drivers.hpp>
#include <exaDEM/shape_reader.hpp>

#include <yaml-cpp/yaml.h>
#include <string>

// Reads back a "drivers:" storage file written by the dump_drivers operator (see
// dump_drivers.cu) and registers each entry into a Drivers container -- directly, without going
// through the register_*/setup_drivers: operator-graph route (which write_op_drivers's
// setup_drivers:/register_*: .msp output is meant for instead).
namespace exaDEM {

/**
 * @brief Reads a drivers storage file and adds each driver it contains to `drivers`.
 * @param filename Path to a "drivers:" storage file, as written by dump_drivers.
 * @param drivers Drivers container to add the read-back drivers to.
 */
inline void read_drivers(const std::string& filename, Drivers& drivers) {
  YAML::Node root = YAML::LoadFile(filename);
  if (!root["drivers"]) {
    color_log::warning("read_drivers", "No 'drivers:' key found in " + filename);
    return;
  }

  int n_drivers = 0;
  for (const auto& entry : root["drivers"]) {
    if (!entry["type"] || !entry["id"]) {
      color_log::error("read_drivers", "Each driver entry needs a 'type' and an 'id'.");
    }
    const DRIVER_TYPE type = exaDEM::get_type(entry["type"].as<std::string>());
    const int id = entry["id"].as<int>();
    Driver_params params = entry["params"].as<Driver_params>();

    if (type == DRIVER_TYPE::SURFACE) {
      Surface driver = {entry["state"].as<SurfaceFields>(), params.input_motion_type_};
      driver.initialize(params);
      drivers.add_driver(id, driver, params);
    } else if (type == DRIVER_TYPE::CYLINDER) {
      Cylinder driver = {entry["state"].as<CylinderFields>(), params.input_motion_type_};
      driver.initialize(params);
      drivers.add_driver(id, driver, params);
    } else if (type == DRIVER_TYPE::BALL) {
      // Ball's constructor takes non-const lvalue refs, unlike the other 3 driver types'
      // aggregate init -- needs a named lvalue, can't bind directly to a temporary.
      BallFields state = entry["state"].as<BallFields>();
      Ball driver(state, params.input_motion_type_);
      driver.initialize(params);
      drivers.add_driver(id, driver, params);
    } else if (type == DRIVER_TYPE::RSHAPE) {
      if (!entry["filename"] || !entry["minkowski"]) {
        color_log::error("read_drivers", "An RSHAPE driver entry needs a 'filename' and a 'minkowski' value.");
      }
      // Mirrors register_stl_mesh's .shp construction sequence: fresh geometry + minkowski
      // radius applied on top (the shape geometry itself carries no minkowski radius).
      shape shp = exaDEM::read_shp(entry["filename"].as<std::string>(), /*big_shape=*/true);
      shp.add_radius(entry["minkowski"].as<double>());
      shp.increase_obb(shp.minkowski());

      RShapeDriver driver = {entry["state"].as<RShapeDriverFields>(), params.input_motion_type_};
      driver.shp_ = shp;
      driver.initialize(params);
      drivers.add_driver(id, driver, params);
    }
    n_drivers++;
  }
  exanb::ldbg << n_drivers << " drivers have been read from " << filename << std::endl;
}

}  // namespace exaDEM
