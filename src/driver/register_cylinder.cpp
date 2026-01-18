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
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <exaDEM/drivers.hpp>

namespace exaDEM {

using namespace exanb;

class RegisterCylinder : public OperatorNode {
  const Driver_params default_params = Driver_params();

  ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, REQUIRED, DocString{"List of Drivers"});
  ADD_SLOT(int, id, INPUT, REQUIRED, DocString{"Driver index"});
  ADD_SLOT(Cylinder_params, state, INPUT, REQUIRED,
           DocString{
               "Current cylinder state, default is {radius: REQUIRED, axis: REQUIRED, center: REQUIRED, vel: [0,0,0], "
               "vrot: [0,0,0], rv: 0, ra: 0}. You need to specify the radius and center"});
  ADD_SLOT(Driver_params, params, INPUT, default_params,
           DocString{"List of params, motion type, motion vectors .... Default is { motion_type: STATIONARY}."});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator add a cylinder to the drivers list.
        )EOF";
  }

  inline void execute() final {
    // proj center over axis
    exaDEM::Cylinder driver{*state, *params};
    driver.initialize();
    drivers->add_driver(*id, driver);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(register_cylinder) {
  OperatorNodeFactory::instance()->register_factory("register_cylinder", make_simple_operator<RegisterCylinder>);
}
}  // namespace exaDEM
