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

#include <exaDEM/driver_reader.hpp>
#include <exaDEM/drivers.hpp>

namespace exaDEM {
using namespace exanb;

class ReadDrivers : public OperatorNode {
  ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, REQUIRED, DocString{"List of Drivers"});
  ADD_SLOT(std::string, filename, INPUT, REQUIRED,
           DocString{"Path to a drivers storage file, as written by dump_drivers."});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        Reads a drivers storage file (written by dump_drivers) and registers each driver it
        contains -- restoring drivers at restart time without going through
        setup_drivers:/register_*: operators or a top-level includes:.

        YAML example:

          - read_drivers:
             filename: ExaDEMOutputDir/CheckpointFiles/drivers_0001400000.msp
      )EOF";
  }

  inline void execute() final { exaDEM::read_drivers(*filename, *drivers); }
};

// === register factories ===
ONIKA_AUTORUN_INIT(read_drivers) {
  OperatorNodeFactory::instance()->register_factory("read_drivers", make_simple_operator<ReadDrivers>);
}
}  // namespace exaDEM
