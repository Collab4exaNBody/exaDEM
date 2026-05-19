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

#include <exaDEM/driver_extractor.hpp>
#include <exaDEM/driver_extractor_impl.hpp>
#include <exaDEM/drivers.hpp>

namespace exaDEM {
using namespace onika::scg;
class DriverExtractorRequireForceMomentOp : public OperatorNode {
  ADD_SLOT(DriverExtractor, driver_extractor, INPUT, OPTIONAL, DocString{"Extract specific data about drivers."});
  ADD_SLOT(bool, trigger_driver_force_moment_update, OUTPUT,
           DocString{"Trigger to update forces and moments of drivers."});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator defines a trigger to update forces and moments of drivers. It should be used in combination with the 'driver_extractor' operator, which extracts specific data about drivers based on defined trackers.

        No parameter.
        YAML example:

          - driver_extractor_require_force_moment
        )EOF";
  }

  inline void execute() final {
    using exanb::Vec3d;
    if (driver_extractor.has_value()) {
      if (driver_extractor->require_interaction()) {
        *trigger_driver_force_moment_update = true;
      } else {
        *trigger_driver_force_moment_update = false;
      }
    } else {
      *trigger_driver_force_moment_update = false;
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(driver_extractor_require_force_moment) {
  OperatorNodeFactory::instance()->register_factory("driver_extractor_require_force_moment",
                                                    make_simple_operator<DriverExtractorRequireForceMomentOp>);
}
}  // namespace exaDEM