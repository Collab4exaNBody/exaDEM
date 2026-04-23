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
#include <exaDEM/driver_extractor.hpp>
#include <exaDEM/driver_extractor_impl.hpp>

namespace exaDEM {
template<typename GridT>
class DriverExtractorOp : public OperatorNode {

  ADD_SLOT(double, physical_time, INPUT, REQUIRED);
  ADD_SLOT(DriverExtractor, driver_extractor, INPUT, DocString{"Extract specific data about drivers."});
  ADD_SLOT(Drivers, drivers, INPUT, OPTIONAL, DocString{"List of Drivers"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator 

        No parameter.
        YAML example:

          - driver_extractor
        )EOF";
  }

  inline void execute() final {
    if (driver_extractor.has_value()) {
      auto& extractor = *driver_extractor;
      bool need_interaction = extractor.require_interaction();

      std::string header = "time ";
      for (auto& tracker : extractor.tracked_drivers) {
        std::string _id = "_" + std::to_string(tracker.id);
        for (auto& field : tracker.fields) {
          header += extractor::to_cstring(field) + _id + " ";
        }
      }
      lout << header << std::endl;

      std::string line = std::to_string(*physical_time) + " "; 
      auto& drvs = *drivers;

      for (auto& tracker : extractor.tracked_drivers) {
        DriverExtractFunc func = {"", tracker};
        drvs.apply(tracker.id, func);
        line += func.stream;
      }

      lout << line << std::endl;

      // specific implementation
      if (need_interaction) {
        // we compute moments and forces for every drivers
        std::vector<Vec3d> forces(drvs.get_size(), 0);
        std::vector<Vec3d> moments(drvs.get_size(), 0);

      }
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(driver_extractor) {
  OperatorNodeFactory::instance()->register_factory("driver_extractor", make_simple_operator<DriverExtractorOp>);
}
}  // namespace exaDEM
