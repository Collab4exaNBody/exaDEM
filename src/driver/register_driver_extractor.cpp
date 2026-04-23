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

namespace exaDEM {
class RegisterDriverExtractor : public OperatorNode {

  ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, REQUIRED, DocString{"List of Drivers"});
  ADD_SLOT(std::vector<extractor::Tracker>, trackers, INPUT,
           REQUIRED, DocString{"List of trackers. format [{id: driver_id_1, fields: [field1, field2 ...],\
           {id: driver_id_2, fields: [field1, field2, ...], ..."});
  ADD_SLOT(DriverExtractor, driver_extractor, INPUT_OUTPUT, DriverExtractor{}, DocString{"Extract specific data about drivers."});
  ADD_SLOT(bool, verbosity, false, PRIVATE, DocString{"Display trackers."});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator add a surface to the drivers list.
        )EOF";
  }

  inline void execute() final {
    std::vector<extractor::Tracker>& Trackers = *trackers;
    if (*verbosity) {
      lout << "Add tracker: " << std::endl;
    }
    for(auto& tracker: Trackers) {
      std::string error_msg = "unknown";
      if (!compatibility(tracker, *drivers, error_msg)) {
        lout << "The following tracker is not compatible" << std::endl;
        lout << "Reason: " << error_msg << std::endl;
        tracker.print();
      }
      if (*verbosity) {
        tracker.print();
      }
      driver_extractor->add(tracker);
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(register_driver_extractor) {
  OperatorNodeFactory::instance()->register_factory("register_driver_extractor", make_simple_operator<RegisterDriverExtractor>);
}
}  // namespace exaDEM
