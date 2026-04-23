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
class DriverExtractorSummary : public OperatorNode {

  ADD_SLOT(DriverExtractor, driver_extractor, INPUT, DocString{"Extract specific data about drivers."});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator displays the driver extractor summary.

        No parameter.
        YAML example:

          - driver_extractor_summary
        )EOF";
  }

  inline void execute() final {
    if (driver_extractor.has_value()) {
      driver_extractor->summary();
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(driver_extractor_summary) {
  OperatorNodeFactory::instance()->register_factory("driver_extractor_summary", make_simple_operator<DriverExtractorSummary>);
}
}  // namespace exaDEM
