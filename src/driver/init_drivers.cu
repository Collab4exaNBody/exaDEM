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
#include <mpi.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <exaDEM/drivers.hpp>

namespace exaDEM {
class InitDrivers : public OperatorNode {
  ADD_SLOT(Drivers, drivers, OUTPUT, DocString{"List of Drivers"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator creates a slot for drivers.
        )EOF";
  }

  inline void execute() final {
    // do nothing
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(init_drivers) {
  OperatorNodeFactory::instance()->register_factory("init_drivers", make_simple_operator<InitDrivers>);
}
}  // namespace exaDEM
