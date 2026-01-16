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
class BackupDrivers : public OperatorNode {
  ADD_SLOT(Drivers, drivers, INPUT, REQUIRED, DocString{"List of Drivers"});
  ADD_SLOT(Drivers, backup_drvs, INPUT_OUTPUT, Drivers(), DocString{"List of backup Drivers"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator creates a copy of the current drivers.
        )EOF";
  }

  inline void execute() final {
    Drivers& drvs = *drivers;
    Drivers& backup = *backup_drvs;
    backup.clear();
    backup = drvs;  // deep copy
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(backup_drivers) {
  OperatorNodeFactory::instance()->register_factory("backup_drivers", make_simple_operator<BackupDrivers>);
}
}  // namespace exaDEM
