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

#include <exaDEM/compute_drivers.hpp>

namespace exaDEM {
using namespace onika::scg;
using onika::lout;

struct ResetForceDriverFunc {
  template <typename DriverT>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(DriverT& drv) const {
    drv.forces() = Vec3d{0, 0, 0};
  }
};

class ResetForceDriverFunctor : public OperatorNode {
  ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, REQUIRED, DocString{"List of Drivers"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
          This operator reset the forces of drivers.
        )EOF";
  }

  inline void execute() final {
    ResetForceDriverFunc func;
    compute_drivers(*drivers, func, parallel_execution_context("reset_force_driver"));
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(reset_force_driver) {
  OperatorNodeFactory::instance()->register_factory("reset_force_driver",
                                                    make_simple_operator<ResetForceDriverFunctor>);
}
}  // namespace exaDEM
