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

struct PushFVRDriverFunc {
  double time;
  double delta_t;
  Driver_params& motion;
  template<typename VectorT>
  void operator()(VectorT& driver) {
    if (motion.is_motion_triggered(time + delta_t)) {
      driver.push_f_v_r(motion, time, delta_t);
    }
  }
};

class PushAccVelocityToPositionDriver : public OperatorNode {
  ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, REQUIRED, DocString{"List of Drivers"});
  ADD_SLOT(double, physical_time, INPUT, REQUIRED);
  ADD_SLOT(double, dt, INPUT, REQUIRED, DocString{"dt is the time increment of the timeloop"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
          This operator updates driver centers using their velocities. Not that accelerations are not used.
        )EOF";
  }

  inline void execute() final {
    auto& drvs = *drivers;
    for (size_t id = 0; id < drivers->get_size(); id++) {
      PushFVRDriverFunc func = {*physical_time, *dt, drvs.get_motion(id)};
      drvs.apply(id, func);
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(push_f_v_r_driver) {
  OperatorNodeFactory::instance()->register_factory("push_f_v_r_driver",
                                                    make_simple_operator<PushAccVelocityToPositionDriver>);
}
}  // namespace exaDEM
