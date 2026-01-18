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
#include <onika/math/quaternion.h>
#include <onika/math/quaternion_yaml.h>

#include <exaDEM/drivers.hpp>

namespace exaDEM {
class ModifyMotionBehavior : public OperatorNode {
  ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, REQUIRED, DocString{"List of Drivers"});
  ADD_SLOT(int, id, INPUT, REQUIRED, DocString{"Driver index"});
  ADD_SLOT(Driver_params, motion, INPUT, REQUIRED,
           DocString{"List of params, motion type, motion vectors .... Example is { motion_type: STATIONARY}."});
  ADD_SLOT(bool, display, INPUT, false, DocString{"Print the new motion type detail."});
  ADD_SLOT(double, time, INPUT, REQUIRED, DocString{"Time to change behavior."});
  ADD_SLOT(double, physical_time, INPUT, REQUIRED, DocString{"Current physical time."});
  ADD_SLOT(double, dt, INPUT, REQUIRED, DocString{"dt is the time increment of the timeloop"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator modifies the motion type of a driver.

        YAML example:

          driver_motion_policy:
            - modify_motion:
               id: 0
               time: 0.5
               motion: { motion_type: LINEAR_COMPRESSIVE_MOTION, motion_vector: [1,0,0], sigma: 5, damprate: 0.999 }
               display: true
            - modify_motion:
               id: 0
               time: 1.5
               motion: { motion_type: STATIONARY }
            - modify_motion:
               id: 0
               time: 2.5
               motion: { motion_type: LINEAR_COMPRESSIVE_MOTION, motion_vector: [1,0,0], sigma: 5, damprate: 0.999 }
            - modify_motion:
               id: 0
               time: 4.0
               motion: { motion_type: LINEAR_MOTION, motion_vector: [1,0,0], const_vel: 3 }

        )EOF";
  }

  inline std::string operator_name() { return "modify_motion"; }

  inline void execute() final {
    ldbg << "time " << *time << " physical_time " << *physical_time << std::endl;
    if (std::abs(double(*time) - double(*physical_time)) > (*dt / 2)) {
      return;
    }

    Driver_params& new_motion = *motion;
    Drivers& drvs = *drivers;

    new_motion.check_motion_coherence();
    std::string msg = "Change the motion type of the driver: [" + std::to_string(*id);
    msg += "] to : [";
    msg += motion_type_to_string(new_motion.motion_type);
    msg += "]";
    color_log::highlight(operator_name(), msg);
    if (*display) {
      new_motion.print_driver_params();
    }

    auto set_motion_type = [&new_motion](auto& d) -> void {
      d.set_params(new_motion);
    };
    drvs.apply(*id, set_motion_type);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(modify_motion) {
  OperatorNodeFactory::instance()->register_factory("modify_motion", make_simple_operator<ModifyMotionBehavior>);
}
}  // namespace exaDEM
