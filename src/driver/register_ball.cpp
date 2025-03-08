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
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exaDEM/driver_base.h>
#include <exaDEM/drivers.h>
#include <exaDEM/ball.h>

namespace exaDEM
{

  using namespace exanb;

  class RegisterBall : public OperatorNode
  {
    static constexpr Driver_params default_params = Driver_params();

    ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, REQUIRED, DocString{"List of Drivers"});
    ADD_SLOT(int, id, INPUT, REQUIRED, DocString{"Driver index"});
    ADD_SLOT(Ball_params, state, INPUT, REQUIRED, DocString{"Current ball state, default is {radius: REQUIRED, center: REQUIRED, vel: [0,0,0], vrot: [0,0,0], rv: 0, ra: 0}. You need to specify the radius and center"});
    ADD_SLOT(Driver_params, params, INPUT, default_params, DocString{"List of params, motion type, motion vectors .... Default is { motion_type: STATIONARY}."});

  public:
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator add a ball (boundary condition) to the drivers list.
        )EOF";
    }

    inline void execute() override final
    {
      exaDEM::Ball driver(*state, *params);
      driver.initialize();
      drivers->add_driver(*id, driver);
    }
  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(register_ball) { OperatorNodeFactory::instance()->register_factory("register_ball", make_simple_operator<RegisterBall>); }
} // namespace exaDEM
