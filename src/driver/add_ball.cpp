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
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exaDEM/driver_base.h>
#include <exaDEM/drivers.h>
#include <exaDEM/ball.h>

namespace exaDEM
{

  using namespace exanb;

  class AddBall : public OperatorNode
  {
    static constexpr Vec3d null = {0.0, 0.0, 0.0};
    static constexpr Driver_params default_params = Driver_params();

    ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, REQUIRED, DocString{"List of Drivers"});
    ADD_SLOT(int, id, INPUT, REQUIRED, DocString{"Driver index"});
    ADD_SLOT(double, radius, INPUT, REQUIRED, DocString{"Radius of the ball, positive and should be superior to the biggest sphere radius in the ball"});
    ADD_SLOT(Vec3d, center, INPUT, REQUIRED, DocString{"Center of the ball"});
    ADD_SLOT(Vec3d, velocity, INPUT, null, DocString{"Ball velocity"});
    ADD_SLOT(Vec3d, vrot, INPUT, null, DocString{"Angular velocity of the ball, default is 0 m.s-"});
    ADD_SLOT(Driver_params, params, INPUT, default_params, DocString{"List of params, motion type, motion vectors ... "});

  public:
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator add a ball (boundary condition) to the drivers list.
        )EOF";
    }

    inline void execute() override final
    {
      exaDEM::Ball driver = {*radius, *center, *velocity, *vrot};
      driver.set_params(*params);
      driver.initialize();
      drivers->add_driver(*id, driver);
    }
  };

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("add_ball", make_simple_operator<AddBall>); }
} // namespace exaDEM
