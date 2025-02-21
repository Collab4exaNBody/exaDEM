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
#include <exaDEM/drivers.h>

namespace exaDEM
{
  using namespace exanb;

  struct push_f_v_r
  {
    const double dt;
    void operator()(Ball& arg)
    {
      arg.push_f_v(dt);
    }

    void operator()(Surface& arg)
    {
      arg.push_f_v(dt);
    }

    void operator()(Stl_mesh& arg)
    {
      arg.push_f_v(dt);
    }
    void operator()(auto&& arg)
    {
      /** nothing */
    }
  };

  class PushAccelToVelocityDriver : public OperatorNode
  {
    ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, REQUIRED, DocString{"List of Drivers"});
    ADD_SLOT(double, dt, INPUT, DocString{"dt is the time increment of the timeloop"});

  public:
    inline std::string documentation() const override final
    {
      return R"EOF(
          This operator updates driver centers using their velocities. Not that accelerations are not used.
        )EOF";
    }

    inline void execute() override final
    {
      double t = *dt;
      push_f_v_r func = {t};
      for (size_t id = 0; id < drivers->get_size(); id++)
      {
        auto &driver = drivers->data(id);
        std::visit(func, driver);
      }
    }
  };

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("push_f_v_driver", make_simple_operator<PushAccelToVelocityDriver>); }
} // namespace exaDEM
