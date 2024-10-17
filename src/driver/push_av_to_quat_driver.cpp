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
#include <exaDEM/drivers.h>
#include <exaDEM/stl_mesh.h>

namespace exaDEM
{

  using namespace exanb;

  struct func_push_av_to_quat
  {
    double t;
    template <typename T> inline void operator()(T &&arg)
    { /* do nothing */
    }
    inline void operator()(exaDEM::Stl_mesh &arg) { arg.push_av_to_quat(t); }
  };

  class PushAngularVelocityToQuaternionDriver : public OperatorNode
  {
    ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, REQUIRED, DocString{"List of Drivers"});
    ADD_SLOT(double, dt, INPUT, DocString{"dt is the time increment of the timeloop"});

  public:
    inline std::string documentation() const override final { return R"EOF( This operator compute the new orientation using the angular velocity. )EOF"; }

    inline void execute() override final
    {
      double t = *dt;
      func_push_av_to_quat func = {t};
      for (size_t id = 0; id < drivers->get_size(); id++)
      {
        auto &driver = drivers->data(id);
        std::visit(func, driver);
      }
    }
  };

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("push_av_to_quat_driver", make_simple_operator<PushAngularVelocityToQuaternionDriver>); }
} // namespace exaDEM
