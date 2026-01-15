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
  struct func_push_av_to_quat
  {
    const double t;
    template <typename T>
    inline void operator()(T& drv) const
    {
       if constexpr ( std::is_same_v< std::remove_cv_t<T> , exaDEM::Stl_mesh > )
       {
         drv.push_av_to_quat(t);
       }
    }
  };

  class PushAngularVelocityToQuaternionDriver : public OperatorNode
  {
    ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, REQUIRED, DocString{"List of Drivers"});
    ADD_SLOT(double, dt, INPUT, REQUIRED, DocString{"dt is the time increment of the timeloop"});

    public:
    inline std::string documentation() const override final { 
      return R"EOF( 
      This operator compute the new orientation using the angular velocity. 

      YAML example [no option]:

        - push_av_to_quat_driver
      )EOF"; 
    }

    inline void execute() override final
    {
      double t = *dt;
      func_push_av_to_quat func = {t};
      for (size_t id = 0; id < drivers->get_size(); id++)
      {
        drivers->apply( id , func );
      }
    }
  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(push_av_to_quat_driver) { OperatorNodeFactory::instance()->register_factory("push_av_to_quat_driver", make_simple_operator<PushAngularVelocityToQuaternionDriver>); }
} // namespace exaDEM
