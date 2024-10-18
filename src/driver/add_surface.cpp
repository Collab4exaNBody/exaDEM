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
#include <exaDEM/surface.h>

namespace exaDEM
{

  using namespace exanb;

  class AddSurface : public OperatorNode
  {
    ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, REQUIRED, DocString{"List of Drivers"});
    ADD_SLOT(size_t, id, INPUT, REQUIRED, DocString{"Driver index"});
    ADD_SLOT(double, offset, INPUT, 0.0, DocString{"Offset from the origin (0,0,0) of the rigid surface"});
    ADD_SLOT(double, velocity, INPUT, 0.0, DocString{"Surface velocity"});
    ADD_SLOT(Vec3d, center, INPUT, Vec3d{0.0, 0.0, 0.0}, DocString{"Normal vector of the rigid surface"});
    ADD_SLOT(Vec3d, normal, INPUT, Vec3d{0.0, 0.0, 1.0}, DocString{"Normal vector of the rigid surface"});
    ADD_SLOT(Vec3d, vrot, INPUT, Vec3d{0.0, 0.0, 0.0}, DocString{"Angular velocity of the surface, default is 0 m.s-"});

  public:
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator add a surface to the drivers list.
        )EOF";
    }

    inline void execute() override final
    {
      exaDEM::Surface driver = {*offset, *normal, *center, *velocity, *vrot}; //
      driver.initialize();                                                    // initialize some values from input parameters such as the projected center of the surface (normal line)
      drivers->add_driver(*id, driver);
    }
  };

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("add_surface", make_simple_operator<AddSurface>); }
} // namespace exaDEM
