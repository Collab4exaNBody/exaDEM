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
#include <exanb/core/domain.h>
#include <mpi.h>
#include <memory>
#include <exaDEM/stl_mesh.h>
#include <exaDEM/drivers.h>
#include <exaDEM/paraview_driver.hpp>
#include <exanb/core/string_utils.h>


namespace exaDEM
{

  using namespace exanb;

  class ParaviewDriver : public OperatorNode
  {
    static constexpr Vec3d null = {0.0, 0.0, 0.0};

    ADD_SLOT(Domain, domain, INPUT, REQUIRED);
    ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, REQUIRED, DocString{"List of Drivers"});
    ADD_SLOT(long, timestep, INPUT, DocString{"Iteration number"});
    ADD_SLOT(std::string, dir_name, INPUT, REQUIRED, DocString{"Main output directory."});

  public:
    inline std::string documentation() const override final { return R"EOF( This operator creates a parview file of stl meshes. )EOF"; }

    inline void execute() override final
    {
      std::string path = *dir_name + "/ParaviewOutputFiles/";

      std::vector<info_ball> balls;
      std::vector<info_surface> surfaces;
      for (size_t id = 0; id < drivers->get_size(); id++)
      {
        if (drivers->type(id) == DRIVER_TYPE::BALL)
        {
          exaDEM::Ball &ball = std::get<exaDEM::Ball>(drivers->data(id));
          balls.push_back({int(id), ball.center, ball.radius, ball.vel});
        }
        if (drivers->type(id) == DRIVER_TYPE::SURFACE)
        {
          exaDEM::Surface &surface = std::get<exaDEM::Surface>(drivers->data(id));
          surfaces.push_back({int(id), surface.normal, surface.offset, surface.vel});
        }
        if (drivers->type(id) == DRIVER_TYPE::STL_MESH)
        {
          exaDEM::Stl_mesh &mesh = std::get<exaDEM::Stl_mesh>(drivers->data(id));
          mesh.shp.write_move_paraview(path, *timestep, mesh.center, mesh.quat);
        }
      }
 
      if( balls.size() > 0 )
      {
        std::filesystem::path dir(path);
        std::string driver_ball_name = "driver_balls_%010d.vtk";
        driver_ball_name = format_string(driver_ball_name,  *timestep);
        write_balls_paraview(balls, path, driver_ball_name);
      }
      if( surfaces.size() > 0 )
      {
        std::filesystem::path dir(path);
        std::string driver_surface_name = "driver_surfaces_%010d.vtk";
        driver_surface_name = format_string(driver_surface_name,  *timestep);
        write_surfaces_paraview(*domain, surfaces, path, driver_surface_name);
      }
    }
  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(paraview_driver) { OperatorNodeFactory::instance()->register_factory("paraview_driver", make_simple_operator<ParaviewDriver>); }
} // namespace exaDEM
