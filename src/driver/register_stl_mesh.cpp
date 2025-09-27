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
#include <onika/math/quaternion.h>
#include <onika/math/quaternion_yaml.h>
#include <mpi.h>
#include <memory>
#include <exaDEM/driver_base.h>
#include <exaDEM/drivers.h>
#include <exaDEM/stl_mesh.h>
#include <exaDEM/shape.hpp>
#include <exaDEM/stl_mesh_to_driver.h>

namespace exaDEM
{

  using namespace exanb;

  class RegisterSTLMesh : public OperatorNode
  {
    const Driver_params default_params = Driver_params();
    static constexpr Stl_params default_stl_mesh_params = Stl_params();

    ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, REQUIRED, DocString{"List of Drivers"});
    ADD_SLOT(int, id, INPUT, REQUIRED, DocString{"Driver index"});
    ADD_SLOT(std::string, filename, INPUT, REQUIRED, DocString{"Input filename"});
    ADD_SLOT(Stl_params, state, INPUT, default_stl_mesh_params, DocString{"Define the center, velocity, angular velocity and the orientatation. Default is: state: {center: [0,0,0], vel: [0,0,0], vrot: [0,0,0], quat: [1,0,0,0]}."});
    ADD_SLOT(Driver_params, params, INPUT, default_params, DocString{"List of params, motion type, motion vectors .... Default is { motion_type: STATIONARY}."});
    ADD_SLOT(double, minskowski, INPUT, REQUIRED, DocString{"Minskowski radius value"});
    ADD_SLOT(bool, binary, INPUT, false, DocString{"Binary mode, it only works if the stl mesh is cmposed of triangles. Default is false."});
    ADD_SLOT(double, scale, INPUT, OPTIONAL, DocString{"Rescale your stl mesh"});
    ADD_SLOT(double, rcut_inc, INPUT, DocString{"value added to the search distance to update neighbor list less frequently. in physical space"});

  public:
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator add a stl mesh to the drivers list.
        )EOF";
    }

    inline std::string operator_name() { return "register_stl_mesh"; }

    inline void execute() override final
    {
      std::string output_name = *filename;
      std::string old_extension_stl = ".stl";
      std::string old_extension_shp = ".shp";
      bool is_stl(false), is_shp(false);
      std::string::size_type it_stl = output_name.find(old_extension_stl);
      std::string::size_type it_shp = output_name.find(old_extension_shp);
      if (it_stl != std::string::npos)
        is_stl = true;
      if (it_shp != std::string::npos)
        is_shp = true;
      if ((is_stl == false) && (is_shp == false))
      {
        color_log::error(operator_name(), "Wrong file extension, available formats: [shp or stl]");
      }
      if ((is_stl == true) && (is_shp == true))
      {
        color_log::error(operator_name(), "The file name contains the stings \"shp\" and \"stl\", impossible to deduce the file format.");
      }

      assert(is_stl != is_shp);
      // load shape
      shape shp;

      if (is_stl)
      {
        stl_mesh_reader reader;
        reader(*filename, *binary);
        output_name.erase(it_stl, old_extension_stl.length());
        shp = build_shape(reader, output_name);
      }
      else if (is_shp)
      {
        // not optimized
        bool big_shape = true;
        shp = read_shp(output_name, big_shape);
      }

      if(scale.has_value())
      {
        if( *scale != 1.0 && *scale > 0.0 )
        {
          shp.rescale(*scale, false);
          shp.write_paraview(); // replace
        }
        else
        {
          if( *scale <= 0.0 ) color_log::error("register_stl_mesh","Impossible to rescale the mesh, scale <= 0.0.");
          if( *scale == 1.0 ) color_log::warning("register_stl_mesh","rescale mesh option is ignored, scale = 1.0.");
        }
      }

      shp.m_radius = *minskowski;
      //shp.increase_obb(*rcut_inc);
      shp.increase_obb(shp.m_radius);
      exaDEM::Stl_mesh driver = {*state, *params};
      driver.set_shape(shp);
      driver.initialize();
      drivers->add_driver(*id, driver);
      lout << "=================================" << std::endl;
    }
  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(register_stl_mesh) { OperatorNodeFactory::instance()->register_factory("register_stl_mesh", make_simple_operator<RegisterSTLMesh>); }
} // namespace exaDEM
