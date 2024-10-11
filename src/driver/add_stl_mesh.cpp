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
#include "exanb/core/operator.h"
#include "exanb/core/operator_slot.h"
#include "exanb/core/operator_factory.h"
#include <exanb/core/quaternion.h>
#include <exanb/core/quaternion_yaml.h>
#include <mpi.h>
#include <memory>
#include <exaDEM/driver_base.h>
#include <exaDEM/drivers.h>
#include <exaDEM/stl_mesh.h>
#include <exaDEM/shape/shape.hpp>
#include <exaDEM/stl_mesh_to_driver.h>

namespace exaDEM
{

	using namespace exanb;

		class AddSTLMesh : public OperatorNode
	{
		static constexpr Vec3d null= { 0.0, 0.0, 0.0 };
		static constexpr Quaternion default_quat= { 1.0 , 0.0, 0.0, 0.0 };

    ADD_SLOT( Drivers     , drivers         , INPUT_OUTPUT, REQUIRED , DocString{"List of Drivers"});
		ADD_SLOT( int         , id              , INPUT       , REQUIRED , DocString{"Driver index"});
		ADD_SLOT( std::string , filename        , INPUT       , REQUIRED , DocString{"Input filename"});
		ADD_SLOT( Vec3d       , center          , INPUT       , null     , DocString{"Defined but not used"});
		ADD_SLOT( Vec3d       , angular_velocity, INPUT       , null     , DocString{"Defined but not used"});
		ADD_SLOT( Vec3d       , velocity        , INPUT       , null     , DocString{"Defined but not used"});
		ADD_SLOT( Quaternion  , orientation     , INPUT       , default_quat , DocString{"Defined but not used"});
    ADD_SLOT( double      , minskowski      , INPUT       , REQUIRED , DocString{"Minskowski radius value"} );
    ADD_SLOT( double      , rcut_inc        , INPUT       , DocString{"value added to the search distance to update neighbor list less frequently. in physical space"} );

		public:

		inline std::string documentation() const override final
		{
			return R"EOF(
        This operator add a stl mesh to the drivers list.
        )EOF";
		}

		inline void execute () override final
		{
			stl_mesh_reader reader;
			reader(*filename);
			std::string output_name_vtk = *filename;
			std::string old_extension_stl = ".stl";
			std::string old_extension_shp = ".shp";
      bool is_stl(false), is_shp(false);
			std::string::size_type it_stl = output_name_vtk.find(old_extension_stl);
			std::string::size_type it_shp = output_name_vtk.find(old_extension_shp);
      if (it_stl != std::string::npos) is_stl = true;
      if (it_shp != std::string::npos) is_shp = true;
      if( (is_stl == false) && (is_shp == false)) 
      {
         lout << "Wrong file extension used in add_stl_mesh, available formats: [shp or stl]" << std::endl; 
        std::abort();
      }
      if( (is_stl == true) && (is_shp == true)) 
      { 
        lout << "Error when checking file name, the file name contains shp and stl and this operator can't deduce the file format." << std::endl; 
        std::abort();
      }

      assert(is_stl != is_shp);

      // load shape
			shape shp; 

			if (is_stl)
			{
				output_name_vtk.erase(it_stl, old_extension_stl.length());
        shp = build_shape(reader, output_name_vtk);
			}
			else if (is_shp)
			{
				//output_name_vtk.erase(it, old_extension_shp.length());
        // not optimized
        bool big_shape = true;
        shp = read_shp(shp, output_name_vtk, big_shape);
			}

			shp.m_radius = *minskowski;
			shp.increase_obb (*rcut_inc);
			exaDEM::Stl_mesh driver= {*center, *velocity, *angular_velocity, *orientation, shp};
			drivers->add_driver(*id, driver);
			lout << "=================================" << std::endl;
		}
	};

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "add_stl_mesh", make_simple_operator< AddSTLMesh > );
	}
}

