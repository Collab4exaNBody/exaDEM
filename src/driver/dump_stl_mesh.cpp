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
//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE
#include "exanb/core/operator.h"
#include "exanb/core/operator_slot.h"
#include "exanb/core/operator_factory.h"
#include "exanb/core/make_grid_variant_operator.h"
#include "exanb/core/parallel_grid_algorithm.h"
#include "exanb/core/grid.h"
#include "exanb/core/domain.h"
#include "exanb/compute/compute_cell_particles.h"
#include <mpi.h>
#include <memory>
#include <exaDEM/stl_mesh.h>
#include <exaDEM/drivers.h>

namespace exaDEM
{

	using namespace exanb;

	template<typename GridT>
		class DumpSTLMesh : public OperatorNode
	{
		static constexpr Vec3d null= { 0.0, 0.0, 0.0 };

		ADD_SLOT( Drivers     , drivers  , INPUT_OUTPUT, REQUIRED , DocString{"List of Drivers"});
		ADD_SLOT( long        , timestep , INPUT                  , DocString{"Iteration number"});
    ADD_SLOT( std::string , dir_name , INPUT , REQUIRED , DocString{"Main output directory."} );

		public:

		inline std::string documentation() const override final
		{
			return R"EOF( This operator outputs driver information. )EOF";
		}

		inline void execute () override final
		{
      std::string path = *dir_name + "/ParaviewOutputFiles/";
			for(size_t id = 0 ; id < drivers->get_size() ; id++)
			{
				if ( drivers->type(id) == DRIVER_TYPE::STL_MESH)
				{
					exaDEM::Stl_mesh& mesh = std::get<exaDEM::Stl_mesh> (drivers->data(id));
					mesh.shp.write_move_paraview(path, *timestep, mesh.center, mesh.quat);
				}
			}
		}
	};

	template<class GridT> using DumpSTLMeshTmpl = DumpSTLMesh<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "dump_stl_mesh", make_grid_variant_operator< DumpSTLMeshTmpl > );
	}
}

