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
#include <exaDEM/driver_base.h>
#include <exaDEM/drivers.h>
#include <exaDEM/driver_stl_mesh.h>
#include <exaDEM/shape/shape.hpp>

namespace exaDEM
{

	using namespace exanb;

	template<typename GridT>
		class PushToQuaternionSTLMesh : public OperatorNode
	{
		ADD_SLOT( Drivers , drivers  , INPUT_OUTPUT, REQUIRED , DocString{"List of Drivers"});
    ADD_SLOT( double  , dt       , INPUT, DocString{"dt is the time increment of the timeloop"});

		public:

		inline std::string documentation() const override final
		{
			return R"EOF(
        )EOF";
		}

		inline void execute () override final
		{
      double t = *dt;
			for(size_t id = 0 ; id < drivers->get_size() ; id++)
			{
				if ( drivers->type(id) == DRIVER_TYPE::STL_MESH)
				{
					exaDEM::Stl_mesh& mesh = std::get<exaDEM::Stl_mesh> (drivers->data(id));
					mesh.update_orientation(t);
				}
			}
		}
	};

	template<class GridT> using PushToQuaternionSTLMeshTmpl = PushToQuaternionSTLMesh<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "push_to_quaternion_stl_mesh", make_grid_variant_operator< PushToQuaternionSTLMeshTmpl > );
	}
}
