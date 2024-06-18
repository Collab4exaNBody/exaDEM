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

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <memory>
#include <exaDEM/gravity_force.h>
#include <exaDEM/interactions_PP.h>

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_operators.h>
#include <exanb/core/basic_types_stream.h>
#include <onika/memory/allocator.h> // for ONIKA_ASSUME_ALIGNED macro
#include <exanb/compute/compute_pair_optional_args.h>
#include <vector>
#include <iomanip>

namespace exaDEM
{
  using namespace exanb;
  
 /* template< class GridT > __global__ void GravityForceGPU(GridT* cells,
  							int* cells2,
  							int* cells_size,
  							Vec3d g)
  {
  	//int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	int size = cells_size[blockIdx.x];
  	if(threadIdx.x < size)
  	{
  		//printf("GRAVITY\n");
  		int cell = cells2[blockIdx.x];
  		int p = threadIdx.x;
  		
  		double mass = cells[cell][field::mass][p];
  		
  		cells[cell][field::fx][p]+= g.x * mass;
  		cells[cell][field::fy][p]+= g.y * mass;
  		cells[cell][field::fz][p]+= g.z * mass;
  		
  		
  	}
  }*/

  template<typename GridT
    , class = AssertGridHasFields< GridT, field::_mass, field::_fx,field::_fy,field::_fz >
    >
  class GravityForce : public OperatorNode
  {
    static constexpr Vec3d default_gravity = { 0.0, 0.0, -9.807 };
    // attributes processed during computation
    using ComputeFields = FieldSet< field::_mass, field::_fx ,field::_fy ,field::_fz >;
    static constexpr ComputeFields compute_field_set {};

    ADD_SLOT( GridT  , grid     , INPUT_OUTPUT );
    ADD_SLOT( Vec3d  , gravity  , INPUT , default_gravity , DocString{"define the gravity constant in function of the gravity axis, default value are x axis = 0, y axis = 0 and z axis = -9.807"});
    ADD_SLOT( Interactions_PP       , interactions_PP   , INPUT_OUTPUT );
    //ADD_SLOT( std::vector<int>, cells_non_empty, INPUT_OUTPUT);

  public:

		inline std::string documentation() const override final
		{
			return R"EOF(
        This operator computes forces related to the gravity.
        )EOF";
		}

		inline void execute () override final
		{
			Interactions_PP& ints= *interactions_PP;
			//auto& g = *grid;
			//const auto cells = g.cells();
			//printf("POISON\n");
			GravityForceFunctor func { *gravity};
			//compute_cell_particles2( *grid , false , func , compute_field_set , parallel_execution_context(), ints.cells_gravity_GPU.data(), ints.init_GPU_size );
			compute_cell_particles( *grid , false , func , compute_field_set , parallel_execution_context() );
			//int blockSize = 256;
			//printf("PARTICULES: %d\n", ints.max_cells_gravity_size);
			//int numBlocks = ints.init_GPU_size;
			
			//GravityForceGPU<<<numBlocks, blockSize>>>(cells, ints.cells_gravity_GPU.data(), ints.cells_gravity_size_GPU.data(), *gravity);
		}
	};

	template<class GridT> using GravityForceTmpl = GravityForce<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "gravity_force", make_grid_variant_operator< GravityForceTmpl > );
	}

}

