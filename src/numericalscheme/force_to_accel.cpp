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
#include <exanb/compute/compute_cell_particles.h>
#include <memory>
#include <exaDEM/force_to_accel.h>
#include <exaDEM/cell_list_wrapper.hpp>

namespace exaDEM
{
using namespace exanb;

  template<typename GridT
    , class = AssertGridHasFields< GridT, field::_mass, field::_fx,field::_fy,field::_fz >
    >
  class ForceToAccel : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet< field::_mass, field::_fx ,field::_fy ,field::_fz >;
    static constexpr ComputeFields compute_field_set {};

    ADD_SLOT( GridT           , grid      , INPUT_OUTPUT );
    ADD_SLOT( CellListWrapper , cell_list , INPUT , DocString{"list of non empty cells within the current grid"});

  public:

		inline std::string documentation() const override final
		{
			return R"EOF(
        This operator computes particle accelerations from forces and mass.
        )EOF";
		}

		inline void execute () override final
		{
      auto [cell_ptr, cell_size] = cell_list->info();
			ForceToAccelFunctor func {};
			compute_cell_particles( *grid , false , func , compute_field_set , parallel_execution_context() , cell_ptr, cell_size );
		}
	};

	template<class GridT> using ForceToAccelTmpl = ForceToAccel<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "force_to_accel", make_grid_variant_operator< ForceToAccelTmpl > );
	}

}

