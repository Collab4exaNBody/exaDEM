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
#include <exanb/extra_storage/move_particles_es.hpp>
#include <exanb/grid_cell_particles/move_particles_across_cells.h>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>

namespace exaDEM
{
	using namespace exanb;

	template <typename GridT> using MovePaticlesWithInteractionTmpl = exanb::MovePaticlesWithES <GridT, exaDEM::Interaction>;
  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "move_particles_interaction", make_grid_variant_operator< MovePaticlesWithInteractionTmpl > );
  }
}
