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
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>

#include <memory>

#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exanb/extra_storage/migration_test.hpp>

namespace exaDEM
{
	using namespace exanb;

	template<typename GridT
		, class = AssertGridHasFields< GridT >
		>
		class CompressInteraction : public OperatorNode
		{
			ADD_SLOT( GridT       , grid              , INPUT_OUTPUT , REQUIRED );
			ADD_SLOT( GridCellParticleInteraction , ges  , INPUT , DocString{"Interaction list"} );

			public:

			inline std::string documentation() const override final
			{
				return R"EOF(
					"This opertor compress interaction by removing inactive interaction. Do not use it if interaction are not rebuilt after."
				        )EOF";
			}

			inline void execute () override final
			{
				if( grid->number_of_cells() == 0 ) { return; }
				auto & cell_interactions = ges->m_data;

				auto save = [] (const exaDEM::Interaction& interaction)
				{
					return interaction.is_active();
				};

#pragma omp parallel for
				for(size_t current_cell = 0 ; current_cell < cell_interactions.size() ; current_cell++)
				{
					auto& storage = cell_interactions[current_cell];
					storage.compress_data(save);
				}
			}
		};

	template<class GridT> using CompressInteractionTmpl = CompressInteraction<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "compress_interaction", make_grid_variant_operator< CompressInteractionTmpl > );
	}
}

