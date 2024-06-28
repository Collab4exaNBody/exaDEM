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

#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/particle_neighbors/chunk_neighbors_apply.h>

#include <memory>

#include <exaDEM/hooke_force_parameters.h>
#include <exaDEM/compute_hooke_force.h>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/shape/shapes.hpp>
#include <exaDEM/shape/shape_detection.hpp>
#include <exaDEM/shape/shape_detection_driver.hpp>
#include <exaDEM/mutexes.h>
#include <exaDEM/drivers.h>
#include <exaDEM/hooke_polyhedron.h>

namespace exaDEM
{
  using namespace exanb;
	using namespace polyhedron;

  template<typename GridT , class = AssertGridHasFields< GridT, field::_radius >>
    class ComputeHookeInteraction : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet< field::_vrot, field::_arot >;
    static constexpr ComputeFields compute_field_set {};

    ADD_SLOT( GridT       , grid                , INPUT_OUTPUT , REQUIRED );
    ADD_SLOT( GridCellParticleInteraction , ges , INPUT_OUTPUT , DocString{"Interaction list"} );
    ADD_SLOT( shapes      , shapes_collection   , INPUT_OUTPUT , DocString{"Collection of shapes"});
		ADD_SLOT( mutexes     , locks               , INPUT_OUTPUT );
    ADD_SLOT( HookeParams , config              , INPUT , REQUIRED , DocString{"Hooke law parameters used to model interactions sphere/driver"}); // can be re-used for to dump contact network
    ADD_SLOT( HookeParams , config_driver       , INPUT , OPTIONAL , DocString{"Hooke law parameters used to model interactions sphere/driver"}); // can be re-used for to dump contact network
    ADD_SLOT( double      , dt                  , INPUT , REQUIRED , DocString{"Timestep"});
    ADD_SLOT( Drivers     , drivers             , INPUT , DocString{"List of Drivers"});
		ADD_SLOT( std::vector<size_t> , idxs        , INPUT_OUTPUT , DocString{"List of non empty cells"});

		public:

		inline std::string documentation() const override final
		{
			return R"EOF(
                  Apply Hooke's law between spheres and between spheres with drivers. This operator requires interactions to have been calculated using the nbh_sphere_sym or nbh_spere_no_sym operators.
                )EOF";
		}

		inline void execute () override final
		{
			if( grid->number_of_cells() == 0 ) { return; }

			Drivers empty;
			Drivers& drvs =  drivers.has_value() ? *drivers : empty;

			const auto cells = grid->cells();
			auto & cell_interactions = ges->m_data;
			auto & shps = *shapes_collection;
			const HookeParams params = *config;
			HookeParams hkp_drvs;
			const double time = *dt;
			mutexes& locker = *locks;
			auto& indexes = *idxs;

			if ( drivers->get_size() > 0 &&  config_driver.has_value() )
			{
				hkp_drvs = *config_driver;
			}


			const hooke_law poly;
			const hooke_law_driver<Cylinder> cyli;
			const hooke_law_driver<Surface>  surf;
			const hooke_law_driver<Ball>     ball;
			const hooke_law_stl stlm = {};

#pragma omp parallel for schedule(dynamic)
			for( size_t ci = 0 ; ci < indexes.size() ; ci ++ )
			{
				size_t current_cell = indexes[ci];  

				auto& interactions = cell_interactions[current_cell];
				const unsigned int data_size = onika::cuda::vector_size( interactions.m_data );
				exaDEM::Interaction* const __restrict__ data_ptr = onika::cuda::vector_data( interactions.m_data ); 

				for( size_t it = 0; it < data_size ; it++ )
				{
					Interaction& item = data_ptr[it];

					if( item.type < 4 ) // polyhedra
					{
						poly(item, cells, params, shps, time, locker);
					}
					else if(item.type == 4) // cylinder
					{
						cyli(item, cells, drvs, hkp_drvs, shps, time, locker);
					}
					else if( item.type == 5) // wall
					{
						surf(item, cells, drvs, hkp_drvs, shps, time, locker);
					}
					else if(item.type == 6) // sphere
					{
						ball(item, cells, drvs, hkp_drvs, shps, time, locker);	
					}
					else if(item.type >= 7 && item.type <= 12) // stl
					{
						stlm(item, cells, drvs, hkp_drvs, shps, time, locker);
					}
				}
			}
		}
	};

	template<class GridT> using ComputeHookeInteractionTmpl = ComputeHookeInteraction<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "hooke_polyhedron_v2", make_grid_variant_operator< ComputeHookeInteractionTmpl > );
	}
}

