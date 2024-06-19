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
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/classifier.hpp>
#include <exaDEM/shape/shapes.hpp>
#include <exaDEM/shape/shape_detection.hpp>
#include <exaDEM/shape/shape_detection_driver.hpp>
#include <exaDEM/mutexes.h>
#include <exaDEM/drivers.h>
#include <exaDEM/hooke_sphere.h>

namespace exaDEM
{
	using namespace exanb;
	using namespace sphere;

	template<bool Sym, typename GridT , class = AssertGridHasFields< GridT, field::_radius >>
		class ComputeHookeClassifierSphere : public OperatorNode
	{
		// attributes processed during computation
		using ComputeFields = FieldSet< field::_vrot, field::_arot >;
		static constexpr ComputeFields compute_field_set {};

		ADD_SLOT( GridT       , grid              , INPUT_OUTPUT , REQUIRED );
		ADD_SLOT( HookeParams , config            , INPUT , REQUIRED ); // can be re-used for to dump contact network
		ADD_SLOT( HookeParams , config_driver     , INPUT , OPTIONAL ); // can be re-used for to dump contact network
		ADD_SLOT( mutexes     , locks             , INPUT_OUTPUT );
		ADD_SLOT( double      , dt                , INPUT , REQUIRED );
		ADD_SLOT( Drivers     , drivers           , INPUT , DocString{"List of Drivers"});
		ADD_SLOT( Classifier  , ic                , INPUT_OUTPUT , DocString{"Interaction lists classified according to their types"} );


		public:

		inline std::string documentation() const override final
		{
			return R"EOF(
                )EOF";
		}

		template<typename... Args>
			inline void run_hooke_law(int id, Classifier& ic, Args&&... args)
			{
				auto [ptr, size] = ic.get_info(id);
				apply_hooke_law(ptr, size, std::forward<Args>(args)...);
			}

		template<typename TMPLC, typename TMPLK, typename... Args>
			inline void apply_hooke_law(exaDEM::Interaction* data_ptr, size_t data_size, TMPLK& kernel, TMPLC* cells, Args&&... args)
			{
#pragma omp parallel for schedule(static)
				for(size_t it = 0 ; it < data_size ; it++)
				{
					exaDEM::Interaction& item = data_ptr[it];
					kernel(item, cells, std::forward<Args>(args)...);
				}
			}


		inline void execute () override final
		{
			if( grid->number_of_cells() == 0 ) { return; }

			Drivers empty;
			Drivers& drvs =  drivers.has_value() ? *drivers : empty;

			const auto cells = grid->cells();
			const HookeParams hkp = *config;
			HookeParams hkp_drvs;

			if ( drivers->get_size() > 0 &&  config_driver.has_value() )
			{
				hkp_drvs = *config_driver;
			}

			const double time = *dt;
			mutexes& locker = *locks;
			auto& classifier = *ic;

			hooke_law<Sym> sph;
			hooke_law_driver<Cylinder> cyli;
			hooke_law_driver<Surface>  surf;
			hooke_law_driver<Ball>     ball;
			hooke_law_stl stlm = {};

			run_hooke_law(0, classifier, sph, cells, hkp, time, locker);	
			run_hooke_law(4, classifier, cyli, cells, drvs, hkp_drvs, time, locker);	
			run_hooke_law(5, classifier, surf, cells, drvs, hkp_drvs, time, locker);	
			run_hooke_law(6, classifier, ball, cells, drvs, hkp_drvs, time, locker);	
			for(int w = 7 ; w <= 9 ; w++)
			{
				run_hooke_law(w, classifier, stlm, cells, drvs, hkp_drvs, time, locker);	
			}
		}
	};

	template<class GridT> using ComputeHookeClassifierSymTmpl = ComputeHookeClassifierSphere<true, GridT>;
	template<class GridT> using ComputeHookeClassifierNoSymTmpl = ComputeHookeClassifierSphere<false, GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "hooke_classifer_sphere_sym", make_grid_variant_operator< ComputeHookeClassifierSymTmpl > );
		OperatorNodeFactory::instance()->register_factory( "hooke_classifer_sphere_no_sym", make_grid_variant_operator< ComputeHookeClassifierNoSymTmpl > );
	}
}

