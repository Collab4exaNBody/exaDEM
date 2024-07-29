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

#include <exaDEM/hooke_force_parameters.h>
#include <exaDEM/compute_hooke_force.h>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/classifier.hpp>
#include <exaDEM/interaction/classifier_for_all.hpp>
#include <exaDEM/interaction/classifier_analyses.hpp>
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

  template<typename GridT , class = AssertGridHasFields< GridT, field::_radius >>
    class ComputeHookeClassifierSphereGPU : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet< field::_vrot, field::_arot >;
    static constexpr ComputeFields compute_field_set {};

    ADD_SLOT( GridT       , grid              , INPUT_OUTPUT , REQUIRED );
    ADD_SLOT( HookeParams , config            , INPUT , REQUIRED , DocString{"Hooke parameters for sphere interactions"}); // can be re-used for to dump contact network
    ADD_SLOT( HookeParams , config_driver     , INPUT , OPTIONAL , DocString{"Hooke parameters for drivers, optional"}); // can be re-used for to dump contact network
    ADD_SLOT( double      , dt                , INPUT , REQUIRED , DocString{"Time step value"});
    ADD_SLOT( bool        , symetric          , INPUT , REQUIRED , DocString{"Activate the use of symetric feature (contact law)"});
    ADD_SLOT( Drivers     , drivers           , INPUT , DocString{"List of Drivers {Cylinder, Surface, Ball, Mesh}"});
    ADD_SLOT( Classifier  , ic                , INPUT_OUTPUT , DocString{"Interaction lists classified according to their types"} );
    // analysis
    ADD_SLOT( long        , timestep          , INPUT , REQUIRED );
    ADD_SLOT( long        , analysis_interaction_dump_frequency , INPUT , REQUIRED, DocString{"Write an interaction dump file"});


    public:

    inline std::string documentation() const override final
    {
      return R"EOF(This operators compute forces between particles and particles/drivers using the Hooke's law.)EOF";
    }

    inline void execute () override final
    {
      if( grid->number_of_cells() == 0 ) { return; }

      /** Analysis */
      const long frequency = *analysis_interaction_dump_frequency;
      bool write_interactions = ( frequency > 0 && (*timestep) % frequency == 0);

      /** Get Driver */
      Drivers* drvs =  drivers.get_pointer();
      auto* cells = grid->cells();
      const HookeParams hkp = *config;
      HookeParams hkp_drvs{};

      if ( drivers->get_size() > 0 &&  config_driver.has_value() )
      {
        hkp_drvs = *config_driver;
      }

      const double time = *dt;
      auto& classifier = *ic;

      hooke_law_driver<Cylinder> cyli;
      hooke_law_driver<Surface>  surf;
      hooke_law_driver<Ball>     ball;
      hooke_law_stl stlm = {};

      if(*symetric)
			{
        hooke_law<true> sph;
        run_contact_law(parallel_execution_context(), 0, classifier, sph, write_interactions, cells, hkp, time);  
      }
      else
      {
        hooke_law<false> sph;
        run_contact_law(parallel_execution_context(), 0, classifier, sph, write_interactions, cells, hkp, time);  
      }
      run_contact_law(parallel_execution_context(), 4, classifier, cyli, write_interactions, cells, drvs->ptr<Cylinder>(), hkp_drvs, time);  
      run_contact_law(parallel_execution_context(), 5, classifier, surf, write_interactions, cells, drvs->ptr<Surface>(), hkp_drvs, time);  
      run_contact_law(parallel_execution_context(), 6, classifier, ball, write_interactions, cells, drvs->ptr<Ball>(), hkp_drvs, time);  
      for(int type = 7 ; type <= 9 ; type++)
      {
        run_contact_law(parallel_execution_context(), type, classifier, stlm, write_interactions, cells, drvs, hkp_drvs, time);  
      }

      if(write_interactions)
      {
        auto stream = create_buffer(*grid, classifier);
        std::string ts = std::to_string(*timestep);
        write_file(stream, "Interaction_" + ts);        
      }
    }
  };

  template<class GridT> using ComputeHookeClassifierGPUTmpl = ComputeHookeClassifierSphereGPU<GridT>;

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "hooke_sphere", make_grid_variant_operator< ComputeHookeClassifierGPUTmpl > );
  }
}

