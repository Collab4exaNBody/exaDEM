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

#include <exaDEM/hooke_force_parameters.h>
#include <exaDEM/compute_hooke_force.h>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/shape/shapes.hpp>
#include <exaDEM/shape/shape_detection.hpp>
#include <exaDEM/shape/shape_detection_driver.hpp>
#include <exaDEM/mutexes.h>
#include <exaDEM/drivers.h>
#include <exaDEM/shape/shapes.hpp>
#include <exaDEM/shape/shape_detection.hpp>
#include <exaDEM/hooke_sphere.h>

namespace exaDEM
{
  using namespace exanb;
  using namespace sphere;

  template<bool sym, typename GridT , class = AssertGridHasFields< GridT, field::_radius >>
    class ComputeHookeInteractionSphere : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet< field::_vrot, field::_arot >;
    static constexpr ComputeFields compute_field_set {};

    ADD_SLOT( GridT                       , grid          , INPUT_OUTPUT , REQUIRED );
    ADD_SLOT( GridCellParticleInteraction , ges           , INPUT_OUTPUT , DocString{"Interaction list"} );
    ADD_SLOT( HookeParams                 , config        , INPUT        , REQUIRED ); // can be re-used for to dump contact network
    ADD_SLOT( HookeParams                 , config_driver , INPUT        , OPTIONAL ); // can be re-used for to dump contact network
    ADD_SLOT( mutexes                     , locks         , INPUT_OUTPUT );
    ADD_SLOT( double                      , dt            , INPUT        , REQUIRED );
    ADD_SLOT( Drivers                     , drivers       , INPUT        , DocString{"List of Drivers"});
    ADD_SLOT( double                      , rcut_max      , INPUT_OUTPUT , 0.0 );
    ADD_SLOT( vector_t<size_t>            , idxs          , INPUT_OUTPUT , DocString{"List of non empty cells"});

    public:

    inline std::string documentation() const override final
    {
      return R"EOF(
                )EOF";
    }

    inline void execute () override final
    {
      Drivers empty;
      Drivers& drvs =  drivers.has_value() ? *drivers : empty;
      const double rcut = config->rcut;
      *rcut_max = std::max( *rcut_max , rcut );

      if( grid->number_of_cells() == 0 ) { return; }

      const auto cells = grid->cells();
      auto & cell_interactions = ges->m_data;
      const HookeParams params = *config;
      const double time = *dt;
      mutexes& locker = *locks;
      auto& indexes = *idxs;
      HookeParams hkp_drvs;
      if ( drivers->get_size() > 0 &&  config_driver.has_value() )
      {
        hkp_drvs = *config_driver;
        *rcut_max = std::max( *rcut_max , hkp_drvs.rcut );
      }


      const hooke_law<sym> sph;
      const exaDEM::sphere::hooke_law_stl stl = {};
      const exaDEM::sphere::hooke_law_driver<Cylinder> cyl;
      const exaDEM::sphere::hooke_law_driver<Surface> surf;
      const exaDEM::sphere::hooke_law_driver<Ball>    ball;
      size_t idxs_size = onika::cuda::vector_size( indexes );
      size_t *idxs_data = onika::cuda::vector_data( indexes ); 

#pragma omp parallel for schedule(dynamic)
      for( size_t ci = 0 ; ci < idxs_size ; ci ++ )
      {
        size_t current_cell = idxs_data[ci];  

        auto& interactions = cell_interactions[current_cell];
        const unsigned int data_size = onika::cuda::vector_size( interactions.m_data );
        exaDEM::Interaction* const __restrict__ data_ptr = onika::cuda::vector_data( interactions.m_data ); 

        for( size_t it = 0; it < data_size ; it++ )
        {
          Interaction& item = data_ptr[it];

          if(item.type == 0) // sphere-sphere
          {
            sph(item, cells, params, time, locker);
          }
          else if(item.type == 4) // cylinder
          {
            cyl(item, cells, drvs, hkp_drvs, time, locker);
          }
          else if(item.type == 5) // surface
          {
            surf(item, cells, drvs, hkp_drvs, time, locker);
          }
          else if(item.type == 6) // ball
          {
            ball(item, cells, drvs, hkp_drvs, time, locker);
          }
          else if(item.type >= 7 && item.type <= 9) // stl
          {
            stl(item, cells, drvs, hkp_drvs, time, locker);
          }
        }
      }
    }
  };

  template<class GridT> using ComputeHookeInteractionSphereSymTmpl = ComputeHookeInteractionSphere<true,GridT>;
  template<class GridT> using ComputeHookeInteractionSphereNoSymTmpl = ComputeHookeInteractionSphere<false,GridT>;

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "hooke_sphere_sym", make_grid_variant_operator< ComputeHookeInteractionSphereSymTmpl > );
    OperatorNodeFactory::instance()->register_factory( "hooke_sphere_no_sym", make_grid_variant_operator< ComputeHookeInteractionSphereNoSymTmpl > );
  }
}

